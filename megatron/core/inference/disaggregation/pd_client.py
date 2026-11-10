# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prefill/decode router client for disaggregated serving behind the HTTP frontend.

:class:`PDInferenceClient` wraps two :class:`InferenceClient` objects, one per pool's
coordinator, and exposes the request API the OpenAI endpoints use
(``add_request_streaming`` / ``add_request_with_id`` / ``abort_request`` / ``get_metrics``),
so chat templates, parsers and detokenization stay unchanged.

Each request runs in two hops:

1. Prefill: the prompt goes to the prefill pool with ``num_tokens_to_generate=1`` and
   ``do_kv_handoff=True``. The prefill engine samples the first token, pins the prompt's KV
   blocks (and, for hybrid models, its live SSM state) and replies with
   ``disaggregated_params = {request_id, block_ids, kv_meta}``.
2. Decode: the prompt, the caller's sampling parameters and that metadata go to the decode pool
   (``SUBMIT_REQUEST_WITH_KV``). The decode engine pulls the KV, resumes from the prefill's
   first token (stopping at once if it is EOS) and generates the rest.

The decode hop always streams internally: its first partial is sent as soon as the import is
admitted, which is when the prefill's pinned blocks can be released (``RELEASE_KV``). For a
caller that did not ask to stream, the decode hop uses a very large streaming interval, so it
sends that one early partial and then only the final reply. Blocks are also released when the
request fails or is aborted, so the prefill pool cannot leak them.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.sampling_params import SamplingParams

logger = logging.getLogger(__name__)

# Streaming interval for the decode hop of a non-streaming request: large enough that only the
# immediate first partial (the release signal) and the final reply are sent.
_NON_STREAMING_DECODE_INTERVAL = 1 << 30


class PDInferenceClient:
    """Routes each request through a prefill pool, then a decode pool.

    Args:
        prefill_client: Client of the prefill pool's coordinator.
        decode_client: Client of the decode pool's coordinator.
    """

    def __init__(self, prefill_client: InferenceClient, decode_client: InferenceClient):
        self.prefill = prefill_client
        self.decode = decode_client
        self._ids = itertools.count()
        self._tasks: Dict[int, asyncio.Task] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def from_addresses(
        cls, prefill_coordinator_addr: str, decode_coordinator_addr: str, **client_kwargs
    ) -> "PDInferenceClient":
        """Build both pool clients with the same options (e.g. block_size_tokens)."""
        return cls(
            InferenceClient(prefill_coordinator_addr, **client_kwargs),
            InferenceClient(decode_coordinator_addr, **client_kwargs),
        )

    # --- lifecycle -------------------------------------------------------------------
    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None, **kwargs) -> None:
        """Connect to both coordinators."""
        self.prefill.start(loop=loop, **kwargs)
        self.decode.start(loop=loop, **kwargs)
        self._loop = self.decode._loop

    def stop(self) -> None:
        """Cancel in-flight routing tasks and stop both clients."""
        for task in list(self._tasks.values()):
            task.cancel()
        self.prefill.stop()
        self.decode.stop()

    # --- request API used by the HTTP endpoints ----------------------------------------
    def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        *,
        multi_modal_data=None,
        offload_params: Optional[dict] = None,
    ) -> asyncio.Future:
        """Submit a request; the future resolves to the final reply dict."""
        return self.add_request_with_id(
            prompt, sampling_params, multi_modal_data=multi_modal_data, offload_params=offload_params
        )[1]

    def add_request_with_id(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        *,
        multi_modal_data=None,
        offload_params: Optional[dict] = None,
    ) -> Tuple[int, asyncio.Future]:
        """Submit a request; returns its router id (for abort_request) and final-reply future."""
        self._check_supported(multi_modal_data, offload_params)
        request_id = next(self._ids)
        task = asyncio.get_running_loop().create_task(
            self._route(request_id, prompt, sampling_params, stream=None)
        )
        self._track(request_id, task)
        return request_id, task

    def add_request_streaming(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        *,
        multi_modal_data=None,
        offload_params: Optional[dict] = None,
    ) -> AsyncStream[dict]:
        """Submit a streaming request: ``{"partial": ...}`` items, then one ``{"final": ...}``."""
        self._check_supported(multi_modal_data, offload_params)
        request_id = next(self._ids)
        stream: AsyncStream[dict] = AsyncStream(
            request_id, lambda: self.abort_request(request_id), loop=self._loop
        )
        task = asyncio.get_running_loop().create_task(
            self._route(request_id, prompt, sampling_params, stream=stream)
        )
        self._track(request_id, task)
        return stream

    def abort_request(self, request_id: int) -> None:
        """Cancel a request in whichever hop it is in; its prefill blocks are released."""
        task = self._tasks.get(int(request_id))
        if task is not None and not task.done():
            task.cancel()

    async def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Both pools' engine metrics, keyed ``prefill/<engine>`` and ``decode/<engine>``.

        A prefill engine finishes every request after one token, so its finished-request
        counter is reported as ``prefill_handoffs_total`` instead, keeping
        ``requests_finished_total`` a count of user requests.
        """
        prefill, decode = await asyncio.gather(self.prefill.get_metrics(), self.decode.get_metrics())
        merged: Dict[str, Dict[str, Any]] = {}
        for engine, values in prefill.items():
            values = dict(values)
            values["prefill_handoffs_total"] = values.pop("requests_finished_total", 0)
            merged[f"prefill/{engine}"] = values
        for engine, values in decode.items():
            merged[f"decode/{engine}"] = values
        return merged

    # Control signals go to both pools.
    def _both(self, method: str, *args) -> None:
        getattr(self.prefill, method)(*args)
        getattr(self.decode, method)(*args)

    def pause_engines(self):
        """Pause both pools."""
        self._both("pause_engines")

    def unpause_engines(self) -> None:
        """Unpause both pools."""
        self._both("unpause_engines")

    def start_cuda_profiler(self) -> None:
        """Start the CUDA profiler on both pools."""
        self._both("start_cuda_profiler")

    def stop_cuda_profiler(self) -> None:
        """Stop the CUDA profiler on both pools."""
        self._both("stop_cuda_profiler")

    def stop_engines(self):
        """Stop both pools' engines."""
        self._both("stop_engines")

    def shutdown_coordinator(self):
        """Shut down both coordinators."""
        self._both("shutdown_coordinator")

    # --- routing ---------------------------------------------------------------------
    @staticmethod
    def _check_supported(multi_modal_data, offload_params) -> None:
        if multi_modal_data:
            raise NotImplementedError("disaggregated serving does not support multimodal inputs")
        if offload_params:
            raise NotImplementedError("disaggregated serving does not support offload_params")

    def _track(self, request_id: int, task: asyncio.Task) -> None:
        self._tasks[request_id] = task

        def _done(done: asyncio.Task) -> None:
            self._tasks.pop(request_id, None)
            # Streaming callers get errors through their stream; retrieve the task's own
            # copy so asyncio does not report it as never retrieved.
            if not done.cancelled():
                done.exception()

        task.add_done_callback(_done)

    @staticmethod
    def prefill_sampling_params(sampling_params: SamplingParams) -> SamplingParams:
        """The prefill hop's parameters: one token, KV hand-off, no streaming or log-probs."""
        params = copy.deepcopy(sampling_params)
        params.num_tokens_to_generate = 1
        params.num_tokens_total = None
        params.do_kv_handoff = True
        params.streaming = False
        params.return_log_probs = False
        params.top_n_logprobs = 0
        params.return_prompt_top_n_logprobs = False
        return params

    def _release(self, prefill_engine_request_id: int) -> None:
        try:
            self.prefill.release_handoff(prefill_engine_request_id)
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "PD: failed to release prefill request %d", prefill_engine_request_id, exc_info=True
            )

    def _release_when_done(self, future: asyncio.Future) -> None:
        """Release a prefill hop whose caller went away once its reply (if any) arrives."""

        def _on_done(fut: asyncio.Future) -> None:
            if fut.cancelled() or fut.exception() is not None:
                return
            params = (fut.result() or {}).get("disaggregated_params")
            if params:
                self._release(int(params["request_id"]))

        future.add_done_callback(_on_done)

    async def _route(
        self,
        request_id: int,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        stream: Optional[AsyncStream],
    ) -> dict:
        start = time.perf_counter()
        try:
            if sampling_params.return_log_probs or sampling_params.top_n_logprobs > 0:
                raise NotImplementedError(
                    "disaggregated serving does not transfer log-probs; request them from an "
                    "aggregated server"
                )
            final = await self._route_hops(request_id, prompt, sampling_params, stream, start)
        except BaseException as error:
            if stream is not None:
                stream.finish(error if isinstance(error, Exception) else None)
            raise
        if stream is not None:
            stream.put({"final": final})
            stream.finish()
        return final

    async def _route_hops(
        self,
        request_id: int,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        stream: Optional[AsyncStream],
        start: float,
    ) -> dict:
        # Hop 1: prefill. A cancelled caller must not strand the pinned blocks, so the
        # prefill is shielded and released whenever its reply lands.
        _, prefill_future = self.prefill.add_request_with_id(
            prompt, self.prefill_sampling_params(sampling_params)
        )
        try:
            prefill_reply = await asyncio.shield(prefill_future)
        except asyncio.CancelledError:
            self._release_when_done(prefill_future)
            raise
        prefill_latency = time.perf_counter() - start

        params = prefill_reply.get("disaggregated_params")
        if prefill_reply.get("status") == "FAILED" or not params:
            # No hand-off to resume (failed prefill): the prefill reply is the answer.
            if params:
                self._release(int(params["request_id"]))
            prefill_reply["latency"] = prefill_latency
            return prefill_reply
        prefill_engine_request_id = int(params["request_id"])
        if sampling_params.num_tokens_to_generate == 1:
            self._release(prefill_engine_request_id)
            prefill_reply.pop("disaggregated_params", None)
            prefill_reply["latency"] = prefill_latency
            return prefill_reply

        # Hop 2: decode, always streamed internally so the first partial can release the
        # prefill's blocks as soon as the import is admitted.
        decode_params = copy.copy(sampling_params)
        decode_params.do_kv_handoff = False
        if stream is None:
            decode_params.streaming_interval = _NON_STREAMING_DECODE_INTERVAL
        released = False
        decode_stream = self.decode.add_request_with_kv_handoff_streaming(
            prompt, decode_params, params["kv_meta"], params["block_ids"]
        )
        final = None
        try:
            async for item in decode_stream:
                if not released:
                    self._release(prefill_engine_request_id)
                    released = True
                if "final" in item:
                    final = item["final"]
                elif stream is not None:
                    stream.put(item)
        finally:
            if not released:
                self._release(prefill_engine_request_id)
            if final is None:
                await decode_stream.aclose()
        if final is None:
            raise RuntimeError(f"PD request {request_id}: decode stream ended without a reply")
        final.pop("disaggregated_params", None)
        # The decode client timed only its own hop; report the whole request, and the
        # prefill hop as the time to the first token.
        final["latency"] = time.perf_counter() - start
        final["ttft"] = prefill_latency
        return final
