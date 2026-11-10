# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""PDInferenceClient routing with fake pool clients: the prefill hop's parameters, release of
the prefill's pinned blocks (on decode's first reply, on abort, on failure), streaming and
non-streaming replies, and merged metrics."""

import asyncio

import pytest

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.disaggregation.pd_client import (
    _NON_STREAMING_DECODE_INTERVAL,
    PDInferenceClient,
)
from megatron.core.inference.sampling_params import SamplingParams

PROMPT = [1, 2, 3, 4, 5]


class FakePrefill:
    def __init__(self, reply=None, delay=0.0):
        self.reply = reply or {
            "status": "COMPLETED",
            "generated_tokens": [7],
            "disaggregated_params": {"request_id": 42, "block_ids": [3, 4], "kv_meta": {"m": 1}},
        }
        self.delay = delay
        self.submitted = []
        self.released = []

    def add_request_with_id(self, prompt, sampling_params):
        self.submitted.append((prompt, sampling_params))
        future = asyncio.get_running_loop().create_future()

        async def _finish():
            await asyncio.sleep(self.delay)
            future.set_result(dict(self.reply))

        asyncio.get_running_loop().create_task(_finish())
        return len(self.submitted) - 1, future

    def release_handoff(self, request_id):
        self.released.append(request_id)

    async def get_metrics(self):
        return {"p0": {"generation_tokens_total": 3, "requests_finished_total": 3}}


class FakeDecode:
    def __init__(self, prefill, items=None, hang=False):
        self.prefill = prefill
        self.items = items
        self.hang = hang
        self.submitted = []
        self.aborted = []
        self.released_at_first_item = None

    def add_request_with_kv_handoff_streaming(self, prompt, sampling_params, kv_meta, block_ids):
        self.submitted.append((prompt, sampling_params, kv_meta, block_ids))
        stream = AsyncStream(len(self.submitted) - 1, lambda: self.aborted.append(True))
        items = self.items or [
            {"partial": {"request_id": 0, "new_tokens": [7]}},
            {"partial": {"request_id": 0, "new_tokens": [8, 9]}},
            {"final": {"status": "COMPLETED", "generated_tokens": [7, 8, 9], "latency": 0.0}},
        ]

        async def _feed():
            await asyncio.sleep(0)
            for i, item in enumerate(items):
                if i == 1:
                    self.released_at_first_item = list(self.prefill.released)
                stream.put(item)
                await asyncio.sleep(0.01)  # let the router consume it
            if not self.hang:
                stream.finish()

        asyncio.get_running_loop().create_task(_feed())
        return stream

    async def get_metrics(self):
        return {"d0": {"generation_tokens_total": 30, "requests_finished_total": 3}}


def _client(prefill=None, decode_kwargs=None):
    prefill = prefill or FakePrefill()
    decode = FakeDecode(prefill, **(decode_kwargs or {}))
    client = PDInferenceClient(prefill, decode)
    return client, prefill, decode


def test_streaming_request_routes_prefill_then_decode():
    async def run():
        client, prefill, decode = _client()
        params = SamplingParams(num_tokens_to_generate=64, streaming_interval=4)
        items = [item async for item in client.add_request_streaming(PROMPT, params)]
        return client, prefill, decode, params, items

    client, prefill, decode, params, items = asyncio.run(run())

    prefill_params = prefill.submitted[0][1]
    assert prefill_params.num_tokens_to_generate == 1 and prefill_params.do_kv_handoff
    assert not prefill_params.streaming
    _, decode_params, kv_meta, block_ids = decode.submitted[0]
    assert (kv_meta, block_ids) == ({"m": 1}, [3, 4])
    assert decode_params.num_tokens_to_generate == 64 and decode_params.streaming_interval == 4
    assert params.num_tokens_to_generate == 64 and not params.do_kv_handoff  # caller's untouched
    assert [list(i) for i in items] == [["partial"], ["partial"], ["final"]]
    assert items[-1]["final"]["generated_tokens"] == [7, 8, 9]
    assert "disaggregated_params" not in items[-1]["final"]
    assert prefill.released == [42]
    assert decode.released_at_first_item == [42]  # released on decode's first reply


def test_non_streaming_request_uses_one_early_decode_partial():
    async def run():
        client, prefill, decode = _client()
        _, future = client.add_request_with_id(PROMPT, SamplingParams(num_tokens_to_generate=8))
        return prefill, decode, await future

    prefill, decode, final = asyncio.run(run())
    assert decode.submitted[0][1].streaming_interval == _NON_STREAMING_DECODE_INTERVAL
    assert final["generated_tokens"] == [7, 8, 9]
    assert final["latency"] > 0 and "ttft" in final
    assert prefill.released == [42]


def test_single_token_request_finishes_at_prefill():
    async def run():
        client, prefill, decode = _client()
        _, future = client.add_request_with_id(PROMPT, SamplingParams(num_tokens_to_generate=1))
        return prefill, decode, await future

    prefill, decode, final = asyncio.run(run())
    assert decode.submitted == []
    assert final["generated_tokens"] == [7]
    assert prefill.released == [42]


def test_failed_prefill_is_returned_without_decode():
    async def run():
        prefill = FakePrefill(reply={"status": "FAILED", "generated_tokens": []})
        client, prefill, decode = _client(prefill)
        _, future = client.add_request_with_id(PROMPT, SamplingParams(num_tokens_to_generate=8))
        return prefill, decode, await future

    prefill, decode, final = asyncio.run(run())
    assert final["status"] == "FAILED"
    assert decode.submitted == [] and prefill.released == []


def test_abort_during_decode_releases_and_aborts_decode():
    async def run():
        client, prefill, decode = _client(
            decode_kwargs={"items": [{"partial": {"request_id": 0, "new_tokens": [7]}}], "hang": True}
        )
        request_id, future = client.add_request_with_id(
            PROMPT, SamplingParams(num_tokens_to_generate=8)
        )
        while not decode.submitted:
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.01)
        client.abort_request(request_id)
        with pytest.raises(asyncio.CancelledError):
            await future
        return prefill, decode

    prefill, decode = asyncio.run(run())
    assert prefill.released == [42]
    assert decode.aborted == [True]


def test_abort_during_prefill_releases_when_prefill_lands():
    async def run():
        client, prefill, decode = _client(FakePrefill(delay=0.05))
        request_id, future = client.add_request_with_id(
            PROMPT, SamplingParams(num_tokens_to_generate=8)
        )
        await asyncio.sleep(0.01)
        client.abort_request(request_id)
        with pytest.raises(asyncio.CancelledError):
            await future
        assert prefill.released == []
        await asyncio.sleep(0.1)
        return prefill, decode

    prefill, decode = asyncio.run(run())
    assert prefill.released == [42]
    assert decode.submitted == []


def test_log_probs_are_rejected():
    async def run():
        client, _, _ = _client()
        _, future = client.add_request_with_id(
            PROMPT, SamplingParams(num_tokens_to_generate=8, return_log_probs=True)
        )
        await future

    with pytest.raises(NotImplementedError):
        asyncio.run(run())


def test_metrics_are_merged_by_pool():
    async def run():
        client, _, _ = _client()
        return await client.get_metrics()

    merged = asyncio.run(run())
    assert merged == {
        "prefill/p0": {"generation_tokens_total": 3, "prefill_handoffs_total": 3},
        "decode/d0": {"generation_tokens_total": 30, "requests_finished_total": 3},
    }
