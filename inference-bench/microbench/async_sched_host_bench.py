# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Host (CPU) cost of the async-scheduling context calls versus active request count.

Builds a DynamicInferenceContext for a tiny hybrid (Mamba + attention) model, fills N
active decode requests, and replays the context calls of an async-scheduled decode step
without a model forward:

    can_prepare_requests -> prepare_requests -> initialize_attention_state (no H2D)
    -> transfer_bookkeeping_to_gpu (+ wait for the copy) -> resolve_requests
    -> commit_sampled_tokens

Each call is timed with perf_counter (CPU wall clock); the only GPU wait is the H2D copy's
event, timed as its own row. Model size does not matter for these calls, so the tiny model
stands in for nano / Super; what matters is N, the block size and the KV pool size (the LRU
prefix cache scans the whole pool).

Steady state: no request finishes (resolve keeps every row), so resolve_requests measures
its no-finish path, and requests cross a KV block boundary at the real rate (each row's
offset in its last block is random, so about N / block_size rows need a new block per step).

Run on one GPU:
    python inference-bench/microbench/async_sched_host_bench.py --counts 1 32 128 256 512 1024
"""

import argparse
import cProfile
import io
import json
import pstats
import statistics
import time

import torch

from megatron.core import parallel_state
from megatron.core.inference.config import (
    InferenceConfig,
    MambaInferenceStateConfig,
    PrefixCachingEvictionPolicy,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.transformer.transformer_config import TransformerConfig

OPS = (
    "can_prepare_requests",
    "prepare_requests",
    "initialize_attention_state",
    "transfer_bookkeeping_to_gpu",
    "wait_h2d",
    "resolve_requests",
    "commit_sampled_tokens",
)


def build_context(args, max_requests):
    layer_type_list = [Symbols.MAMBA, Symbols.ATTENTION]
    mamba_config = MambaInferenceStateConfig(
        layer_type_list, (544, 4), (8, 64, 16), torch.bfloat16, torch.bfloat16
    )
    return DynamicInferenceContext(
        model_config=TransformerConfig(
            params_dtype=torch.bfloat16, num_layers=2, kv_channels=8, num_attention_heads=2
        ),
        inference_config=InferenceConfig(
            max_sequence_length=args.max_sequence_length,
            num_cuda_graphs=args.num_cuda_graphs,
            buffer_size_gb=args.buffer_gb,
            block_size_tokens=args.block_size,
            max_tokens=max(args.max_tokens, max_requests),
            max_requests=max_requests,
            mamba_inference_state_config=mamba_config,
            unified_memory_level=0,
            enable_prefix_caching=args.prefix_caching,
            prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
            prefix_caching_mamba_gb=(
                args.mamba_prefix_gb if args.prefix_caching and args.mamba_prefix_gb else None
            ),
        ),
    )


def fill_decode_rows(ctx, n, block_size, generator):
    """N active decode rows, each owning one KV block, at random offsets in that block."""
    ctx.total_request_count = n
    ctx.paused_request_count = 0
    ctx.num_prefill_requests = 0
    ctx.active_token_count = n
    rows = slice(0, n)
    last_offsets = torch.randint(0, block_size - 1, (n,), generator=generator, dtype=torch.int32)
    ctx.request_ids[rows] = torch.arange(n, dtype=torch.int32)
    ctx.request_in_prefill_status_tensor[rows] = 0
    ctx.request_query_lengths[rows] = 1
    ctx.request_output_lengths[rows] = ctx.max_sequence_length
    ctx.request_kv_length_offsets[rows] = last_offsets
    ctx.request_last_kv_block_offset[rows] = last_offsets
    block_ids = ctx.kv_block_allocator.allocate_memory_blocks(n)
    assert block_ids is not None, "KV pool too small for this request count"
    ctx.request_to_kv_block_ids[rows, 0] = block_ids
    ctx.request_last_kv_block_id[rows] = block_ids
    ctx.request_kv_block_counts[rows] = 1
    ctx.token_to_input_ids[rows] = torch.arange(n, dtype=torch.long)
    ctx.token_to_pos_ids[rows] = ctx.request_kv_length_offsets[rows]
    ctx.token_to_request_idx[rows] = torch.arange(n, dtype=torch.int32)
    ctx.token_to_position_in_request[rows] = ctx.token_to_pos_ids[rows]
    ctx.token_to_block_idx[rows] = ctx.request_last_kv_block_id[rows]
    ctx.token_to_local_position_within_kv_block[rows] = ctx.token_to_pos_ids[rows] % block_size
    slots = ctx.mamba_metadata.batch_allocate_slots(n)
    assert slots is not None, "Mamba state pool too small for this request count"
    ctx.mamba_metadata.request_to_mamba_state_idx[rows] = slots


def run_count(args, n):
    ctx = build_context(args, max_requests=max(n, args.min_max_requests))
    generator = torch.Generator().manual_seed(0)
    fill_decode_rows(ctx, n, args.block_size, generator)
    times = {op: [] for op in OPS}
    keep_all = torch.ones(n, dtype=torch.int32)
    sampled = torch.arange(n, dtype=torch.int64)

    def timed(op, fn):
        start = time.perf_counter()
        result = fn()
        times[op].append(time.perf_counter() - start)
        return result

    for step in range(args.warmup + args.steps):
        if step == args.warmup:
            times = {op: [] for op in OPS}
        ok = timed("can_prepare_requests", ctx.can_prepare_requests)
        assert ok, f"cannot prepare at step {step} (N={n}); raise --buffer-gb or lower --steps"
        timed("prepare_requests", ctx.prepare_requests)
        timed(
            "initialize_attention_state",
            lambda: ctx.initialize_attention_state(transfer_bookkeeping_to_gpu=False),
        )
        done = timed(
            "transfer_bookkeeping_to_gpu",
            lambda: ctx.transfer_bookkeeping_to_gpu(
                skip_token_input_ids=True, record_done_event=True
            ),
        )
        timed("wait_h2d", lambda: done.synchronize() if done is not None else None)
        timed("resolve_requests", lambda: ctx.resolve_requests(keep_all))
        timed("commit_sampled_tokens", lambda: ctx.commit_sampled_tokens(sampled))
    torch.cuda.synchronize()

    if n in args.profile_counts:
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(50):
            ctx.prepare_requests()
            ctx.initialize_attention_state(transfer_bookkeeping_to_gpu=False)
            done = ctx.transfer_bookkeeping_to_gpu(skip_token_input_ids=True, record_done_event=True)
            done.synchronize()
            ctx.resolve_requests(keep_all)
            ctx.commit_sampled_tokens(sampled)
        profiler.disable()
        out = io.StringIO()
        pstats.Stats(profiler, stream=out).sort_stats("tottime").print_stats(18)
        print(f"===== cProfile N={n}, 50 steps, by tottime =====")
        print("\n".join(out.getvalue().splitlines()[:40]))

    row = {"active_requests": n, "kv_pool_blocks": ctx.kv_block_allocator.pool_size}
    for op, samples in times.items():
        row[f"{op}_p50_us"] = round(statistics.median(samples) * 1e6, 1)
        row[f"{op}_p90_us"] = round(sorted(samples)[int(0.9 * (len(samples) - 1))] * 1e6, 1)
    row["total_p50_us"] = round(sum(row[f"{op}_p50_us"] for op in OPS), 1)
    del ctx
    torch.cuda.empty_cache()
    return row


ADMIT_OPS = (
    "prompt_to_tensor",  # engine.add_request: prompt list -> tensor
    "build_request",  # DynamicInferenceRequest(...): sampling-param copy + block hashes
    "check_availability_1",  # scheduler: _can_schedule_chunked_prefill
    "prefix_skip_match",  # scheduler: _compute_prefix_match for the prefix skip
    "check_availability_2",  # scheduler: check_availability(req, chunk)
    "add_request",  # context.add_request, total
    "add_request/allocate_memory_blocks",
    "add_request/init_value_pages",
    "add_request/rest",  # add_request minus the two sections above
    "flush_value_pages",  # batched_value_pages: the one deferred zeroing launch, per request
)


_ADMIT_PROFILER = cProfile.Profile()


class _SectionTimer:
    """Wraps a bound method so its wall time accumulates into `totals[name]`."""

    def __init__(self, totals, name, fn):
        self.totals, self.name, self.fn = totals, name, fn

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return self.fn(*args, **kwargs)
        finally:
            self.totals[self.name] = self.totals.get(self.name, 0.0) + time.perf_counter() - start


def run_admission(args, n_resident, k_admit, prompt_device):
    """Admit K requests on top of N resident decode rows, replaying the engine's per-request
    calls; returns median microseconds per admitted request for each section."""
    per_request = {op: [] for op in ADMIT_OPS}
    vocab = 100_000
    for trial in range(args.admit_warmup + args.admit_trials):
        ctx = build_context(args, max_requests=max(n_resident + k_admit, args.min_max_requests))
        generator = torch.Generator().manual_seed(trial)
        if n_resident:
            fill_decode_rows(ctx, n_resident, args.block_size, generator)
        nested = {}
        allocator = ctx.kv_block_allocator
        allocator.allocate_memory_blocks = _SectionTimer(
            nested, "add_request/allocate_memory_blocks", allocator.allocate_memory_blocks
        )
        ctx._initialize_mha_value_pages = _SectionTimer(
            nested, "add_request/init_value_pages", ctx._initialize_mha_value_pages
        )
        prompts = [
            torch.randint(0, vocab, (args.prompt_len,), generator=generator).tolist()
            for _ in range(k_admit)
        ]
        # The engine sets termination_id (tokenizer eod) in _add_request; match it so
        # tracked_metadata does not take its missing-termination_id warning path.
        sampling = SamplingParams(num_tokens_to_generate=256, termination_id=2)
        torch.cuda.synchronize()
        step = {op: 0.0 for op in ADMIT_OPS}
        for i, prompt in enumerate(prompts):
            t0 = time.perf_counter()
            tokens = torch.tensor(prompt, dtype=torch.int64, device=prompt_device)
            t1 = time.perf_counter()
            req = DynamicInferenceRequest(
                request_id=10_000 + i,
                prompt_tokens=tokens,
                sampling_params=sampling,
                enable_prefix_caching=args.prefix_caching,
                block_size_tokens=args.block_size,
            )
            t2 = time.perf_counter()
            ok = all(ctx.check_availability(req, prefill_chunk_length=1))
            t3 = time.perf_counter()
            if args.prefix_caching:
                ctx._compute_prefix_match(req, req.remaining_prompt_length)
            t4 = time.perf_counter()
            ok = ok and all(ctx.check_availability(req, req.remaining_prompt_length))
            t5 = time.perf_counter()
            assert ok, "admission refused; raise --buffer-gb / --max-tokens"
            if args.profile_admission and trial == args.admit_warmup + args.admit_trials - 1:
                _ADMIT_PROFILER.enable()
            ctx.add_request(req)
            if args.profile_admission and trial == args.admit_warmup + args.admit_trials - 1:
                _ADMIT_PROFILER.disable()
            t6 = time.perf_counter()
            step["prompt_to_tensor"] += t1 - t0
            step["build_request"] += t2 - t1
            step["check_availability_1"] += t3 - t2
            step["prefix_skip_match"] += t4 - t3
            step["check_availability_2"] += t5 - t4
            step["add_request"] += t6 - t5
        start = time.perf_counter()
        ctx._flush_pending_value_pages()
        step["flush_value_pages"] = time.perf_counter() - start
        for name in ("add_request/allocate_memory_blocks", "add_request/init_value_pages"):
            step[name] = nested.get(name, 0.0)
        step["add_request/rest"] = (
            step["add_request"]
            - step["add_request/allocate_memory_blocks"]
            - step["add_request/init_value_pages"]
        )
        torch.cuda.synchronize()
        if trial >= args.admit_warmup:
            for op in ADMIT_OPS:
                per_request[op].append(step[op] / k_admit)
        del ctx
        torch.cuda.empty_cache()
    row = {"resident": n_resident, "admitted": k_admit, "prompt_len": args.prompt_len}
    for op, samples in per_request.items():
        row[op] = round(statistics.median(samples) * 1e6, 1)
    total_ops = ADMIT_OPS[:6] + ("flush_value_pages",)
    row["total"] = round(sum(row[op] for op in total_ops), 1)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 32, 128, 256, 512, 1024])
    parser.add_argument("--steps", type=int, default=300, help="timed decode steps per count")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--buffer-gb", type=float, default=2.0)
    parser.add_argument("--max-sequence-length", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--min-max-requests", type=int, default=1024)
    parser.add_argument("--num-cuda-graphs", type=int, default=32)
    parser.add_argument("--no-prefix-caching", dest="prefix_caching", action="store_false")
    parser.add_argument("--mamba-prefix-gb", type=float, default=0.5)
    parser.add_argument(
        "--admit",
        type=str,
        nargs="*",
        default=None,
        help="admission mode: RESIDENT:ADMITTED pairs, e.g. 0:64 256:64",
    )
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--profile-admission", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline"],
        help="optimization sets to ablate, e.g. baseline cpu_prompts cpu_prompts+scalar_metadata",
    )
    parser.add_argument("--admit-trials", type=int, default=10)
    parser.add_argument("--admit-warmup", type=int, default=2)
    parser.add_argument("--profile-counts", type=int, nargs="*", default=[])
    parser.add_argument("--out", default=None, help="write the rows as JSON lines here")
    args = parser.parse_args()

    torch.distributed.init_process_group(
        "nccl", init_method="tcp://127.0.0.1:29571", world_size=1, rank=0
    )
    torch.cuda.set_device(0)
    parallel_state.initialize_model_parallel()

    if args.admit is not None:
        from megatron.core.inference import admission_opts

        variants = [v.split("+") if v != "baseline" else [] for v in args.variants]
        rows = []
        for pair in args.admit:
            n_resident, k_admit = (int(x) for x in pair.split(":"))
            for opts in variants:
                admission_opts.ENABLED.clear()
                admission_opts.ENABLED.update(opts)
                device = "cpu" if "cpu_prompts" in opts else torch.cuda.current_device()
                row = run_admission(args, n_resident, k_admit, device)
                row["variant"] = "+".join(opts) or "baseline"
                rows.append(row)
        if args.profile_admission:
            out = io.StringIO()
            pstats.Stats(_ADMIT_PROFILER, stream=out).sort_stats("tottime").print_stats(25)
            print("===== cProfile of context.add_request (last trial, all variants) =====")
            print("\n".join(out.getvalue().splitlines()[:45]))
        print(f"admission: median us per admitted request (prompt_len={args.prompt_len})")
        cols = ["variant", "resident", "admitted"] + list(ADMIT_OPS) + ["total"]
        print(" | ".join(cols))
        for row in rows:
            print(" | ".join(str(row[c]) for c in cols))
        if args.out:
            with open(args.out, "w") as f:
                f.write("\n".join(json.dumps(row) for row in rows) + "\n")
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()
        return

    rows = [run_count(args, n) for n in args.counts]
    header = ["N", "pool"] + [op for op in OPS] + ["total"]
    print("p50 microseconds per decode step (CPU wall clock)")
    print("  ".join(f"{h:>14}" for h in header))
    for row in rows:
        values = [row["active_requests"], row["kv_pool_blocks"]]
        values += [row[f"{op}_p50_us"] for op in OPS] + [row["total_p50_us"]]
        print("  ".join(f"{v:>14}" for v in values))
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(json.dumps(row) for row in rows) + "\n")

    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
