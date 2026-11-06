"""FA4 paged decode attention on real per-rank KV-length snapshots, sweeping num_splits.

FA4's split heuristic (num_splits=0) picks min(num_SMs // (batch * kv_heads), ...) splits, which is 1 at
our batch sizes: one CTA per request, so a long-tailed batch waits for its longest request. This times
one attention layer's decode kernel (+ split-KV combine) per snapshot and num_splits, and the same
batch with every request at the mean length ("balanced": same total KV, no tail), with L2 flushed
before every call. Shapes default to Super on a TP4 rank: 8 query heads, 1 KV head, head dim 128,
256-token pages.

    python3 microbench/fa4_decode_splits_bench.py --snapshots <kv_snapshots.json> --out <results.jsonl>

kv_snapshots.json: {"name": [context length of each active request], ...}.
"""

import argparse
import json
import statistics

import torch
from flash_attn.cute import flash_attn_varlen_func


def build(lengths, args, gen):
    """Paged K/V caches (pages randomly placed, as after fragmentation), block table, q."""
    dev = "cuda"
    pages = [(n + args.page_size - 1) // args.page_size for n in lengths]
    num_pages = sum(pages)
    perm = torch.randperm(num_pages, generator=gen).to(dev, torch.int32)
    table = torch.zeros(len(lengths), max(pages), dtype=torch.int32, device=dev)
    off = 0
    for i, p in enumerate(pages):
        table[i, :p] = perm[off : off + p]
        off += p
    shape = (num_pages, args.page_size, args.kv_heads, args.head_dim)
    k = torch.randn(shape, dtype=torch.bfloat16, device=dev)
    v = torch.randn(shape, dtype=torch.bfloat16, device=dev)
    q = torch.randn(len(lengths), args.q_heads, args.head_dim, dtype=torch.bfloat16, device=dev)
    cu_q = torch.arange(len(lengths) + 1, dtype=torch.int32, device=dev)
    seqused = torch.tensor(lengths, dtype=torch.int32, device=dev)
    return q, k, v, cu_q, seqused, table


def attn(inputs, max_len, splits):
    q, k, v, cu_q, seqused, table = inputs
    out, _ = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu_q, max_seqlen_q=1, max_seqlen_k=max_len, seqused_k=seqused,
        page_table=table, softmax_scale=q.shape[-1] ** -0.5, causal=True, num_splits=splits,
    )
    return out


def time_graph(fn, iters, reps):
    """Median microseconds per fn() call, fn captured `iters` times in one CUDA graph."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(1e3 * a.elapsed_time(b) / iters)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", type=int, nargs="+", default=[0, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--q-heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=1)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--page-size", type=int, default=256)
    ap.add_argument("--variants", nargs="+", default=["real", "balanced"], choices=["real", "balanced"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--hbm-tb-s", type=float, default=8.0, help="for the KV-read floor")
    args = ap.parse_args()

    snapshots = json.load(open(args.snapshots))
    gen = torch.Generator().manual_seed(0)
    flush = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda")  # > L2
    flush_us = time_graph(lambda: flush.zero_(), args.iters, args.reps)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"{torch.cuda.get_device_name(0)}, {sms} SMs; L2 flush {flush_us:.1f} us (subtracted)", flush=True)
    out = open(args.out, "w")
    for name, lengths in snapshots.items():
        mean = round(sum(lengths) / len(lengths))
        for variant, lens in (("real", lengths), ("balanced", [mean] * len(lengths))):
            if variant not in args.variants:
                continue
            inputs = build(lens, args, gen)
            max_len = max(lens)
            kv_bytes = 2 * sum(lens) * args.kv_heads * args.head_dim * 2
            floor_us = kv_bytes / (args.hbm_tb_s * 1e12) * 1e6
            ref = attn(inputs, max_len, 1)
            for s in args.splits:
                err = (attn(inputs, max_len, s) - ref).abs().max().item()

                def fn():
                    flush.zero_()
                    attn(inputs, max_len, s)

                us = time_graph(fn, args.iters, args.reps) - flush_us
                rec = {"snapshot": name, "variant": variant, "num_splits": s, "us": us,
                       "batch": len(lens), "max_len": max_len, "mean_len": mean,
                       "total_kv_tokens": sum(lens), "kv_read_floor_us": floor_us, "max_abs_err": err}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"{name:28s} {variant:8s} splits {s:3d}: {us:8.1f} us  (KV-read floor "
                      f"{floor_us:6.1f} us, max {max_len}, mean {mean}, err {err:.1e})", flush=True)
            del inputs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
