"""Per-block vs once-per-rank symmetric-memory barrier: correctness and cost vs grid size.

  per_block   symm_mem_sync: every block sends one remote atomic per peer and waits on one
              signal-pad slot per peer, so its cost grows with the grid
  end         symm_mem_sync_end: blocks sign in on a local counter; the last one adds 1 to this
              rank's slot on every rank (one multimem.red) and takes one arrival from each slot

For each barrier and grid size it
  1. checks it: launch i reads every peer's value from launch i - 1 over NVLink (double-buffered)
     and writes its own for launch i + 1, with the barrier as the only synchronization; a lost
     arrival hangs, a missed ordering shows up as errors;
  2. times a kernel containing only the barrier against an empty kernel (the launch floor), as a
     CUDA graph of N launches, median of R replays, max over ranks.

One process per GPU (torchrun, or srun with SLURM_PROCID / SLURM_LOCALID / SLURM_NTASKS):

    torchrun --nproc-per-node 4 inference-bench/microbench/barrier_only_bench.py
    srun -N 16 --ntasks-per-node 4 --segment 16 ... python3 inference-bench/microbench/barrier_only_bench.py
"""

import argparse
import os
import re

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from megatron.core.inference.communication.torch_symm_triton.barrier import (
    BARRIER_STATE_BYTES,
    symm_mem_sync,
    symm_mem_sync_end,
)
from megatron.core.inference.communication.torch_symm_triton.utils import sync_threads

NUM_WARPS = 4  # 128 threads: both barriers need world_size threads per block
KINDS = {"per_block": 0, "end": 1}


@triton.jit
def _barrier(sig, barrier_mc, barrier_local, RANK: tl.constexpr, WORLD_SIZE: tl.constexpr,
             KIND: tl.constexpr):
    if KIND == 0:
        sync_threads()
        symm_mem_sync(sig, None, RANK, WORLD_SIZE, hasPreviousMemAccess=True,
                      hasSubsequentMemAccess=True)
    else:
        symm_mem_sync_end(barrier_mc, barrier_local, tl.num_programs(0), RANK, WORLD_SIZE)


@triton.jit
def _barrier_kernel(sig, barrier_mc, barrier_local, RANK: tl.constexpr, WORLD_SIZE: tl.constexpr,
                    KIND: tl.constexpr):
    _barrier(sig, barrier_mc, barrier_local, RANK, WORLD_SIZE, KIND)


@triton.jit
def _empty_kernel(sig, barrier_mc, barrier_local, RANK: tl.constexpr, WORLD_SIZE: tl.constexpr,
                  KIND: tl.constexpr):
    pass


@triton.jit
def _check_kernel(sig, barrier_mc, barrier_local, step_ptr, count_ptr, buffer_ptrs, data_off,
                  max_blocks, errors, RANK: tl.constexpr, WORLD_SIZE: tl.constexpr,
                  KIND: tl.constexpr):
    step = tl.load(step_ptr.to(tl.pointer_type(tl.uint32)), volatile=True)
    block = tl.program_id(0)
    peers = tl.arange(0, WORLD_SIZE)
    ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    read_off = data_off + (((step % 2) * max_blocks + block) * WORLD_SIZE) * 4
    write_off = data_off + ((((step + 1) % 2) * max_blocks + block) * WORLD_SIZE) * 4
    seen = tl.load((tl.load(ptrs + peers) + read_off).to(tl.pointer_type(tl.uint32)) + peers,
                   volatile=True)
    bad = tl.sum((seen != step).to(tl.int32))
    if bad > 0:
        tl.atomic_add(errors, bad)
    tl.store((tl.load(ptrs + RANK) + write_off).to(tl.pointer_type(tl.uint32)) + RANK, step + 1)
    _barrier(sig, barrier_mc, barrier_local, RANK, WORLD_SIZE, KIND)
    # The last block to finish advances the check's launch counter.
    sync_threads()
    done = tl.atomic_add(count_ptr.to(tl.pointer_type(tl.int32)), 1, sem="acq_rel", scope="gpu")
    if done == tl.num_programs(0) - 1:
        tl.store(count_ptr.to(tl.pointer_type(tl.int32)), 0)
        tl.store(step_ptr.to(tl.pointer_type(tl.uint32)), step + 1)


def first_host(nodelist):
    """First host of a Slurm nodelist: 'nvl72023-T[16-17,20]' -> 'nvl72023-T16'."""
    m = re.match(r"([^,\[]+)(?:\[([^\]]+)\])?", nodelist)
    prefix, ranges = m.group(1), m.group(2)
    return prefix + ranges.split(",")[0].split("-")[0] if ranges else prefix


def init():
    if "RANK" not in os.environ:  # srun: one task per GPU
        os.environ.update(RANK=os.environ["SLURM_PROCID"], LOCAL_RANK=os.environ["SLURM_LOCALID"],
                          WORLD_SIZE=os.environ["SLURM_NTASKS"])
        os.environ.setdefault("MASTER_ADDR", first_host(os.environ["SLURM_JOB_NODELIST"]))
        os.environ.setdefault("MASTER_PORT", "29514")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    return dist.get_rank(), dist.get_world_size()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, nargs="+", default=[152, 1, 8, 32, 128, 152])
    ap.add_argument("--launches", type=int, default=200)
    ap.add_argument("--check-launches", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=9, help="timed replays; the median is reported")
    args = ap.parse_args()
    rank, world = init()
    max_blocks = max(args.blocks)
    if hasattr(symm_mem, "set_signal_pad_size"):
        symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), max_blocks * world * 4))
    # Symmetric buffer: barrier state | check counters | check data (2 x max_blocks x world)
    words_off = BARRIER_STATE_BYTES
    data_off = words_off + 1024
    buf = symm_mem.empty(data_off + 2 * max_blocks * world * 4, dtype=torch.uint8, device="cuda")
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    sig, mc, local = hdl.signal_pad_ptrs_dev, hdl.multicast_ptr, buf.data_ptr()
    errors = torch.zeros(1, dtype=torch.int32, device="cuda")

    def reset():
        buf.zero_()
        torch.cuda.synchronize()
        dist.barrier()

    rows = []
    for blocks in args.blocks:
        result = {}
        for name, kind in KINDS.items():
            meta = dict(RANK=rank, WORLD_SIZE=world, KIND=kind, num_warps=NUM_WARPS)
            reset()
            errors.zero_()
            for _ in range(args.check_launches):
                _check_kernel[(blocks,)](sig, mc, local, local + words_off, local + words_off + 4,
                                         hdl.buffer_ptrs_dev, data_off, max_blocks, errors, **meta)
            torch.cuda.synchronize()
            dist.all_reduce(errors)
            result[name + "_err"] = int(errors.item())
            reset()
            for label, kernel in ((name, _barrier_kernel), (name + "_empty", _empty_kernel)):
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    kernel[(blocks,)](sig, mc, local, **meta)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.launches):
                        kernel[(blocks,)](sig, mc, local, **meta)
                graph.replay()
                torch.cuda.synchronize()
                times = []
                for _ in range(args.repeats):
                    dist.barrier()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end) * 1e3 / args.launches)
                us = torch.tensor([sorted(times)[len(times) // 2]], device="cuda")
                dist.all_reduce(us, op=dist.ReduceOp.MAX)
                result[label] = us.item()
        rows.append((blocks, result))
        if rank == 0:
            print(f"[{blocks} blocks] done", flush=True)

    if rank == 0:
        sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        print(f"world_size={world}, SMs per GPU: {sms}  (barrier cost = barrier kernel - empty "
              f"kernel, us per launch; CUDA graph of {args.launches}, median of {args.repeats} "
              f"replays, max over ranks)")
        print(f"{'blocks':>6s} {'per_block':>10s} {'end':>8s}   errors (per_block/end)")
        for blocks, r in rows:
            print(f"{blocks:6d} {r['per_block'] - r['per_block_empty']:10.2f} "
                  f"{r['end'] - r['end_empty']:8.2f}   {r['per_block_err']}/{r['end_err']}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
