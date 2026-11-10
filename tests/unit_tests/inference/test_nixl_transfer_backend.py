# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed unit test of the one-sided NIXL transfer backend on real GPUs.

Two KV heads with a TP4 prefill, so every prefill rank holds one replicated
head (the layout Nemotron-H models use at TP4/TP8). Each rank registers a
prefill buffer filled with a known global KV, then decode buffers pull it:

* decode TP4 (heads replicated again): each rank reads from its own-index
  prefill replica;
* decode TP2 on ranks 0 and 1 (one head per rank, not replicated): rank 1
  reads head 1 from prefill rank 3, a cross-GPU read.

Skipped unless the ``nixl`` package is installed and >= 4 CUDA ranks run.
"""

import os

import pytest
import torch

L, H, HD, T, NB = 4, 2, 16, 8, 6  # layers, kv heads, head dim, tokens/block, pool blocks
SRC_BLOCKS = [1, 3]
DST_BLOCKS = [4, 0]


def _have_nixl() -> bool:
    try:
        import nixl  # noqa: F401  # pylint: disable=unused-import
    except ImportError:
        return False
    return True


def _global_blocks(device):
    """Global KV for the request's blocks: (block, kv, layer, token, head, dim)
    with a distinct value per (block, kv, layer, token, head)."""
    g = torch.zeros(len(SRC_BLOCKS), 2, L, T, H, HD, device=device)
    for b in range(len(SRC_BLOCKS)):
        for kv in range(2):
            for l in range(L):
                for t in range(T):
                    for h in range(H):
                        g[b, kv, l, t, h, :] = (((b * 2 + kv) * L + l) * T + t) * 10 + h
    return g


def _backend(role, rank, tp_size, tp_rank, device):
    from megatron.core.inference.disaggregation.transfer_backends.nixl import NixlTransferBackend

    heads_local = H // tp_size if H >= tp_size else 1
    buf = torch.full((2, L, NB, T, heads_local, HD), -1.0, device=device)
    backend = NixlTransferBackend(
        agent_name=f"{role}-tp{tp_size}-rank{rank}",
        memory_buffer=buf,
        expected_num_blocks=NB,
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_kv_heads_global=H,
        heads_per_partition=heads_local,
        head_dim=HD,
        tokens_per_block=T,
        global_rank=tp_rank,
        pp_size=1,
        pp_rank=0,
        num_layers_global=L,
        layer_start=0,
        layer_end=L,
    )
    return backend, buf


def _pull_and_check(backend, buf, prefill_metas, g):
    lo, hi = backend._layout.head_range()
    backend.begin_pull_blocks({"tp_metas": prefill_metas}, SRC_BLOCKS, DST_BLOCKS).wait()
    torch.cuda.synchronize()
    for i, block in enumerate(DST_BLOCKS):
        if not torch.equal(buf[:, :, block], g[i, :, :, :, lo:hi, :]):
            return False
    untouched = [b for b in range(NB) if b not in DST_BLOCKS]
    return bool((buf[:, :, untouched] == -1.0).all())


@pytest.mark.skipif(
    not (
        _have_nixl()
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 4
        and int(os.environ.get("WORLD_SIZE", "1")) >= 4
    ),
    reason="requires the nixl package and torchrun with >= 4 CUDA ranks",
)
def test_nixl_pull_replicated_kv_heads():
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    control_group = dist.new_group(backend="gloo")
    active = rank < 4

    g = _global_blocks(device)
    prefill = None
    if active:
        prefill, prefill_buf = _backend("prefill", rank, 4, rank, device)
        head = rank // 2
        for i, block in enumerate(SRC_BLOCKS):
            prefill_buf[:, :, block] = g[i, :, :, :, head : head + 1, :]
        torch.cuda.synchronize()

    metas = [None] * dist.get_world_size()
    dist.all_gather_object(metas, prefill.export_meta() if active else None, group=control_group)
    prefill_metas = metas[:4]

    ok = True
    keep = []
    if active:
        decode4, decode4_buf = _backend("decode", rank, 4, rank, device)
        ok &= _pull_and_check(decode4, decode4_buf, prefill_metas, g)
        keep.append(decode4)
        if rank < 2:
            decode2, decode2_buf = _backend("decode", rank, 2, rank, device)
            ok &= _pull_and_check(decode2, decode2_buf, prefill_metas, g)
            keep.append(decode2)

    result = torch.tensor(int(ok))
    dist.all_reduce(result, op=dist.ReduceOp.MIN, group=control_group)
    # Prefill buffers must stay registered until every decode read finished.
    for backend in keep + ([prefill] if prefill is not None else []):
        backend.close()
    assert result.item() == 1
