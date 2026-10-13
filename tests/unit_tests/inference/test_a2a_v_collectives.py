# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Known-value tests for the NVLink all-to-all-v dispatch / combine kernels.

Every rank builds the same global batch from a shared seed and keeps its own slice, so each rank
can compute the exact expected result for every row without any extra communication.
"""

import pytest
import torch

from tests.unit_tests.test_utilities import Utils

_HIDDEN = 256
_TOPK = 6
_NUM_LOCAL_EXPERTS = 4
_PER_RANK_MAX_TOKENS = 16


def _local_rows(routing, rank):
    """[rows] bool: the row has at least one expert on `rank`."""
    lo, hi = rank * _NUM_LOCAL_EXPERTS, (rank + 1) * _NUM_LOCAL_EXPERTS
    return ((routing >= lo) & (routing < hi)).any(dim=1)


def _num_dest_ranks(routing, ep_size):
    """[rows] fp32: number of distinct destination ranks per row (0 for all -1 rows)."""
    ranks = torch.arange(ep_size, device=routing.device).view(1, ep_size, 1)
    expanded = routing.unsqueeze(1)
    hit = (expanded >= ranks * _NUM_LOCAL_EXPERTS) & (expanded < (ranks + 1) * _NUM_LOCAL_EXPERTS)
    return hit.any(dim=2).sum(dim=1).to(torch.float32)


def _routing(valid, ep_size, gen, device):
    """Distinct experts per token on a random subset of ranks, so holes exist even at small EP.

    The last token routes to no expert (all -1), like a CUDA-graph padding row.
    """
    min_ranks = -(-_TOPK // _NUM_LOCAL_EXPERTS)
    rows = []
    for _ in range(valid):
        num_ranks = int(torch.randint(min_ranks, ep_size + 1, (1,), generator=gen, device=device))
        ranks = torch.randperm(ep_size, generator=gen, device=device)[:num_ranks]
        experts = (
            ranks.view(-1, 1) * _NUM_LOCAL_EXPERTS
            + torch.arange(_NUM_LOCAL_EXPERTS, device=device)
        ).flatten()
        rows.append(experts[torch.randperm(experts.numel(), generator=gen, device=device)[:_TOPK]])
    routing = torch.stack(rows).to(torch.int64)
    if valid >= 2:
        routing[-1] = -1
    return routing


class TestAllToAllV:

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)

    @classmethod
    def teardown_class(cls):
        from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    def _require_buffers(self):
        """Return (ep_group, buffers) or skip if NVLink symmetric memory is unavailable."""
        from megatron.core import parallel_state
        from megatron.core.inference.communication.torch_symm_triton import (
            is_device_nvls_capable,
        )
        from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

        device = torch.device("cuda", torch.cuda.current_device())
        if not is_device_nvls_capable(device):
            pytest.skip("NVLink symmetric memory requires a Hopper+ GPU (SM >= 9)")
        ep_group = parallel_state.get_expert_model_parallel_group()
        ep_size = torch.distributed.get_world_size(ep_group)
        if ep_size < 2:
            pytest.skip("requires an expert-model-parallel world size >= 2")
        if _TOPK > ep_size * _NUM_LOCAL_EXPERTS:
            pytest.skip("not enough experts for distinct top-k picks")

        global_max = _PER_RANK_MAX_TOKENS * ep_size
        buffers = {}
        for key, width, dtype in (
            ("hidden", _HIDDEN, torch.bfloat16),
            ("routing", _TOPK, torch.int64),
            ("probs", _TOPK, torch.float32),
            ("partial", _HIDDEN, torch.bfloat16),
            ("meta", None, torch.int32),
        ):
            shape = [ep_size] if width is None else [global_max, width]
            buf = SymmetricMemoryManager.get_buffer(f"test_a2a_{key}", process_group=ep_group)
            if buf.symm_mem_hdl is None:
                pytest.skip(f"symmetric memory unavailable: {buf.init_failure_reason}")
            buffers[key] = buf.maybe_get_tensor(shape, dtype=dtype)
            assert buffers[key]["handle"] is not None
        return ep_group, buffers

    # single_token: a batch of 1, so every rank but one dispatches zero tokens.
    @pytest.mark.parametrize("single_token", [False, True])
    def test_dispatch_and_combine(self, single_token):
        from megatron.core.inference.communication.torch_symm_triton import (
            a2a_combine_v,
            a2a_dispatch_v,
        )
        from megatron.core.inference.moe.metadata import fused_metadata_update

        ep_group, bufs = self._require_buffers()
        ep_size = torch.distributed.get_world_size(ep_group)
        rank = torch.distributed.get_rank(ep_group)
        device = torch.device("cuda", torch.cuda.current_device())
        global_max = _PER_RANK_MAX_TOKENS * ep_size

        batch = 1 if single_token else 2 * ep_size
        counts = [batch // ep_size + (1 if r < batch % ep_size else 0) for r in range(ep_size)]
        local_tokens, offset, valid = counts[rank], sum(counts[:rank]), sum(counts)

        # Identical on every rank.
        gen = torch.Generator(device=device).manual_seed(20260709 + batch)
        full_hidden = torch.randn(valid, _HIDDEN, generator=gen, device=device).to(torch.bfloat16)
        full_routing = _routing(valid, ep_size, gen, device)
        full_probs = torch.rand(valid, _TOPK, generator=gen, device=device)
        full_partial = torch.randn(valid, _HIDDEN, generator=gen, device=device).to(
            torch.bfloat16
        )
        local = _local_rows(full_routing, rank)

        in_hidden = full_hidden[offset : offset + local_tokens].contiguous()
        in_routing = full_routing[offset : offset + local_tokens].contiguous()
        in_probs = full_probs[offset : offset + local_tokens].contiguous()

        step_metadata = torch.zeros(3, dtype=torch.int32, device=device)
        fused_metadata_update(
            local_tokens=local_tokens,
            local_buf=bufs["meta"]["tensor"],
            symm_mem_hdl=bufs["meta"]["handle"],
            step_metadata=step_metadata,
        )

        # NaN sentinels: a hole that gets written, or a row pulled from a non-destination rank,
        # shows up as a mismatch.
        recv_hidden = bufs["hidden"]["tensor"].view(global_max, _HIDDEN)
        recv_hidden[:valid] = float("nan")
        torch.cuda.synchronize()
        torch.distributed.barrier(ep_group)

        a2a_dispatch_v(
            bufs["hidden"]["tensor"],
            bufs["routing"]["tensor"],
            bufs["probs"]["tensor"],
            in_hidden,
            in_routing,
            in_probs,
            bufs["hidden"]["handle"],
            bufs["routing"]["handle"],
            bufs["probs"]["handle"],
            rank_token_offset=step_metadata[1:2],
            ep_max_tokens=step_metadata[2:3],
            per_rank_max_tokens=_PER_RANK_MAX_TOKENS,
            num_local_experts=_NUM_LOCAL_EXPERTS,
        )
        torch.cuda.synchronize()

        assert int(step_metadata[0]) == valid
        assert torch.equal(bufs["routing"]["tensor"].view(global_max, _TOPK)[:valid], full_routing)
        assert torch.equal(bufs["probs"]["tensor"].view(global_max, _TOPK)[:valid], full_probs)
        assert torch.equal(recv_hidden[:valid][local], full_hidden[local])
        assert bool(torch.isnan(recv_hidden[:valid][~local].float()).all())

        # Each rank exposes partials only on the rows routed to it.
        partial = bufs["partial"]["tensor"].view(global_max, _HIDDEN)
        partial[:valid] = float("nan")
        partial[:valid][local] = full_partial[local]
        torch.cuda.synchronize()
        torch.distributed.barrier(ep_group)

        output = torch.empty(local_tokens, _HIDDEN, dtype=torch.bfloat16, device=device)
        a2a_combine_v(
            output,
            bufs["partial"]["tensor"],
            bufs["partial"]["handle"],
            in_routing,
            rank_token_offset=step_metadata[1:2],
            ep_max_tokens=step_metadata[2:3],
            per_rank_max_tokens=_PER_RANK_MAX_TOKENS,
            num_local_experts=_NUM_LOCAL_EXPERTS,
        )
        torch.cuda.synchronize()

        # Every destination holds the same partial row, so the sum is m * partial, which is
        # exact in fp32 for m <= ep_size; all -1 rows have m = 0 and combine to zero.
        num_dest = _num_dest_ranks(in_routing, ep_size).view(local_tokens, 1)
        expected = (num_dest * full_partial[offset : offset + local_tokens].float()).to(
            torch.bfloat16
        )
        assert torch.equal(output, expected)

        # Keep the next test's sentinel writes behind every rank's pulls from this one.
        torch.distributed.barrier(ep_group)
