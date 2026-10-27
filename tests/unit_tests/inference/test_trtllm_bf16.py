# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the TRT-LLM BF16 routed MoE path and the EP token-sum consensus."""

import pytest
import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.moe import trtllm_bf16


def _reference(hidden, routing, probs, fc1, fc2, local_offset):
    """fp32 squared-ReLU MoE over the local experts; non-local / -1 ids contribute nothing."""
    out = torch.zeros(hidden.shape[0], fc2.shape[1], device=hidden.device)
    for e in range(fc1.shape[0]):
        tok, slot = (routing == e + local_offset).nonzero(as_tuple=True)
        if tok.numel():
            h = torch.relu(hidden[tok].float() @ fc1[e].float().t()) ** 2
            y = h.bfloat16().float() @ fc2[e].float().t()
            out.index_add_(0, tok, y * probs[tok, slot][:, None])
    return out


@pytest.mark.internal
@pytest.mark.skipif(
    not trtllm_bf16.HAVE_FLASHINFER_TRTLLM_BF16, reason="needs FlashInfer trtllm_bf16_routed_moe"
)
class TestTrtllmBf16RoutedMoe:

    # Ultra-like expert shape (scaled down) and a non-128 intermediate to exercise padding.
    @pytest.mark.parametrize("hidden,inter", [(1024, 1280), (512, 320)])
    @pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("valid,rows", [(1, 8), (37, 64), (64, 64)])
    def test_bounded_rows_match_reference(self, hidden, inter, out_dtype, valid, rows):
        from flashinfer.fused_moe.core import ActivationType

        torch.manual_seed(valid)
        num_experts, local, offset, topk, buffer_rows = 64, 8, 16, 6, 256
        fc1 = (torch.randn(local, inter, hidden, device="cuda") * hidden**-0.5).bfloat16()
        fc2 = (torch.randn(local, hidden, inter, device="cuda") * inter**-0.5).bfloat16()
        w1, w2, padded = trtllm_bf16.build_trtllm_bf16_weights(fc1, fc2, gated=False)

        # Worst-case buffers holding stale garbage, as after an earlier larger step.
        h = (torch.randn(buffer_rows, hidden, device="cuda") * 0.5).bfloat16()
        r = torch.randint(0, num_experts, (buffer_rows, topk), device="cuda")
        p = torch.rand(buffer_rows, topk, device="cuda")
        r[:valid] = torch.rand(valid, num_experts, device="cuda").topk(topk, dim=1).indices
        p[:valid] = p[:valid] / p[:valid].sum(1, keepdim=True)
        # Rows in [valid, rows) keep stale expert ids; pack_routing's real-token-count mask
        # routes them to -1 (on the server this happens per rank, before the dispatch).
        out = torch.full((buffer_rows, hidden), 7.0, device="cuda", dtype=out_dtype)
        valid_t = torch.tensor([valid], dtype=torch.int32, device="cuda")

        with torch.inference_mode():
            packed = trtllm_bf16.pack_routing(r, p, valid_t)
            trtllm_bf16.trtllm_bf16_routed_moe_forward(
                h, packed, w1, w2,
                num_experts=num_experts, local_expert_offset=offset, intermediate_size=padded,
                activation_type=ActivationType.Relu2.value, rows=rows, out=out, valid_tokens=valid_t,
            )
            finalized = trtllm_bf16.trtllm_bf16_routed_moe_forward(
                h, packed, w1, w2,
                num_experts=num_experts, local_expert_offset=offset, intermediate_size=padded,
                activation_type=ActivationType.Relu2.value, rows=rows,
            )
        torch.cuda.synchronize()
        ref = _reference(h[:valid], r[:valid], p[:valid], fc1, fc2, offset)
        scale = ref.norm().clamp_min(1e-6)
        assert (out[:valid].float() - ref).norm() / scale < 2e-2
        assert (finalized[:valid].float() - ref).norm() / scale < 2e-2
        # Rows past the valid prefix are never written.
        assert torch.all(out[valid:] == 7.0)



@pytest.mark.internal
class TestPackRouting:

    # (rows, topk, real token count, SP row offset): real count is in the global pre-SP frame.
    @pytest.mark.parametrize(
        "rows,topk,real,offset",
        [(1, 22, 1, 0), (64, 22, 37, 0), (300, 6, 300, 0), (5, 8, None, 0), (16, 22, 40, 32)],
    )
    def test_matches_torch(self, rows, topk, real, offset):
        torch.manual_seed(rows)
        ids = torch.randint(-1, 512, (rows, topk), device="cuda")
        probs = torch.rand(rows, topk, device="cuda")
        real_t = None if real is None else torch.tensor([real], dtype=torch.int32, device="cuda")
        ref = (ids.to(torch.int32) << 16) | (
            probs.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
        )
        if real is not None:
            ref[max(0, real - offset):] = -1 << 16
        got = trtllm_bf16.pack_routing(ids, probs, real_t, row_offset=offset)
        assert torch.equal(got, ref)


if trtllm_bf16.HAVE_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _one_token_finalize_kernel(
        gemm2_ptr,
        weights_ptr,
        inverse_map_ptr,
        output_ptr,
        valid_tokens_ptr,
        hidden_size,
        gemm2_rows,
        TOP_K: tl.constexpr,
        BLOCK_TOP_K: tl.constexpr,
        BLOCK_H: tl.constexpr,
        NUM_TOKEN_BLOCKS: tl.constexpr,
        NUM_H_BLOCKS: tl.constexpr,
    ):
        """The previous finalize kernel (one token per iteration, every slot of a routed
        token), kept as the bit-exact reference for the tiled kernel."""
        token_pid = tl.program_id(0)
        hidden_pid = tl.program_id(1)
        valid_tokens = tl.load(valid_tokens_ptr).to(tl.int32)
        top_k_offsets = tl.arange(0, BLOCK_TOP_K)
        top_k_mask = top_k_offsets < TOP_K
        for token in tl.range(token_pid, valid_tokens, NUM_TOKEN_BLOCKS):
            mapped_rows = tl.load(
                inverse_map_ptr + token * TOP_K + top_k_offsets, mask=top_k_mask, other=-1
            ).to(tl.int32)
            local_mask = top_k_mask & (mapped_rows >= 0) & (mapped_rows < gemm2_rows)
            has_local_expert = tl.sum(local_mask.to(tl.int32), axis=0) > 0
            for hidden_offset in tl.range(
                hidden_pid * BLOCK_H, hidden_size, NUM_H_BLOCKS * BLOCK_H
            ):
                offsets = hidden_offset + tl.arange(0, BLOCK_H)
                offsets_mask = offsets < hidden_size
                accumulator = tl.zeros([BLOCK_H], dtype=tl.float32)
                if has_local_expert:
                    for top_k_idx in range(TOP_K):
                        mapped_row = tl.load(inverse_map_ptr + token * TOP_K + top_k_idx).to(
                            tl.int32
                        )
                        is_local = (mapped_row >= 0) & (mapped_row < gemm2_rows)
                        safe_row = tl.maximum(mapped_row, 0).to(tl.int64)
                        weight = tl.load(
                            weights_ptr + token * TOP_K + top_k_idx, mask=is_local, other=0.0
                        ).to(tl.float32)
                        value = tl.load(
                            gemm2_ptr + safe_row * hidden_size + offsets,
                            mask=is_local & offsets_mask,
                            other=0.0,
                        ).to(tl.float32)
                        accumulator += value * weight
                tl.store(
                    output_ptr + token * hidden_size + offsets,
                    accumulator.to(output_ptr.dtype.element_ty),
                    mask=offsets_mask,
                )


def _one_token_finalize(gemm2_output, expert_weights, inverse_map, output, valid_tokens, top_k):
    """Launch the reference kernel the way the previous trtllm_finalize did."""
    rows, hidden_size = output.shape
    num_token_blocks = min(rows, 128)
    num_h_blocks = min(triton.cdiv(hidden_size, 256), 8)
    _one_token_finalize_kernel[(num_token_blocks, num_h_blocks)](
        gemm2_output,
        expert_weights,
        inverse_map,
        output,
        valid_tokens,
        hidden_size,
        gemm2_output.shape[0],
        TOP_K=top_k,
        BLOCK_TOP_K=triton.next_power_of_2(top_k),
        BLOCK_H=256,
        NUM_TOKEN_BLOCKS=num_token_blocks,
        NUM_H_BLOCKS=num_h_blocks,
        num_warps=4,
    )
    return output


def _finalize_inputs(
    rows, hidden, top_k, num_experts, local_start, num_local, padding_every=0, weights_dtype=None
):
    """Finalize inputs laid out like trtllm_bf16_routed_moe(do_finalize=False) returns them.

    Each token routes to top_k distinct experts (every padding_every-th token to none). GEMM2
    rows are grouped by local expert, each group padded to a multiple of 8, and inverse_map
    maps a (token, slot) pair to its GEMM2 row, or -1 when the expert is not local.
    """
    torch.manual_seed(1234)
    ids = torch.rand(rows, num_experts, device="cuda").topk(top_k, dim=1).indices
    if padding_every:
        ids[::padding_every] = -1
    local_ids = ids - local_start
    is_local = (ids >= 0) & (local_ids >= 0) & (local_ids < num_local)
    pairs = is_local.flatten().nonzero().squeeze(1)
    experts = local_ids.flatten()[pairs]
    order = torch.argsort(experts, stable=True)
    counts = torch.bincount(experts, minlength=num_local)
    padded = (counts + 7) // 8 * 8
    group_start = torch.cumsum(padded, 0) - padded
    sorted_experts = experts[order]
    rank_in_group = (
        torch.arange(order.numel(), device="cuda")
        - (torch.cumsum(counts, 0) - counts)[sorted_experts]
    )
    inverse_map = torch.full((rows * top_k,), -1, dtype=torch.int32, device="cuda")
    inverse_map[pairs[order]] = (group_start[sorted_experts] + rank_in_group).int()
    gemm2 = torch.randn(max(int(padded.sum()), 8), hidden, device="cuda").bfloat16()
    weights = torch.rand(rows, top_k, device="cuda").to(weights_dtype or torch.bfloat16)
    return gemm2, weights, inverse_map.view(rows, top_k), is_local.any(dim=1)


def _vt(n):
    return torch.tensor([n], dtype=torch.int32, device="cuda")


@pytest.mark.internal
@pytest.mark.skipif(not trtllm_bf16.HAVE_TRITON, reason="needs Triton")
class TestTrtllmFinalize:

    @pytest.mark.parametrize("hidden", [128, 1000, 2048])
    @pytest.mark.parametrize(
        "num_experts,local_start,num_local,padding_every",
        [(64, 0, 64, 0), (64, 8, 8, 0), (512, 24, 8, 5)],
    )
    @pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
    def test_bit_exact_vs_one_token(
        self, hidden, num_experts, local_start, num_local, padding_every, out_dtype, weights_dtype
    ):
        """The tiled kernel sums the same slots in the same order as the previous kernel, so
        every written element is bit-identical; rows >= valid_tokens stay untouched."""
        rows, top_k, valid = 300, 22, 287
        gemm2, weights, inverse_map, _ = _finalize_inputs(
            rows, hidden, top_k, num_experts, local_start, num_local, padding_every, weights_dtype
        )

        def fresh_out():
            return torch.full((rows, hidden), float("nan"), device="cuda", dtype=out_dtype)

        expected = _one_token_finalize(gemm2, weights, inverse_map, fresh_out(), _vt(valid), top_k)
        result = trtllm_bf16.trtllm_finalize(
            gemm2, weights, inverse_map, fresh_out(), _vt(valid), top_k
        )
        assert torch.equal(result[:valid], expected[:valid])
        assert torch.isnan(result[valid:]).all()

    # BLOCK_T=1 masks a row with the tile's scalar slot mask, larger tiles per row.
    @pytest.mark.parametrize("block_t", [1, 4])
    @pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
    def test_skip_unrouted_rows(self, block_t, out_dtype):
        """write_unrouted_rows=False leaves rows with no local expert untouched and writes
        every other valid row bit-identically to the default."""
        rows, hidden, top_k, valid = 300, 1000, 22, 287
        gemm2, weights, inverse_map, routed = _finalize_inputs(
            rows, hidden, top_k, 512, 24, 8, padding_every=5
        )
        routed[valid:] = False
        assert routed.any() and not routed[:valid].all()

        def run(write_unrouted_rows):
            out = torch.full((rows, hidden), float("nan"), device="cuda", dtype=out_dtype)
            return trtllm_bf16.trtllm_finalize(
                gemm2,
                weights,
                inverse_map,
                out,
                _vt(valid),
                top_k,
                write_unrouted_rows=write_unrouted_rows,
                config={"BLOCK_T": block_t},
            )

        full, skipped = run(True), run(False)
        assert torch.equal(skipped[routed], full[routed])
        assert torch.isnan(skipped[~routed]).all()
        assert not torch.isnan(full[:valid]).any()
        assert torch.isnan(full[valid:]).all()

    @pytest.mark.parametrize(
        "config",
        [
            {"BLOCK_T": 4, "BLOCK_H": 512, "programs_per_sm": 8, "num_warps": 4},
            {"BLOCK_T": 2, "BLOCK_H": 128, "programs_per_sm": 1, "num_warps": 1},
            {"BLOCK_T": 8, "BLOCK_H": 256, "programs_per_sm": 16, "num_warps": 8},
            {"BLOCK_T": 1, "BLOCK_H": 1024, "programs_per_sm": 4, "num_warps": 4},
        ],
    )
    @pytest.mark.parametrize("write_unrouted_rows", [True, False])
    def test_result_independent_of_launch_config(self, config, write_unrouted_rows):
        """Tile sizes and program counts change only who computes an element, not how."""
        rows, hidden, top_k, valid = 300, 1000, 22, 291
        gemm2, weights, inverse_map, _ = _finalize_inputs(
            rows, hidden, top_k, 512, 24, 8, padding_every=7
        )

        def run(cfg):
            out = torch.full((rows, hidden), float("nan"), device="cuda")
            return trtllm_bf16.trtllm_finalize(
                gemm2,
                weights,
                inverse_map,
                out,
                _vt(valid),
                top_k,
                write_unrouted_rows=write_unrouted_rows,
                config=cfg,
            )

        # Bitwise, so that unwritten (NaN) rows compare equal too.
        assert torch.equal(run(None).view(torch.int32), run(config).view(torch.int32))

    @pytest.mark.parametrize("write_unrouted_rows", [True, False])
    def test_zero_valid_tokens_writes_nothing(self, write_unrouted_rows):
        rows, hidden, top_k = 64, 256, 22
        gemm2, weights, inverse_map, _ = _finalize_inputs(rows, hidden, top_k, 64, 0, 8)
        out = torch.full((rows, hidden), float("nan"), device="cuda")
        trtllm_bf16.trtllm_finalize(
            gemm2, weights, inverse_map, out, _vt(0), top_k, write_unrouted_rows=write_unrouted_rows
        )
        assert torch.isnan(out).all()


class _FakeGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class _FakeZmq:
    """Reduces the local values with fixed peer values, like the EP leader would."""

    def __init__(self, peers):
        self.peers = peers  # list of (token_count, is_non_decode, no_graph, eager_tokens)

    def sync_all_reduce_max_and_sum(self, max_vals, sum_vals):
        rows = [(*max_vals, *sum_vals)] + self.peers
        return tuple(max(r[i] for r in rows) for i in range(3)), (sum(r[3] for r in rows),)


class TestEpMoeRowBoundConsensus:

    @pytest.fixture(autouse=True)
    def _group_size(self, monkeypatch):
        # get_pg_size returns 1 without an initialized process group.
        from megatron.core.inference import batch_dimensions_utils

        monkeypatch.setattr(batch_dimensions_utils, "get_pg_size", lambda g: g.size())

    def test_all_decode_reduces_max_and_sum(self):
        local = InferenceBatchDimensions(token_count=5, prefill_req_count=0, decode_req_count=5)
        zmq = _FakeZmq([(9, 0, 0, 64), (3, 0, 0, 64), (1, 0, 0, 64)])
        state = InferenceBatchDimensions.sync_ep_moe_row_bound_state(
            local, True, 64, ep_group=_FakeGroup(4), ep_zmq_communicator=zmq
        )
        assert state == (9, False, False, 256)

    def test_prefill_with_graphs_everywhere(self):
        local = InferenceBatchDimensions(token_count=5, prefill_req_count=0, decode_req_count=5)
        zmq = _FakeZmq([(256, 1, 0, 256), (3, 0, 0, 64)])
        state = InferenceBatchDimensions.sync_ep_moe_row_bound_state(
            local, True, 64, ep_group=_FakeGroup(3), ep_zmq_communicator=zmq
        )
        assert state == (256, True, False, 256 + 128)

    def test_any_rank_without_graph(self):
        local = InferenceBatchDimensions(token_count=5, prefill_req_count=0, decode_req_count=5)
        zmq = _FakeZmq([(4096, 1, 1, 4096), (3, 0, 0, 64)])
        state = InferenceBatchDimensions.sync_ep_moe_row_bound_state(
            local, True, 64, ep_group=_FakeGroup(3), ep_zmq_communicator=zmq
        )
        assert state == (4096, True, True, 4096 + 128)

    def test_ep1_is_local(self):
        local = InferenceBatchDimensions(token_count=5, prefill_req_count=1, decode_req_count=4)
        state = InferenceBatchDimensions.sync_ep_moe_row_bound_state(
            local, False, 64, ep_group=_FakeGroup(1)
        )
        assert state == (5, True, True, 64)
