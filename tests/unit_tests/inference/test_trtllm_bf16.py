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
