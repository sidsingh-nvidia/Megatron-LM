# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for megatron.core.inference.moe.sigmoid_topk (tl.topk sigmoid router kernel)."""

import pytest
import torch

from megatron.core.inference.moe.sigmoid_topk import sigmoid_topk_routing
from megatron.core.transformer.moe.router import InferenceTopKRouter


def _compiled(logits, bias, topk, scale):
    return InferenceTopKRouter._compiled_topk_routing(
        logits, topk, False, None, None, scale, "sigmoid", bias, False, None, True, None
    )


def _dense(probs, idx, num_experts):
    return torch.zeros(probs.shape[0], num_experts, device=probs.device).scatter_(1, idx, probs)


@pytest.mark.internal
class TestSigmoidTopk:

    # Nemotron 3 Ultra/Super (512 experts, top-22) and nano-3.5 (128, top-6).
    @pytest.mark.parametrize("num_experts,topk,scale", [(512, 22, 5.0), (128, 6, 2.5), (96, 8, None)])
    @pytest.mark.parametrize("num_tokens", [1, 2, 3, 64, 1000])
    def test_matches_compiled_router(self, num_experts, topk, scale, num_tokens):
        torch.manual_seed(num_tokens)
        logits = torch.randn(num_tokens, num_experts, device="cuda") * 2
        bias = torch.randn(num_experts, device="cuda") * 0.05
        with torch.inference_mode():
            ref_p, ref_i = _compiled(logits, bias, topk, scale)
            p, i = sigmoid_topk_routing(logits, bias, topk, scale)
        assert p.shape == ref_p.shape and i.shape == ref_i.shape and i.dtype == torch.int64
        assert torch.equal(torch.sort(i, dim=1).values, torch.sort(ref_i, dim=1).values)
        torch.testing.assert_close(
            _dense(p, i, num_experts), _dense(ref_p, ref_i, num_experts), rtol=1e-5, atol=1e-7
        )
        # Sorted by biased score, largest first.
        biased = (torch.sigmoid(logits) + bias).gather(1, i)
        assert torch.all(biased[:, :-1] >= biased[:, 1:])

    def test_ties_pick_lower_expert(self):
        logits = torch.zeros(4, 64, device="cuda")
        bias = torch.zeros(64, device="cuda")
        _, i = sigmoid_topk_routing(logits, bias, 8)
        assert torch.equal(i, torch.arange(8, device="cuda").expand(4, 8))

    def test_empty_batch(self):
        p, i = sigmoid_topk_routing(torch.empty(0, 128, device="cuda"), torch.zeros(128, device="cuda"), 6)
        assert p.shape == (0, 6) and i.shape == (0, 6)

    def test_cuda_graph_replay_reads_new_inputs(self):
        logits = torch.randn(16, 512, device="cuda")
        bias = torch.randn(512, device="cuda") * 0.05
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            sigmoid_topk_routing(logits, bias, 22, 5.0)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            p, i = sigmoid_topk_routing(logits, bias, 22, 5.0)
        for seed in range(3):
            torch.manual_seed(seed)
            logits.copy_(torch.randn_like(logits))
            graph.replay()
            ref_p, ref_i = sigmoid_topk_routing(logits.clone(), bias, 22, 5.0)
            assert torch.equal(i, ref_i) and torch.equal(p, ref_p)
