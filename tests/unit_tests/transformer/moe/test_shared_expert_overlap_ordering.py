# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Launch order of the inference shared-expert overlap around the routed dispatch/combine."""

from types import SimpleNamespace

from megatron.core.transformer.moe.moe_layer import MoELayer


def _layer(order, pending_input):
    shared_experts = SimpleNamespace(
        inference_overlap_launch=lambda hidden: order.append("shared_launch") or "shared_out",
        inference_overlap_wait_before_combine=lambda: order.append("wait_shared"),
    )
    dispatcher = SimpleNamespace(
        token_dispatch=lambda hidden, probs: order.append("dispatch") or (hidden, probs),
        token_combine=lambda output: order.append("combine") or output,
    )
    return SimpleNamespace(
        config=SimpleNamespace(overlap_dispatch_backward_with_experts_wgrad=False),
        token_dispatcher=dispatcher,
        shared_experts=shared_experts,
        _pending_shared_expert_input=pending_input,
        _overlapped_shared_expert_output=None,
    )


def test_shared_expert_launches_after_dispatch_and_combine_waits_for_it():
    """The shared expert's TP collectives run between the dispatch and the combine."""
    order = []
    layer = _layer(order, pending_input="hidden")

    MoELayer.dispatch(layer, "hidden", "probs")
    assert layer._pending_shared_expert_input is None
    assert layer._overlapped_shared_expert_output == "shared_out"
    MoELayer.combine(layer, "routed")

    assert order == ["dispatch", "shared_launch", "wait_shared", "combine"]


def test_no_shared_expert_overlap_leaves_dispatch_and_combine_alone():
    order = []
    layer = _layer(order, pending_input=None)

    MoELayer.dispatch(layer, "hidden", "probs")
    MoELayer.combine(layer, "routed")

    assert order == ["dispatch", "combine"]
