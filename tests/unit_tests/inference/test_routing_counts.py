# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Diagnostic per-step expert routing counts (MCORE_ROUTING_COUNTS): the flag-gated counting
kernel, its CUDA-graph behavior, and the log summary."""

import json

import pytest
import torch

from megatron.core.inference.moe.routing_counts import RoutingCounter

NUM_EXPERTS = 16
TOP_K = 4
EP_SIZE = 2  # 8 experts per EP rank


@pytest.fixture
def counter(monkeypatch, tmp_path):
    """A fresh RoutingCounter with two registered layers, dumping to a temp file."""
    path = tmp_path / "counts.jsonl"
    monkeypatch.setenv("MCORE_ROUTING_COUNTS_PATH", str(path))
    for attr, value in (("counts", None), ("flag", None), ("num_experts", 0), ("ep_size", 1),
                        ("num_layers", 0), ("path", None)):
        monkeypatch.setattr(RoutingCounter, attr, value)
    assert RoutingCounter.register_layer(NUM_EXPERTS, EP_SIZE) == 0
    assert RoutingCounter.register_layer(NUM_EXPERTS, EP_SIZE) == 1
    yield RoutingCounter, path


def _routing(rows, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(0, NUM_EXPERTS, (rows, TOP_K), generator=g, device="cuda")


def _reference(routing, valid):
    ids = routing[:valid].flatten()
    ids = ids[ids >= 0]
    return torch.bincount(ids, minlength=NUM_EXPERTS).tolist()


def _pack(routing):
    """TRT-LLM packed routing: expert << 16 | bf16 prob bits (prob 0.5 here); -1 stays -1 << 16."""
    prob_bits = torch.full_like(routing, 0x3F00)
    return ((routing.to(torch.int32) << 16) | prob_bits.to(torch.int32)).contiguous()


@pytest.mark.parametrize("packed", [False, True])
def test_counts_live_rows_only_when_armed(counter, packed):
    rc, _ = counter
    routing = _routing(64)
    routing[5, 1] = -1  # a masked (padding) slot
    valid = torch.tensor([40], dtype=torch.int32, device="cuda")
    data = _pack(routing) if packed else routing

    rc.count(0, data, valid, packed=packed)  # flag off: no-op
    assert rc.read_and_reset() == [[0] * NUM_EXPERTS] * 2

    rc.arm(True)
    rc.count(0, data, valid, packed=packed)
    rc.count(1, data, valid, packed=packed)
    rc.arm(False)
    counts = rc.read_and_reset()
    assert counts[0] == _reference(routing, 40) == counts[1]
    assert sum(counts[0]) == 40 * TOP_K - 1
    assert rc.read_and_reset() == [[0] * NUM_EXPERTS] * 2  # reset


def test_cuda_graph_replay_follows_flag(counter):
    rc, _ = counter
    routing = _routing(32, seed=1)
    valid = torch.tensor([32], dtype=torch.int32, device="cuda")
    rc.count(0, routing, valid, packed=False)  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rc.count(0, routing, valid, packed=False)

    graph.replay()  # flag off
    assert rc.read_and_reset()[0] == [0] * NUM_EXPERTS
    rc.arm(True)
    graph.replay()
    valid.fill_(10)  # the graph reads the live count at replay
    graph.replay()
    rc.arm(False)
    graph.replay()
    expected = [a + b for a, b in zip(_reference(routing, 32), _reference(routing, 10))]
    assert rc.read_and_reset()[0] == expected


def test_summary_and_dump(counter):
    rc, path = counter
    balanced = [10] * NUM_EXPERTS
    skewed = [10] * NUM_EXPERTS
    skewed[0] = 50  # expert 0 is hot
    line = rc.summary([balanced, skewed])
    # EP2 (8 experts per rank): balanced 1.00; skewed loads 120 vs 80 -> max/mean 1.20.
    assert line == (
        "routing: 160 routed rows/layer | EP rank load max/mean 1.10 (worst layer 1.20) "
        "| per-layer max/mean 1.00,1.20"
    )
    assert rc.summary([[0] * NUM_EXPERTS]) == "routing: no tokens counted"

    rc.dump(7, 12.5, [balanced, skewed])
    rec = json.loads(path.read_text().splitlines()[-1])
    assert rec["step"] == 7 and rec["routed_rows"] == 10 * NUM_EXPERTS
    assert rec["counts"] == [balanced, skewed]
