# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Decode context parallel (DCP) group layout of inference shards."""

import pytest

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.inference.shards import (
    get_decode_context_parallel_size,
    register_decode_context_parallel_view,
)


@pytest.mark.parametrize(
    "tp, num_kv_heads, expected",
    [(8, 2, 4), (8, 1, 8), (4, 2, 2), (8, 8, 1), (4, 8, 1), (8, None, 1), (1, 2, 1)],
)
def test_dcp_size(tp, num_kv_heads, expected):
    assert get_decode_context_parallel_size(tp, num_kv_heads) == expected


def test_dcp_size_rejects_tp_not_multiple_of_kv_heads():
    with pytest.raises(AssertionError):
        get_decode_context_parallel_size(8, 3)


def _dcp_and_tp_groups(monkeypatch, tp, cp, dp, pp, num_kv_heads, rank_offset, tp_pp_dp):
    dims = ["tp", "cp", "pp", "dp"] if tp_pp_dp else ["tp", "cp", "dp", "pp"]
    sizes = {"tp": tp, "cp": cp, "dp": dp, "pp": pp}
    monkeypatch.setenv("WORLD_SIZE", str(rank_offset + tp * cp * dp * pp))
    grid = HyperCommGrid([sizes[d] for d in dims], dims, rank_offset=rank_offset)
    dcp_size = get_decode_context_parallel_size(tp, num_kv_heads)
    register_decode_context_parallel_view(grid, dcp_size, tp_pp_dp)
    return dcp_size, grid.get_rank_enum("dcp", view="dcp"), grid.get_rank_enum("tp")


def test_dcp_groups_ultra_decode_pool(monkeypatch):
    """TP8 x DP2 with 2 KV heads: ranks 0-3 hold KV head 0, ranks 4-7 hold KV head 1."""
    _, dcp, _ = _dcp_and_tp_groups(monkeypatch, 8, 1, 2, 1, 2, 0, False)
    assert dcp == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]


@pytest.mark.parametrize("tp_pp_dp", [False, True])
@pytest.mark.parametrize("rank_offset", [0, 12])
@pytest.mark.parametrize(
    "tp, dp, pp, num_kv_heads", [(8, 2, 1, 2), (8, 1, 2, 2), (4, 2, 2, 1), (8, 2, 1, 8), (4, 3, 1, 8)]
)
def test_dcp_groups_are_the_tp_ranks_sharing_a_kv_head(
    monkeypatch, tp, dp, pp, num_kv_heads, rank_offset, tp_pp_dp
):
    """Each DCP group is the TP ranks that attention gives the same KV head:
    ``tp_rank // (tp_size // num_kv_heads)``; with at least as many KV heads as TP ranks every
    group is a single rank."""
    dcp_size, dcp, tp_groups = _dcp_and_tp_groups(
        monkeypatch, tp, 1, dp, pp, num_kv_heads, rank_offset, tp_pp_dp
    )
    expected = [
        group[head * dcp_size : (head + 1) * dcp_size]
        for group in tp_groups
        for head in range(tp // dcp_size)
    ]
    assert sorted(dcp) == sorted(expected)
    if num_kv_heads >= tp:
        assert all(len(group) == 1 for group in dcp)
