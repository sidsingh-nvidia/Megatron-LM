# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Grid cap of the symmetric-memory collectives: the EP heuristic, the guarded setter, and the
inference_moe_nvl_max_num_blocks config knob."""

import pytest

from megatron.core.inference.communication.torch_symm_triton.kernel_config import (
    SYMM_MEM_KERNEL_CONFIG,
    UNIFORM_COLLECTIVES_MAX_NUM_BLOCKS,
    SymmMemKernelConfig,
    ep_collectives_max_num_blocks,
)
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from tests.unit_tests.inference.test_moe_combine_precision_config import _make_config

_DEFAULT_MAX_NUM_BLOCKS = SymmMemKernelConfig().max_num_blocks


@pytest.fixture
def empty_registry():
    """Run with no symmetric-memory buffer registered; restore the registry and cap after."""
    saved_buffers = dict(SymmetricMemoryManager._buffers)
    saved_cap = SYMM_MEM_KERNEL_CONFIG.max_num_blocks
    SymmetricMemoryManager._buffers.clear()
    yield
    SymmetricMemoryManager._buffers.clear()
    SYMM_MEM_KERNEL_CONFIG._set_max_num_blocks(saved_cap)
    SymmetricMemoryManager._buffers.update(saved_buffers)


@pytest.mark.parametrize(
    "num_sms, ep_size, expected",
    [
        (152, 2, 608),
        (152, 4, 608),
        (152, 8, 608),
        (152, 16, 152),
        (152, 64, 152),
        (132, 8, 528),
        (132, 32, 132),
    ],
)
def test_ep_collectives_max_num_blocks(num_sms, ep_size, expected):
    assert ep_collectives_max_num_blocks(num_sms, ep_size) == expected


def test_uniform_cap_stays_at_128_and_never_exceeds_max_num_blocks():
    assert SymmMemKernelConfig(max_num_blocks=608).uniform_max_num_blocks == 128
    assert SymmMemKernelConfig(max_num_blocks=64).uniform_max_num_blocks == 64
    assert UNIFORM_COLLECTIVES_MAX_NUM_BLOCKS == _DEFAULT_MAX_NUM_BLOCKS


def test_signal_pad_bytes_follows_max_num_blocks():
    assert SymmMemKernelConfig(max_num_blocks=608).signal_pad_bytes(8) == 608 * 8 * 4


def test_lower_max_num_blocks_before_allocation(empty_registry):
    SymmetricMemoryManager.set_max_num_blocks(96)
    assert SYMM_MEM_KERNEL_CONFIG.max_num_blocks == 96


def test_raise_max_num_blocks_before_allocation(empty_registry):
    symm_mem = pytest.importorskip("torch.distributed._symmetric_memory")
    if not hasattr(symm_mem, "set_signal_pad_size"):
        pytest.skip("raising the cap needs set_signal_pad_size (torch >= 2.10)")
    SymmetricMemoryManager.set_max_num_blocks(608)
    assert SYMM_MEM_KERNEL_CONFIG.max_num_blocks == 608
    assert SYMM_MEM_KERNEL_CONFIG.uniform_max_num_blocks == 128


def test_set_max_num_blocks_refused_after_allocation(empty_registry):
    before = SYMM_MEM_KERNEL_CONFIG.max_num_blocks
    SymmetricMemoryManager._buffers["fake"] = object()
    # Same value: a no-op, so a second context with the same EP size is fine.
    SymmetricMemoryManager.set_max_num_blocks(before)
    with pytest.raises(AssertionError, match="already have signal pads"):
        SymmetricMemoryManager.set_max_num_blocks(before // 2)
    assert SYMM_MEM_KERNEL_CONFIG.max_num_blocks == before


def test_check_num_blocks_uses_current_cap(empty_registry):
    SymmetricMemoryManager.set_max_num_blocks(64)
    SYMM_MEM_KERNEL_CONFIG.check_num_blocks(64)
    with pytest.raises(AssertionError, match="max_num_blocks=64"):
        SYMM_MEM_KERNEL_CONFIG.check_num_blocks(65)


def test_config_knob_defaults_to_heuristic():
    assert _make_config().inference_moe_nvl_max_num_blocks is None
    assert _make_config(inference_moe_nvl_max_num_blocks=128).inference_moe_nvl_max_num_blocks == 128


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"inference_moe_nvl_max_num_blocks": 0}, "must be > 0"),
        (
            {"inference_moe_nvl_max_num_blocks": 128, "inference_moe_token_dispatcher_type": "nccl"},
            "only to the 'nvls' and 'nvl_a2a'",
        ),
    ],
)
def test_invalid_config_knob_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_config(**overrides)
