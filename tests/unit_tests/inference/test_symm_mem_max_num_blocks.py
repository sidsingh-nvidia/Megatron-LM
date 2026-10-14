# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Grid cap heuristic of the EP symmetric-memory collectives."""

import pytest

from megatron.core.inference.communication.torch_symm_triton.kernel_config import (
    SymmMemKernelConfig,
    ep_collectives_max_num_blocks,
)


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


def test_signal_pad_bytes_follows_max_num_blocks():
    assert SymmMemKernelConfig(max_num_blocks=608).signal_pad_bytes(8) == 608 * 8 * 4
