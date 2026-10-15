# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Launch limits shared by the symmetric-memory Triton collectives.

Every kernel that calls symm_mem_sync gives each (block, peer) pair one int32 slot in
the symmetric-memory signal pad, at block_id * world_size + rank. SymmetricMemoryBuffer
sizes the pad from max_num_blocks when each buffer is allocated, so no launch may use
more than max_num_blocks blocks.

Do not autotune the number of blocks (the grid size, or BLOCK_SIZE where the grid is
derived from it) with triton.autotune or any other per-rank tuning:

  1. A grid larger than max_num_blocks writes barrier slots past the end of the
     signal pad.
  2. Autotuning picks a config from local timings, so ranks can choose different
     grids. A rank then waits on barrier slots that its peers never signal, and the
     collective hangs.
  3. The autotune benchmark launches the kernel a timing-dependent number of times on
     each rank, so barrier calls no longer pair up across ranks.

Grid sizes must be a deterministic function of values that are identical on every
rank, capped at max_num_blocks.
"""

from dataclasses import dataclass

# See ep_collectives_max_num_blocks.
_SMALL_EP_MAX_SIZE = 8
_SMALL_EP_BLOCKS_PER_SM = 4


def ep_collectives_max_num_blocks(num_sms: int, ep_size: int) -> int:
    """Grid cap for the EP collectives that process one token per block.

    These are the NVLS all-gather-v / reduce-scatter-v and the all-to-all-v dispatch /
    combine kernels. Each block has one token in flight, so the bytes in flight per GPU
    are about num_blocks x (ranks the token reaches) x row size. At small EP a token
    reaches few ranks (about 4 at EP=4 with top-22 of 512 experts), so one block per SM
    leaves NVLink underused and large batches become latency-bound: on GB300 at EP=4, four
    blocks per SM cut the all-to-all-v layer time at 128K tokens from 1.62 ms to 0.73 ms.
    From EP=16 on, one block per SM already keeps the link busy, and extra blocks only
    add barrier traffic.

    Depends only on values that are identical on every rank, as the grid must be.

    Args:
        num_sms: streaming multiprocessors per GPU.
        ep_size: expert-parallel world size.
    """
    blocks_per_sm = _SMALL_EP_BLOCKS_PER_SM if ep_size <= _SMALL_EP_MAX_SIZE else 1
    return blocks_per_sm * num_sms


@dataclass
class SymmMemKernelConfig:
    """Launch limits for the symmetric-memory collectives.

    The inference context sets max_num_blocks once at startup, before any symmetric-memory
    buffer is allocated. Do not change it after that: the signal pads are sized from
    max_num_blocks at allocation time, so raising it would let launches overflow pads that
    already exist.
    """

    max_num_blocks: int = 128
    """Grid cap for every kernel that synchronizes through symm_mem_sync."""

    max_block_size: int = 1024
    """Threads per block cap; each thread moves one 64- or 128-bit chunk."""

    warp_size: int = 32

    def signal_pad_bytes(self, world_size: int) -> int:
        """Signal pad bytes symm_mem_sync needs for a group of world_size ranks."""
        # One int32 slot per (block, peer), at block_id * world_size + rank.
        return self.max_num_blocks * world_size * 4

    def check_num_blocks(self, num_blocks: int) -> None:
        """Assert a launch fits in the signal pads sized from max_num_blocks."""
        assert num_blocks <= self.max_num_blocks, (
            f"Launching {num_blocks} blocks, but symmetric-memory signal pads are sized for "
            f"at most max_num_blocks={self.max_num_blocks}."
        )


SYMM_MEM_KERNEL_CONFIG = SymmMemKernelConfig()
