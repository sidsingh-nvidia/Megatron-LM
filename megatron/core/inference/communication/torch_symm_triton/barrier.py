# Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from: https://github.com/meta-pytorch/kraken.git

from unittest.mock import MagicMock

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = MagicMock()
    tl = MagicMock()
    triton.jit = null_decorator

from .utils import get_flat_bid, get_flat_tid, sync_threads


@triton.jit
def _send_signal(addrs, sem: tl.constexpr):
    tl.inline_asm_elementwise(
        f"""
        {{
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            send_signal:
                atom.global.{sem}.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                setp.eq.u32 %p0, %tmp32_0, 0;
                @!%p0 bra send_signal;
        }}
        """,
        "=r, l",
        [addrs],
        dtype=addrs.dtype,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_signal(addrs, sem: tl.constexpr):
    tl.inline_asm_elementwise(
        f"""
        {{
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            wait_signal:
                atom.global.sys.{sem}.cas.b32 %tmp32_0, [$1], 1, 0;
                setp.eq.u32 %p0, %tmp32_0, 1;
                @!%p0 bra wait_signal;
        }}
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def symm_mem_sync(
    signal_pad_ptrs,
    block_id,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    hasPreviousMemAccess: tl.constexpr = False,
    hasSubsequentMemAccess: tl.constexpr = False,
):
    """
    Synchronizes blocks with matching block_id across participating devices.

    Note: the function itself is not a system level barrier/fence. It is a
    building block for expressing different synchronization patterns.

    Pattern 0: Ensures that all writes to symm_mem buffers from previous
    kernels across all devices are visible to the current kernel:

        symm_mem_sync(..., hasPreviousMemAccess=False, hasSubsequentMemAccess=True)

    Pattern 1: Ensures that all writes to symm_mem buffers from the current
    block are visible to all remote blocks with matching blockIdx:

        symm_mem_sync(..., hasPreviousMemAccess=True, hasSubsequentMemAccess=True)

    Pattern 2: Ensures that symm_mem buffers read by the current kernel are safe
    for writing by subsequent kernels across all devices.

        symm_mem_sync(..., hasPreviousMemAccess=True, hasSubsequentMemAccess=False)

    CUDA graph friendliness:

        This barrier operates through atomic operations on a zero-filled signal
        pad, which resets to a zero-filled state after each successful
        synchronization. This design eliminates the need for incrementing a
        flag from host.
    """
    if block_id is None:
        block_id = get_flat_bid()
    flat_tid = get_flat_tid()

    remote_ranks = tl.arange(0, world_size)
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks).to(tl.pointer_type(tl.uint32))
    send_addrs = remote_signal_pad_addrs + block_id * world_size + rank

    local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint32))
    wait_addrs = local_signal_pad_addr + block_id * world_size + remote_ranks

    if flat_tid < world_size:
        _send_signal(send_addrs, "release" if hasPreviousMemAccess else "relaxed")
        _wait_signal(wait_addrs, "acquire" if hasSubsequentMemAccess else "relaxed")


# ── One barrier per rank per kernel, with multicast arrivals ─────────────────────────────────
#
# symm_mem_sync above runs in every block and sends one remote atomic per peer per block, so its
# cost grows with the grid (13.6 us at 152 blocks on 64 ranks). The barriers below synchronize
# each rank once per kernel: one multimem.red arrival per rank, waited on in local memory.
#
# Every SymmetricMemoryBuffer reserves BARRIER_STATE_BYTES after its data, zeroed at allocation;
# barrier_ptrs(hdl) gives its multicast address (a write lands on every rank's copy) and this
# rank's local address. Layout, as uint32 words (ws = world size):
#
#     words: [ end slot 0 .. end slot ws-1 | start slot 0 .. start slot ws-1 | start ctr | end ctr ]
#             ^ slot q counts rank q's       ^ same, for start barriers         ^ local block
#               arrivals at end barriers                                         sign-in counters
#
# End barrier (symm_mem_sync_end), 3 ranks, rank 1's view:
#
#     rank 1's blocks   b0   b1   b2 ... b151      every block: end ctr += 1, then exits ...
#                        \    |    |    /
#                         end ctr reaches 152      ... except the last one to sign in, which:
#                                 |
#        1. multimem.red.release: end slot 1 += 1 on ranks 0, 1 and 2 (one instruction)
#                                 |
#        2. threads 0..2 wait on rank 1's own copy, then take one arrival each:
#               end slot 0 > 0 ?  -1     (rank 0 has arrived)
#               end slot 1 > 0 ?  -1     (rank 1: its own add, step 1)
#               end slot 2 > 0 ?  -1     (rank 2 has arrived)
#
#     rank 1's end slots over one barrier:
#                         start   1 lands   0 lands   2 lands   consumed
#        slot [0,1,2]:  [0,0,0] -> [0,1,0] -> [1,1,0] -> [1,1,1] -> [0,0,0]
#
#     A fast rank may arrive for the next barrier before a slow one consumes this one: its slot
#     goes 1 -> 2, the slow rank takes one, and 1 stays for the next barrier. It cannot get further
#     ahead: passing the next barrier needs the slow rank's arrival. So slots hold 0, 1 or 2.
#
# Start barrier (symm_mem_sync_start, relaxed): block 0 arrives for the rank (start slot rank += 1
# on every rank); every block waits, without consuming, until each peer's start slot is nonzero,
# skipping its own rank's slot, so no block ever waits on a block of its own rank. The last block
# to sign in on the start counter then takes one arrival from every start slot.
#
#     rank 1:  block 0: start slot 1 += 1 everywhere --.
#              blocks 0..151: wait start slot 0 > 0,    |  then the last to sign in takes one
#                             wait start slot 2 > 0  <--'  from start slots 0, 1 and 2
#
# Only rank q adds to slot q and every arrival is taken exactly once, so all slots return to zero:
# no state outlives a barrier. Start and end use separate slots because their arrivals come from
# different threads (block 0 vs the last block), which need not land in order; arrivals of one
# kind come from successive kernels, which do. Collectives on the same buffer must not run
# concurrently.

BARRIER_STATE_BYTES = 1024
MAX_BARRIER_WORLD_SIZE = BARRIER_STATE_BYTES // 8 - 1

# Handle id -> (handle, byte offset of its barrier state); the handle is kept alive with its entry.
_BARRIER_STATE_OFFSETS: dict = {}


def register_barrier_state(symm_mem_hdl, offset: int) -> None:
    """Record where a symmetric buffer's (zeroed) barrier state starts, in bytes."""
    _BARRIER_STATE_OFFSETS[id(symm_mem_hdl)] = (symm_mem_hdl, offset)


def barrier_ptrs(symm_mem_hdl) -> tuple:
    """(multicast, local) addresses of a symmetric buffer's barrier state, as passed to
    symm_mem_sync_start / symm_mem_sync_end."""
    _, offset = _BARRIER_STATE_OFFSETS[id(symm_mem_hdl)]
    return (
        symm_mem_hdl.multicast_ptr + offset,
        symm_mem_hdl.buffer_ptrs[symm_mem_hdl.rank] + offset,
    )


@triton.jit
def _multicast_add_one(mc_addr, sem: tl.constexpr):
    """Atomically add 1 to this word on every rank with one multicast reduction."""
    tl.inline_asm_elementwise(
        f"""
        {{
            multimem.red.{sem}.sys.global.add.u32 [$1], 1;
            mov.u32 $0, 0;
        }}
        """,
        "=r, l",
        [mc_addr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_nonzero(addrs, skip, sem: tl.constexpr):
    """Spin until this slot holds an arrival (read only), unless skip is nonzero."""
    tl.inline_asm_elementwise(
        f"""
        {{
            .reg .u32   %v;
            .reg .pred  %p, %s;

            setp.ne.u32 %s, $2, 0;
            @%s bra done_nonzero;
            wait_nonzero:
                ld.{sem}.sys.global.u32 %v, [$1];
                setp.eq.u32 %p, %v, 0;
                @%p bra wait_nonzero;
            done_nonzero:
            mov.u32 $0, 0;
        }}
        """,
        "=r, l, r",
        [addrs, skip],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_and_take_one(addrs, sem: tl.constexpr):
    """Spin until this slot holds an arrival, then atomically take one."""
    tl.inline_asm_elementwise(
        f"""
        {{
            .reg .u32   %v, %minus_one;
            .reg .pred  %p;

            wait_arrival:
                ld.{sem}.sys.global.u32 %v, [$1];
                setp.eq.u32 %p, %v, 0;
                @%p bra wait_arrival;
            mov.u32 %minus_one, 0xFFFFFFFF;
            red.relaxed.sys.global.add.u32 [$1], %minus_one;
            mov.u32 $0, 0;
        }}
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def symm_mem_sync_start(
    barrier_mc_ptr,
    barrier_local_ptr,
    num_blocks,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """
    Start-of-kernel barrier with relaxed arrival (replaces symm_mem_sync(False, False) at the
    start of a kernel): every block waits until every rank has entered the kernel.

    Block 0 arrives for the rank with one multicast add to its start slot. Every block then
    waits, in local memory and without consuming, until each peer's start slot holds an arrival;
    it skips this rank's own slot, so no block waits on another block of its own rank and the
    grid needs no co-residency. The last block to sign in afterwards takes one arrival from every
    start slot (its own included: block 0 arrived before it signed in).
    """
    flat_tid = get_flat_tid()
    words = barrier_local_ptr.to(tl.pointer_type(tl.uint32))
    start_slots = words + world_size
    if get_flat_bid() == 0:
        if flat_tid == 0:
            _multicast_add_one(barrier_mc_ptr.to(tl.pointer_type(tl.uint32)) + world_size + rank,
                               "relaxed")
    peers = tl.arange(0, world_size)
    if flat_tid < world_size:
        _wait_nonzero(start_slots + peers, (peers == rank).to(tl.uint32), "relaxed")
    sync_threads()
    counter = (words + 2 * world_size).to(tl.pointer_type(tl.int32))
    arrived = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
    if arrived == num_blocks - 1:
        tl.store(counter, 0)
        if flat_tid < world_size:
            _wait_and_take_one(start_slots + peers, "relaxed")


@triton.jit
def symm_mem_sync_end(
    barrier_mc_ptr,
    barrier_local_ptr,
    num_blocks,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    hasPreviousMemAccess: tl.constexpr = True,
    hasSubsequentMemAccess: tl.constexpr = True,
):
    """
    End-of-kernel barrier (replaces symm_mem_sync(True, True) at the end of a kernel): once
    every rank's kernel has passed it, this kernel's writes on every rank are visible and the
    buffers it read are free to reuse.

    Every block signs in on the local end counter; the last block to sign in resets it, adds 1
    to this rank's end slot on every rank with one multimem.red (release: covers every block's
    writes, published to it by their acq_rel sign-ins), then takes one arrival from each end
    slot of the local copy (acquire). Other blocks exit without waiting, so the grid needs no
    co-residency; the kernel boundary orders the next kernel after the barrier. A nonzero slot
    means that rank's arrival for this barrier landed: one rank's end arrivals come from
    successive kernels and land in order. num_blocks: as for symm_mem_sync_start.
    """
    sync_threads()
    words = barrier_local_ptr.to(tl.pointer_type(tl.uint32))
    counter = (words + 2 * world_size + 1).to(tl.pointer_type(tl.int32))
    arrived = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
    if arrived == num_blocks - 1:
        tl.store(counter, 0)
        flat_tid = get_flat_tid()
        if flat_tid == 0:
            _multicast_add_one(barrier_mc_ptr.to(tl.pointer_type(tl.uint32)) + rank,
                               "release" if hasPreviousMemAccess else "relaxed")
        peers = tl.arange(0, world_size)
        if flat_tid < world_size:
            _wait_and_take_one(words + peers, "acquire" if hasSubsequentMemAccess else "relaxed")
