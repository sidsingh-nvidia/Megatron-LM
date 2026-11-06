# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-step expert routing counts for diagnosing expert-parallel load imbalance.

Enabled with MCORE_ROUTING_COUNTS=1. The NVLS dispatcher on expert-parallel rank 0 then adds a
counting kernel after each MoE layer's all-gather-v: it histograms the gathered routing (every
token of the EP group, so one rank sees the whole step) into a fixed [layers, experts] int32
buffer with atomics. The kernel is captured in the CUDA graphs but returns at once unless a device
flag is set; the engine sets the flag around the steps it times (logging steps), then reads and
clears the counts. Other ranks never launch the kernel.

If MCORE_ROUTING_COUNTS_PATH is set, each counted step is also appended to that file as one JSON
line: {"step", "step_time_ms", "routed_rows", "counts": [[per-expert count] per MoE layer]}.
"""

import json
import os
from typing import List, Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

_MAX_LAYERS = 256  # upper bound on MoE layers (incl. MTP) per model; sizes the counts buffer


def enabled() -> bool:
    """Whether routing counting is requested (MCORE_ROUTING_COUNTS=1)."""
    return os.environ.get("MCORE_ROUTING_COUNTS", "0") == "1"


if HAVE_TRITON:

    @triton.jit
    def _count_routing_kernel(
        routing_ptr, valid_tokens_ptr, flag_ptr, counts_ptr, max_elems,
        TOP_K: tl.constexpr, PACKED: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """counts[expert] += 1 for every live (row < valid_tokens, expert >= 0) routing slot,
        if the flag is set. PACKED routing is TRT-LLM's int32 (expert << 16 | bf16 prob)."""
        if tl.load(flag_ptr) == 0:
            return
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = (offs < max_elems) & (offs < tl.load(valid_tokens_ptr) * TOP_K)
        vals = tl.load(routing_ptr + offs, mask=live, other=-1)
        experts = (vals >> 16) if PACKED else vals
        experts = experts.to(tl.int32)
        tl.atomic_add(counts_ptr + experts, 1, mask=live & (experts >= 0), sem="relaxed")


class RoutingCounter:
    """Class-level state shared by every MoE layer's dispatcher (one model per process)."""

    counts: Optional[torch.Tensor] = None  # [_MAX_LAYERS, num_experts] int32
    flag: Optional[torch.Tensor] = None  # [1] int32; nonzero = count this forward
    num_experts: int = 0
    ep_size: int = 1  # this run's expert-parallel size; experts are placed contiguously
    num_layers: int = 0  # MoE layers registered so far
    path: Optional[str] = None

    @classmethod
    def register_layer(cls, num_experts: int, ep_size: int) -> int:
        """Allocate the buffers on first use and return the next MoE layer's slot."""
        if cls.counts is None:
            device = torch.cuda.current_device()
            cls.num_experts = num_experts
            cls.ep_size = ep_size
            cls.counts = torch.zeros(_MAX_LAYERS, num_experts, dtype=torch.int32, device=device)
            cls.flag = torch.zeros(1, dtype=torch.int32, device=device)
            cls.path = os.environ.get("MCORE_ROUTING_COUNTS_PATH") or None
        assert num_experts == cls.num_experts, "all MoE layers must have the same expert count"
        assert cls.num_layers < _MAX_LAYERS, "raise _MAX_LAYERS"
        cls.num_layers += 1
        return cls.num_layers - 1

    @classmethod
    def active(cls) -> bool:
        """Whether this process counts (it registered at least one layer)."""
        return cls.counts is not None

    @classmethod
    def count(cls, layer: int, routing: torch.Tensor, valid_tokens: torch.Tensor, packed: bool):
        """Launch the (flag-gated) counting kernel on the gathered [rows, top_k] routing."""
        rows, top_k = routing.shape
        block = 1024
        _count_routing_kernel[(triton.cdiv(rows * top_k, block),)](
            routing, valid_tokens, cls.flag, cls.counts[layer], rows * top_k,
            TOP_K=top_k, PACKED=packed, BLOCK=block,
        )

    @classmethod
    def arm(cls, on: bool) -> None:
        """Set or clear the device flag (stream-ordered, outside any CUDA graph)."""
        if cls.flag is not None:
            cls.flag.fill_(1 if on else 0)

    @classmethod
    def read_and_reset(cls) -> List[List[int]]:
        """Counts of the forwards counted since the last call, [layer][expert]; then zero them.
        Synchronizes with the stream."""
        counts = cls.counts[: cls.num_layers].cpu().tolist()
        cls.counts.zero_()
        return counts

    @classmethod
    def summary(cls, counts: List[List[int]]) -> str:
        """One log line for this run's EP layout: routed rows per layer and the busiest EP rank's
        routed rows over the mean, averaged over layers, at the worst layer, and for every MoE
        layer in order (comma-separated, for parsing)."""
        if not counts or not any(any(layer) for layer in counts):
            return "routing: no tokens counted"
        ep, per = cls.ep_size, cls.num_experts // cls.ep_size
        ratios = []
        for layer in counts:
            loads = [sum(layer[r * per : (r + 1) * per]) for r in range(ep)]
            mean = sum(loads) / ep
            # nan for a layer that routed nothing, so the list stays aligned with layer order.
            ratios.append(max(loads) / mean if mean > 0 else float("nan"))
        live = [r for r in ratios if r == r]
        return (
            f"routing: {sum(counts[0])} routed rows/layer | EP rank load max/mean "
            f"{sum(live) / len(live):.2f} (worst layer {max(live):.2f}) | per-layer max/mean "
            + ",".join(f"{r:.2f}" for r in ratios)
        )

    @classmethod
    def dump(cls, step: int, step_time_ms: float, counts: List[List[int]]) -> None:
        """Append one counted step to MCORE_ROUTING_COUNTS_PATH, if set."""
        if cls.path is None:
            return
        with open(cls.path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "step": step,
                        "step_time_ms": step_time_ms,
                        "routed_rows": sum(counts[0]) if counts else 0,
                        "counts": counts,
                    }
                )
                + "\n"
            )
