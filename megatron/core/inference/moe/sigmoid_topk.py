# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single-kernel sigmoid top-k routing with an expert bias, for inference.

Computes, per token, the top-k experts by sigmoid(logit) + bias and their routing probabilities
sigmoid(logit) / sum(selected sigmoid(logit)) [* scaling_factor] in one launch, replacing the
compiled sigmoid + bias, topk and gather/normalize kernels.

Selection uses Triton's bitonic `tl.topk` on packed int64 keys: the high 32 bits are the biased
score mapped to an order-preserving signed integer, the low 32 bits are (num_experts - 1 - expert),
so equal scores resolve to the lower expert index. `tl.topk` needs a power-of-two k, so the kernel
takes the top next_power_of_2(k) keys (sorted, largest first) and keeps the first k. The cost is a
fixed bitonic network over the expert dimension, independent of k.

Experts come out sorted by biased score (largest first), not in torch.topk(sorted=False) order;
the expert set is the same and probabilities match to fp32 rounding (the sum order differs).
"""

from typing import Optional
from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
    HAVE_TL_TOPK = hasattr(tl, "topk")
except ImportError:
    HAVE_TRITON = False
    HAVE_TL_TOPK = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


@triton.jit
def _sigmoid_topk_kernel(
    logits_ptr,
    expert_bias_ptr,
    probs_ptr,
    indices_ptr,
    num_tokens,
    scaling_factor,
    NUM_EXPERTS: tl.constexpr,
    EXPERTS_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_POW2: tl.constexpr,
    APPLY_SCALING: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    rows64 = rows.to(tl.int64)
    cols = tl.arange(0, EXPERTS_POW2)
    row_mask = rows < num_tokens
    col_mask = cols < NUM_EXPERTS
    row_ptrs = logits_ptr + rows64[:, None] * NUM_EXPERTS

    logits = tl.load(row_ptrs + cols[None, :], mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    bias = tl.load(expert_bias_ptr + cols, mask=col_mask, other=0.0)
    routing_scores = tl.sigmoid(logits) + bias[None, :]

    # Order-preserving int32 view of the fp32 scores (negative floats flip their magnitude bits),
    # packed above the reversed expert index; padding columns get the smallest possible key.
    bits = routing_scores.to(tl.int32, bitcast=True)
    keys = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    keys = tl.where(col_mask[None, :], keys, -2147483648)
    packed = (keys.to(tl.int64) << 32) | (NUM_EXPERTS - 1 - cols)[None, :].to(tl.int64)

    top = tl.topk(packed, TOPK_POW2, dim=1)  # [BLOCK_T, TOPK_POW2], largest first
    experts = (NUM_EXPERTS - 1) - (top & 0xFFFFFFFF).to(tl.int32)

    slots = tl.arange(0, TOPK_POW2)
    keep = row_mask[:, None] & (slots < TOPK)[None, :]
    selected = tl.load(row_ptrs + experts, mask=keep, other=0.0)
    scores = tl.where(keep, tl.sigmoid(selected), 0.0)
    probs = scores / (tl.sum(scores, axis=1)[:, None] + 1e-20)
    if APPLY_SCALING:
        probs = probs * scaling_factor

    out = rows64[:, None] * TOPK + slots[None, :]
    tl.store(probs_ptr + out, probs, mask=keep)
    tl.store(indices_ptr + out, experts.to(tl.int64), mask=keep)


def sigmoid_topk_routing(
    logits: torch.Tensor,
    expert_bias: torch.Tensor,
    topk: int,
    scaling_factor: Optional[float] = None,
    block_tokens: int = 1,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k of sigmoid(logits) + expert_bias; probabilities from sigmoid(logits), normalized.

    Args:
        logits: [num_tokens, num_experts] fp32 router logits, contiguous.
        expert_bias: [num_experts] fp32 routing bias (selection only).
        topk: experts per token, 1 < topk <= num_experts.
        scaling_factor: optional multiplier applied to the normalized probabilities.
        block_tokens: tokens per program (power of two).
        num_warps: warps per program.

    Returns:
        probs [num_tokens, topk] fp32 and indices [num_tokens, topk] int64, sorted by biased
        score (largest first).
    """
    assert HAVE_TL_TOPK, "sigmoid_topk_routing needs a Triton with tl.topk"
    assert logits.dim() == 2 and logits.dtype == torch.float32 and logits.is_contiguous()
    assert expert_bias.dtype == torch.float32 and expert_bias.numel() == logits.shape[1]
    num_tokens, num_experts = logits.shape
    assert 1 < topk <= num_experts
    probs = torch.empty(num_tokens, topk, dtype=torch.float32, device=logits.device)
    indices = torch.empty(num_tokens, topk, dtype=torch.int64, device=logits.device)
    if num_tokens == 0:
        return probs, indices
    experts_pow2 = triton.next_power_of_2(num_experts)
    _sigmoid_topk_kernel[(triton.cdiv(num_tokens, block_tokens),)](
        logits,
        expert_bias,
        probs,
        indices,
        num_tokens,
        float(scaling_factor) if scaling_factor else 1.0,
        NUM_EXPERTS=num_experts,
        EXPERTS_POW2=experts_pow2,
        TOPK=topk,
        TOPK_POW2=min(triton.next_power_of_2(topk), experts_pow2),
        APPLY_SCALING=bool(scaling_factor),
        BLOCK_T=block_tokens,
        num_warps=num_warps,
    )
    return probs, indices
