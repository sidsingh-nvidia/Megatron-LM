# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FlashInfer TRT-LLM BF16 routed MoE for inference.

Adapted from nvcsathe/Megatron-LM@trtllm-fused-moe: BlockMajorK expert weights, packed
(expert id << 16 | bf16 prob) routing, and a padding-aware finalize into the NVLS combine
buffer. Here the finalize loops only over the valid token prefix and also writes a BF16
combine buffer.

The kernel's cost scales with the rows it is handed even when they route to no expert, so
callers pass only a host-known bound on the gathered rows (see the NVLS dispatcher's
_host_valid_tokens_estimate), not the worst-case dispatch buffer.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

from megatron.core.inference.moe.permute import _get_num_sms
from megatron.core.utils import null_decorator

try:
    from flashinfer.fused_moe import WeightLayout, trtllm_bf16_routed_moe
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    HAVE_FLASHINFER_TRTLLM_BF16 = True
    _IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    HAVE_FLASHINFER_TRTLLM_BF16 = False
    _IMPORT_ERROR = exc

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()

_TILE = 128


def require_flashinfer_trtllm_bf16() -> None:
    """Raise when FlashInfer's TRT-LLM BF16 routed MoE is unavailable."""
    if not HAVE_FLASHINFER_TRTLLM_BF16:
        raise RuntimeError(
            "inference_grouped_gemm_backend='trtllm_bf16_routed' requires a FlashInfer with "
            "flashinfer.fused_moe.trtllm_bf16_routed_moe and the BlockMajorK weight helpers."
        ) from _IMPORT_ERROR


def _pad_expert_weights(fc1, fc2, gated):
    """Zero-pad one expert's intermediate dimension to a multiple of 128."""
    intermediate_size = fc2.shape[1]
    padded = (intermediate_size + _TILE - 1) // _TILE * _TILE
    padding = padded - intermediate_size
    if padding == 0:
        return fc1, fc2, padded
    if gated:
        hidden_size = fc1.shape[1]
        fc1 = F.pad(fc1.reshape(2, intermediate_size, hidden_size), (0, 0, 0, padding))
        fc1 = fc1.reshape(2 * padded, hidden_size)
    else:
        fc1 = F.pad(fc1, (0, 0, 0, padding))
    return fc1, F.pad(fc2, (0, padding)), padded


def _convert_expert(w1, w2, gated, cache):
    """BlockMajorK uint8 copies of one expert's (already padded) fc1 / fc2."""
    w1 = w1.contiguous().view(torch.uint8)
    w2 = w2.contiguous().view(torch.uint8)
    idx1 = _maybe_get_cached_w3_w1_permute_indices(cache, w1, _TILE, is_gated_act_gemm=gated)
    idx2 = get_w2_permute_indices_with_cache(cache, w2, _TILE)
    return (
        convert_to_block_layout(w1[idx1].contiguous(), _TILE),
        convert_to_block_layout(w2[idx2].contiguous(), _TILE),
    )


@torch.no_grad()
def build_trtllm_bf16_weights(
    fc1: torch.Tensor, fc2: torch.Tensor, gated: bool
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Convert [E, fc1_out, H] / [E, H, I] BF16 expert weights to TRT-LLM BlockMajorK.

    Each expert is converted into a preallocated output, so the peak is one extra copy of the
    layer's experts (not two, as with stacking a list). Returns (fc1, fc2, padded intermediate).
    """
    require_flashinfer_trtllm_bf16()
    assert fc1.dtype == torch.bfloat16 and fc2.dtype == torch.bfloat16
    assert fc1.shape[-1] % _TILE == 0, f"hidden size {fc1.shape[-1]} must be a multiple of 128"
    cache: dict = {}
    out1 = out2 = None
    intermediate_size = None
    for expert in range(fc1.shape[0]):
        w1, w2, intermediate_size = _pad_expert_weights(fc1[expert], fc2[expert], gated)
        b1, b2 = _convert_expert(w1, w2, gated, cache)
        if out1 is None:
            out1 = torch.empty((fc1.shape[0], *b1.shape), dtype=torch.uint8, device=fc1.device)
            out2 = torch.empty((fc2.shape[0], *b2.shape), dtype=torch.uint8, device=fc2.device)
        out1[expert].copy_(b1)
        out2[expert].copy_(b2)
    return out1.view(torch.bfloat16), out2.view(torch.bfloat16), intermediate_size


# Launch config of _finalize_kernel, tuned on GB200 for Ultra (top-22 of 512 experts, hidden
# 2048) at EP=64. Each program reduces a [BLOCK_T tokens, BLOCK_H columns] tile. The kernel is
# latency-bound (a token's top-k slots are mostly not local, so it moves little data), so many
# small programs per SM pay most: a one-token tile skips exactly that token's non-local slots,
# and a full-width tile reads the token's routing once rather than once per column block.
_FINALIZE_CONFIG = {"BLOCK_T": 1, "BLOCK_H": 2048, "programs_per_sm": 64, "num_warps": 2}


@triton.jit
def _finalize_kernel(
    gemm2_ptr,
    weights_ptr,
    inverse_map_ptr,
    output_ptr,
    valid_tokens_ptr,
    hidden_size,
    gemm2_rows,
    TOP_K: tl.constexpr,
    TOP_K_POW2: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    WRITE_UNROUTED: tl.constexpr,
):
    """Sum each valid token's local expert outputs, weighted by its routing probs.

    Grid: (persistent programs over BLOCK_T-token tiles, BLOCK_H column blocks). A tile first
    reads its whole [BLOCK_T, TOP_K] inverse-map block and skips, with a branch uniform across
    the tile, the slots none of its tokens has on this rank. Within a used slot, tokens whose
    slot is not local are masked rather than branched on. Every element sums its local slots in
    slot order with fused multiply-adds, whatever the launch config.

    A token with no local expert gets an explicit zero row, since the combine buffer persists
    across steps; with WRITE_UNROUTED=False its row is not written, for consumers that never
    read it (the all-to-all-v combine). The grid is fixed for CUDA graph replay; the device
    scalar bounds the live prefix.
    """
    offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    offsets_mask = offsets < hidden_size
    valid_tokens = tl.load(valid_tokens_ptr)
    slots = tl.arange(0, TOP_K_POW2)

    for tile in tl.range(tl.program_id(0), tl.cdiv(valid_tokens, BLOCK_T), tl.num_programs(0)):
        tokens = (tile * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
        token_mask = tokens < valid_tokens
        # The tile's [BLOCK_T, TOP_K] inverse-map block in one load.
        mapped_rows = tl.load(
            inverse_map_ptr + tokens[:, None] * TOP_K + slots[None, :],
            mask=token_mask[:, None] & (slots[None, :] < TOP_K),
            other=-1,
        )
        local = (mapped_rows >= 0) & (mapped_rows < gemm2_rows)
        # Bit s set: some token of the tile has slot s on this rank.
        slot_used = tl.max(local.to(tl.int64), axis=0)
        used_bits = tl.sum(slot_used << slots.to(tl.int64), axis=0)
        accumulator = tl.zeros([BLOCK_T, BLOCK_H], dtype=tl.float32)
        for slot in tl.static_range(TOP_K):
            # Skipping a slot adds nothing, exactly like loading it fully masked.
            if ((used_bits >> slot) & 1) != 0:
                mapped_row = tl.load(
                    inverse_map_ptr + tokens * TOP_K + slot, mask=token_mask, other=-1
                )
                is_local = (mapped_row >= 0) & (mapped_row < gemm2_rows)
                weight = tl.load(weights_ptr + tokens * TOP_K + slot, mask=is_local, other=0.0)
                # Masked lanes load 0 and add nothing; they cost no memory traffic.
                value = tl.load(
                    gemm2_ptr
                    + tl.maximum(mapped_row, 0).to(tl.int64)[:, None] * hidden_size
                    + offsets[None, :],
                    mask=is_local[:, None] & offsets_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                # Explicit fused multiply-add: left to the compiler, `acc += value * weight` is
                # fused or not depending on the output dtype and tile shape.
                accumulator = tl.fma(value, weight.to(tl.float32)[:, None], accumulator)

        if WRITE_UNROUTED:
            row_mask = token_mask
        elif BLOCK_T == 1:
            # The token has a local expert iff it uses a slot: a scalar, which the store
            # broadcasts without converting the routing block's layout.
            row_mask = token_mask & (used_bits != 0)
        else:
            row_mask = tl.max(local.to(tl.int32), axis=1) > 0
        tl.store(
            output_ptr + tokens[:, None] * hidden_size + offsets[None, :],
            accumulator.to(output_ptr.dtype.element_ty),
            mask=row_mask[:, None] & offsets_mask[None, :],
        )


def trtllm_finalize(
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    inverse_map: torch.Tensor,
    output: torch.Tensor,
    valid_tokens: torch.Tensor,
    top_k: int,
    write_unrouted_rows: bool = True,
    config: dict | None = None,
) -> torch.Tensor:
    """Finalize trtllm_bf16_routed_moe(do_finalize=False) into an FP32 or BF16 buffer.

    Args:
        gemm2_output: expert-grouped, tile-padded GEMM2 output [permuted rows, hidden].
        expert_weights: routing weights, rows * top_k elements.
        inverse_map: [token, top_k] -> GEMM2 row; negative for non-local or padding pairs.
        output: [rows, hidden] destination, normally the NVLS combine (RSV) buffer.
        valid_tokens: device int32[1], the live token prefix; rows past it are not written.
        top_k: routed experts per token.
        write_unrouted_rows: write a zero row for valid tokens with no local expert (default).
            Pass False when the consumer never reads those rows (the all-to-all-v combine):
            they are then left untouched.
        config: overrides of _FINALIZE_CONFIG (for tuning; the result does not depend on it).
    """
    assert HAVE_TRITON, "Triton is required for the TRT-LLM MoE finalize."
    assert gemm2_output.is_contiguous() and output.is_contiguous()
    assert output.dtype in (torch.float32, torch.bfloat16)
    assert valid_tokens.dtype == torch.int32 and valid_tokens.numel() == 1
    assert 0 < top_k <= 64, "the per-tile slot mask is an int64"
    rows, hidden_size = output.shape
    assert gemm2_output.shape[1] == hidden_size
    assert expert_weights.numel() >= rows * top_k and inverse_map.numel() >= rows * top_k
    if rows == 0:
        return output
    cfg = {**_FINALIZE_CONFIG, **(config or {})}
    block_h = min(triton.next_power_of_2(hidden_size), cfg["BLOCK_H"])
    num_h_blocks = triton.cdiv(hidden_size, block_h)
    # Static grid (CUDA graphs): enough programs to fill every SM programs_per_sm times, but no
    # more token programs than rows has tiles.
    num_token_programs = min(
        triton.cdiv(rows, cfg["BLOCK_T"]),
        max(1, cfg["programs_per_sm"] * _get_num_sms(output.device) // num_h_blocks),
    )
    _finalize_kernel[(num_token_programs, num_h_blocks)](
        gemm2_output,
        expert_weights,
        inverse_map,
        output,
        valid_tokens,
        hidden_size,
        gemm2_output.shape[0],
        TOP_K=top_k,
        TOP_K_POW2=triton.next_power_of_2(top_k),
        BLOCK_T=cfg["BLOCK_T"],
        BLOCK_H=block_h,
        WRITE_UNROUTED=write_unrouted_rows,
        num_warps=cfg["num_warps"],
    )
    return output


@triton.jit
def _pack_routing_kernel(
    ids_ptr,
    probs_ptr,
    out_ptr,
    real_count_ptr,
    numel,
    row_offset,
    TOP_K: tl.constexpr,
    HAS_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """out = (expert id << 16) | bf16 bits of prob, for one rank's local [rows, TOP_K] routing.

    Rows whose global index (local row + row_offset) is at or past the real token count are
    CUDA-graph padding: they are not loaded and are written as expert -1, probability 0.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    if HAS_COUNT:
        real_count = tl.load(real_count_ptr)
        mask_live = mask & (offs // TOP_K + row_offset < real_count)
    else:
        mask_live = mask
    ids = tl.load(ids_ptr + offs, mask=mask_live, other=-1).to(tl.int32)
    probs = tl.load(probs_ptr + offs, mask=mask_live, other=0.0).to(tl.bfloat16)
    prob_bits = probs.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    tl.store(out_ptr + offs, (ids << 16) | prob_bits, mask=mask)


def pack_routing(
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    real_token_count: torch.Tensor | None = None,
    row_offset: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack [rows, top_k] expert ids and probs into TRT-LLM's int32 (id << 16 | bf16 prob).

    One kernel in place of the torch cast / shift / bitcast / mask / or chain. Runs on each
    rank's local routing before the EP dispatch, so only the packed int32 is gathered. With
    real_token_count (device int32[1], in the global pre-SP frame) and row_offset (this rank's
    SP offset), padding rows are written as expert -1 -- the job mask_routing_padding does for
    the unpacked routing map.
    """
    assert routing_map.shape == probs.shape and routing_map.is_contiguous() and probs.is_contiguous()
    rows, top_k = routing_map.shape
    if out is None:
        out = torch.empty((rows, top_k), dtype=torch.int32, device=routing_map.device)
    numel = rows * top_k
    if numel == 0:
        return out
    block = 1024
    _pack_routing_kernel[(triton.cdiv(numel, block),)](
        routing_map,
        probs,
        out,
        real_token_count if real_token_count is not None else out,
        numel,
        row_offset,
        TOP_K=top_k,
        HAS_COUNT=real_token_count is not None,
        BLOCK=block,
    )
    return out


def trtllm_bf16_routed_moe_forward(
    hidden_states: torch.Tensor,
    packed_routing: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    *,
    num_experts: int,
    local_expert_offset: int,
    intermediate_size: int,
    activation_type: int,
    rows: int,
    out: torch.Tensor | None = None,
    valid_tokens: torch.Tensor | None = None,
    write_unrouted_rows: bool = True,
) -> torch.Tensor:
    """Run TRT-LLM BF16 routed MoE on the first `rows` rows of packed routing (pack_routing).

    Every row in [0, rows) must carry this step's routing or expert -1. With `out` (the NVLS combine
    buffer) the kernel skips its own finalize and trtllm_finalize writes the valid prefix of
    `out`, which is returned. Without `out`, returns the kernel's finalized [rows, hidden].
    write_unrouted_rows=False (with `out` only) leaves the rows of tokens with no local expert
    untouched, for consumers that never read them (the all-to-all-v combine).
    """
    require_flashinfer_trtllm_bf16()
    assert hidden_states.dtype == torch.bfloat16, "trtllm_bf16_routed requires BF16 input"
    assert packed_routing.dtype == torch.int32, "routing must be packed (pack_routing)"
    assert 0 < rows <= hidden_states.shape[0], f"rows={rows}, buffer {hidden_states.shape[0]}"
    top_k = packed_routing.shape[1]
    result = trtllm_bf16_routed_moe(
        topk_ids=packed_routing[:rows],
        hidden_states=hidden_states[:rows],
        gemm1_weights=fc1_weight,
        gemm2_weights=fc2_weight,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=fc1_weight.shape[0],
        routed_scaling_factor=None,
        # Packed precomputed routing: weights are used as-is, no softmax/renormalization.
        routing_method_type=0,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.BlockMajorK,
        do_finalize=out is None,
        tune_max_num_tokens=rows,
        activation_type=activation_type,
    )
    if out is None:
        return result[0] if isinstance(result, (list, tuple)) else result
    assert valid_tokens is not None, "valid_tokens is required when writing to `out`"
    trtllm_finalize(
        gemm2_output=result[0],
        expert_weights=result[1],
        inverse_map=result[2],
        output=out[:rows],
        valid_tokens=valid_tokens,
        top_k=top_k,
        write_unrouted_rows=write_unrouted_rows,
    )
    return out
