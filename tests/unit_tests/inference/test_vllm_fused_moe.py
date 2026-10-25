# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for megatron.core.inference.moe.vllm_fused_moe.

Tests cover:
- _get_default_config: full launch config (tile sizes, warps, stages) from M/E/top_k
- _moe_align_block_size_cuda_graphable: indirection table construction
- _moe_sum: fused topk reduction with routing weight application
- vllm_fused_moe: end-to-end correctness vs sequential reference
"""

import os
import tempfile

# Redirect Triton cache to /tmp BEFORE triton is imported (at module level) so
# compiled kernels don't accumulate in ~/.triton/ across test runs.
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(tempfile.gettempdir(), "triton_test_cache"))

import pytest
import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _one_token_per_program_moe_sum_kernel(
        input_ptr,
        output_ptr,
        topk_weights_ptr,
        valid_tokens_ptr,
        routing_map_ptr,
        local_expert_start,
        num_local_experts: tl.constexpr,
        K,
        topk: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
        APPLY_WEIGHTS: tl.constexpr,
        ACC_FP64: tl.constexpr,
    ):
        """The previous _moe_sum kernel (one token per program, branch per slot), kept as
        the bit-exact reference for the tiled kernel."""
        pid = tl.program_id(0)
        valid_tokens = tl.load(valid_tokens_ptr)
        for token_id in tl.range(pid, valid_tokens, BLOCK_M):
            token_id_i64 = token_id.to(tl.int64)
            base = token_id_i64 * topk * K
            for k_idx in range(NUM_K_BLOCKS):
                offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
                k_mask = offs_k < K
                acc = tl.zeros([BLOCK_K], dtype=tl.float64 if ACC_FP64 else tl.float32)
                for t in range(topk):
                    eid = tl.load(routing_map_ptr + token_id * topk + t)
                    lid = eid - local_expert_start
                    if lid >= 0 and lid < num_local_experts:
                        v = tl.load(input_ptr + base + t * K + offs_k, mask=k_mask, other=0.0)
                        if APPLY_WEIGHTS:
                            w = tl.load(topk_weights_ptr + token_id * topk + t)
                            if ACC_FP64:
                                acc += v.to(tl.float64) * w.to(tl.float64)
                            else:
                                acc += v.to(tl.float32) * w
                        else:
                            if ACC_FP64:
                                acc += v.to(tl.float64)
                            else:
                                acc += v.to(tl.float32)
                tl.store(
                    output_ptr + token_id_i64 * K + offs_k, acc.to(tl.float32), mask=k_mask
                )


def _one_token_per_program_moe_sum(
    input, topk_weights, K, valid, routing_map, local_start, num_local, out, apply_w, fp64
):
    """Launch the reference kernel the way the previous _moe_sum did."""
    topk = routing_map.shape[1]
    BLOCK_K = min(triton.next_power_of_2(K), 1024)
    BLOCK_M = torch.cuda.get_device_properties(input.device).multi_processor_count
    _one_token_per_program_moe_sum_kernel[(BLOCK_M,)](
        input,
        out,
        topk_weights,
        valid,
        routing_map,
        local_start,
        num_local,
        K,
        topk=topk,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        NUM_K_BLOCKS=triton.cdiv(K, BLOCK_K),
        APPLY_WEIGHTS=apply_w,
        ACC_FP64=fp64,
    )
    return out


def _vt(n):
    """Create a valid_tokens scalar int32 CUDA tensor."""
    return torch.tensor(n, dtype=torch.int32, device="cuda")


def _ref_sequential_moe(
    hidden_states,
    probs,
    fc1_weight,
    fc2_weight,
    routing_map,
    num_local_experts,
    local_expert_start,
    valid_tokens,
):
    """PyTorch reference: sequential per-token MoE computation.

    For each valid token, for each topk slot routed to a local expert:
        intermediate = squared_relu(hidden @ fc1_weight[expert].T)
        output += prob * (intermediate @ fc2_weight[expert].T)
    """
    vt = valid_tokens if isinstance(valid_tokens, int) else valid_tokens.item()
    max_tokens, topk = routing_map.shape
    hidden_size = hidden_states.shape[1]

    out = torch.zeros(max_tokens, hidden_size, device="cuda", dtype=torch.float32)

    for t in range(vt):
        acc = torch.zeros(hidden_size, device="cuda", dtype=torch.float32)
        for k in range(topk):
            eid = routing_map[t, k].item()
            lid = eid - local_expert_start
            if 0 <= lid < num_local_experts:
                h = hidden_states[t].float()
                fc1_out = h @ fc1_weight[lid].float().T
                activated = torch.clamp(fc1_out, min=0.0) ** 2
                fc2_out = activated @ fc2_weight[lid].float().T
                acc += probs[t, k].item() * fc2_out
        out[t] = acc

    return out


def _make_moe_inputs(
    max_tokens, hidden_size, ffn_hidden, topk, num_experts, valid_tokens=None, seed=42
):
    """Create random inputs for fused MoE testing."""
    torch.manual_seed(seed)
    if valid_tokens is None:
        valid_tokens = max_tokens
    hidden = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
    routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
    fc1_weight = (
        torch.randn(num_experts, ffn_hidden, hidden_size, device="cuda", dtype=torch.bfloat16)
        * 0.01
    )
    fc2_weight = (
        torch.randn(num_experts, hidden_size, ffn_hidden, device="cuda", dtype=torch.bfloat16)
        * 0.01
    )
    return hidden, probs, routing_map, fc1_weight, fc2_weight


def _make_swiglu_inputs(max_tokens, hidden_size, ffn_hidden, topk, num_experts, seed=42):
    """Create random inputs for SwiGLU fused MoE: FC1 output is 2*ffn_hidden (gate|up)."""
    torch.manual_seed(seed)
    hidden = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
    routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
    fc1_weight = (
        torch.randn(num_experts, 2 * ffn_hidden, hidden_size, device="cuda", dtype=torch.bfloat16)
        * 0.01
    )
    fc2_weight = (
        torch.randn(num_experts, hidden_size, ffn_hidden, device="cuda", dtype=torch.bfloat16)
        * 0.01
    )
    return hidden, probs, routing_map, fc1_weight, fc2_weight


def _ref_sequential_moe_swiglu(
    hidden_states,
    probs,
    fc1_weight,
    fc2_weight,
    routing_map,
    num_local_experts,
    local_expert_start,
    valid_tokens,
):
    """PyTorch reference for SwiGLU MoE: gate|up split -> SiLU(gate)*up -> FC2."""
    vt = valid_tokens if isinstance(valid_tokens, int) else valid_tokens.item()
    max_tokens, topk = routing_map.shape
    hidden_size = hidden_states.shape[1]
    n_half = fc1_weight.shape[1] // 2
    out = torch.zeros(max_tokens, hidden_size, device="cuda", dtype=torch.float32)
    for t in range(vt):
        acc = torch.zeros(hidden_size, device="cuda", dtype=torch.float32)
        for k in range(topk):
            eid = routing_map[t, k].item()
            lid = eid - local_expert_start
            if 0 <= lid < num_local_experts:
                h = hidden_states[t].float()
                fc1_out = h @ fc1_weight[lid].float().T  # [2*ffn_hidden]
                gate, up = fc1_out[:n_half], fc1_out[n_half:]
                activated = torch.nn.functional.silu(gate) * up
                fc2_out = activated @ fc2_weight[lid].float().T
                acc += probs[t, k].item() * fc2_out
        out[t] = acc
    return out


# ──────────────────────────────────────────────────────────────────────
# _get_default_config (mirrors vLLM's get_default_config)
# ──────────────────────────────────────────────────────────────────────


class TestGetDefaultConfig:

    @pytest.mark.parametrize(
        "M,expected_block_m",
        [
            (1, 16),
            (16, 16),
            (32, 16),
            (33, 32),
            (64, 32),
            (96, 32),
            (97, 64),
            (256, 64),
            (512, 64),
            (513, 128),
            (1024, 128),
            (4096, 128),
        ],
    )
    def test_block_m_thresholds(self, M, expected_block_m):
        from megatron.core.inference.moe.vllm_fused_moe import _get_default_config

        cfg = _get_default_config(M=M, E=8, top_k=2)
        assert cfg['BLOCK_SIZE_M'] == expected_block_m

    def test_block_n_and_k_split_at_64(self):
        from megatron.core.inference.moe.vllm_fused_moe import _get_default_config

        small = _get_default_config(M=64, E=8, top_k=2)
        large = _get_default_config(M=65, E=8, top_k=2)
        assert small['BLOCK_SIZE_N'] == 64 and small['BLOCK_SIZE_K'] == 128
        assert large['BLOCK_SIZE_N'] == 128 and large['BLOCK_SIZE_K'] == 64

    def test_group_size_m_uses_tokens_per_expert(self):
        from megatron.core.inference.moe.vllm_fused_moe import _get_default_config

        # 1024 / 8 = 128 → group_m = 1 (boundary not strict-greater).
        assert _get_default_config(M=1024, E=8, top_k=2)['GROUP_SIZE_M'] == 1
        # 2048 / 8 = 256 > 128 → group_m = 16.
        assert _get_default_config(M=2048, E=8, top_k=2)['GROUP_SIZE_M'] == 16
        # Many experts, few tokens-per-expert → group_m = 1.
        assert _get_default_config(M=512, E=64, top_k=2)['GROUP_SIZE_M'] == 1

    def test_num_warps_split_at_128(self):
        from megatron.core.inference.moe.vllm_fused_moe import _get_default_config

        assert _get_default_config(M=128, E=8, top_k=2)['num_warps'] == 4
        assert _get_default_config(M=129, E=8, top_k=2)['num_warps'] == 8

    def test_num_stages_split_at_32(self):
        from megatron.core.inference.moe.vllm_fused_moe import _get_default_config

        assert _get_default_config(M=32, E=8, top_k=2)['num_stages'] == 4
        assert _get_default_config(M=33, E=8, top_k=2)['num_stages'] == 3


# ──────────────────────────────────────────────────────────────────────
# _moe_align_block_size_cuda_graphable
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.internal
class TestMoeAlignBlockSize:

    @pytest.mark.parametrize(
        "max_tokens,topk,num_experts,block_size",
        [
            (4, 1, 4, 16),
            (8, 2, 4, 16),
            (16, 2, 8, 32),
            (32, 4, 8, 64),
            (64, 6, 8, 128),
            (128, 8, 16, 64),
            (1, 1, 4, 16),
        ],
    )
    def test_output_shapes(self, max_tokens, topk, num_experts, block_size):
        from megatron.core.inference.moe.vllm_fused_moe import (
            _ceil_div,
            _moe_align_block_size_cuda_graphable,
        )

        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        sorted_ids, expert_ids, num_post_padded = _moe_align_block_size_cuda_graphable(
            routing_map, block_size, num_experts, 0, _vt(max_tokens)
        )

        max_sorted = max_tokens * topk + block_size * (num_experts + 1)
        max_blocks = _ceil_div(max_sorted, block_size)

        assert sorted_ids.shape == (max_sorted,)
        assert sorted_ids.dtype == torch.int32
        assert expert_ids.shape == (max_blocks,)
        assert expert_ids.dtype == torch.int32
        assert num_post_padded.shape == (1,)
        assert num_post_padded.dtype == torch.int32

    @pytest.mark.parametrize(
        "max_tokens,topk,num_experts", [(16, 2, 4), (32, 4, 8), (64, 6, 8), (128, 8, 16)]
    )
    @pytest.mark.parametrize("block_size", [16, 32, 64, 128])
    def test_num_post_padded_is_aligned(self, max_tokens, topk, num_experts, block_size):
        from megatron.core.inference.moe.vllm_fused_moe import _moe_align_block_size_cuda_graphable

        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        _, _, num_post_padded = _moe_align_block_size_cuda_graphable(
            routing_map, block_size, num_experts, 0, _vt(max_tokens)
        )
        npp = num_post_padded.item()
        assert npp % block_size == 0, f"num_post_padded={npp} not aligned to {block_size}"

    @pytest.mark.parametrize("max_tokens,topk,num_experts", [(8, 2, 4), (16, 4, 8), (32, 6, 8)])
    def test_all_local_tokens_present(self, max_tokens, topk, num_experts, block_size=16):
        """Every (token, topk) pair routed to a local expert appears in sorted_token_ids."""
        from megatron.core.inference.moe.vllm_fused_moe import _moe_align_block_size_cuda_graphable

        torch.manual_seed(42)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        sentinel = max_tokens * topk
        sorted_ids, _, num_post_padded = _moe_align_block_size_cuda_graphable(
            routing_map, block_size, num_experts, 0, _vt(max_tokens)
        )

        valid_sorted = sorted_ids[: num_post_padded.item()]
        real_ids = valid_sorted[valid_sorted < sentinel]

        expected_pairs = set()
        for t in range(max_tokens):
            for k in range(topk):
                expected_pairs.add(t * topk + k)
        actual_pairs = set(real_ids.cpu().tolist())
        assert actual_pairs == expected_pairs

    @pytest.mark.parametrize(
        "max_tokens,topk,num_experts,local_start,num_local",
        [(32, 4, 8, 0, 4), (32, 4, 8, 4, 4), (64, 6, 16, 4, 8), (64, 6, 16, 0, 1)],
    )
    def test_expert_subset_only_local_tokens(
        self, max_tokens, topk, num_experts, local_start, num_local
    ):
        """Only tokens routed to local experts appear in sorted_token_ids."""
        from megatron.core.inference.moe.vllm_fused_moe import _moe_align_block_size_cuda_graphable

        torch.manual_seed(42)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        sentinel = max_tokens * topk
        sorted_ids, _, num_post_padded = _moe_align_block_size_cuda_graphable(
            routing_map, 16, num_local, local_start, _vt(max_tokens)
        )

        valid_sorted = sorted_ids[: num_post_padded.item()]
        real_ids = valid_sorted[valid_sorted < sentinel]

        expected_pairs = set()
        rm_flat = routing_map.cpu()
        for t in range(max_tokens):
            for k in range(topk):
                eid = rm_flat[t, k].item()
                if local_start <= eid < local_start + num_local:
                    expected_pairs.add(t * topk + k)

        actual_pairs = set(real_ids.cpu().tolist())
        assert actual_pairs == expected_pairs

    @pytest.mark.parametrize("block_size", [16, 32, 64, 128])
    def test_expert_ids_cover_all_blocks(self, block_size):
        """expert_ids has a valid expert index for every block in [0, num_post_padded/block_size)."""
        from megatron.core.inference.moe.vllm_fused_moe import _moe_align_block_size_cuda_graphable

        max_tokens, topk, num_experts = 32, 4, 8
        torch.manual_seed(42)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        _, expert_ids, num_post_padded = _moe_align_block_size_cuda_graphable(
            routing_map, block_size, num_experts, 0, _vt(max_tokens)
        )

        n_blocks = num_post_padded.item() // block_size
        active_eids = expert_ids[:n_blocks].cpu()
        assert (active_eids >= 0).all(), "Found negative expert_id in active range"
        assert (active_eids < num_experts).all(), "Found expert_id >= num_experts"


# ──────────────────────────────────────────────────────────────────────
# _moe_sum
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.internal
class TestMoeSum:

    def _ref_moe_sum(
        self,
        input,
        topk_weights,
        max_tokens,
        topk,
        K,
        routing_map,
        local_expert_start,
        num_local_experts,
    ):
        """PyTorch reference for _moe_sum: reduce topk with local-expert filtering."""
        out = torch.zeros(max_tokens, K, device="cuda", dtype=torch.float32)
        for t in range(max_tokens):
            acc = torch.zeros(K, device="cuda", dtype=torch.float32)
            for k in range(topk):
                eid = routing_map[t, k].item()
                lid = eid - local_expert_start
                if 0 <= lid < num_local_experts:
                    v = input[t * topk + k].float()
                    w = topk_weights[t, k].item()
                    acc += v * w
            out[t] = acc
        return out

    @pytest.mark.parametrize(
        "max_tokens,topk,K,num_experts",
        [(4, 1, 64, 4), (8, 2, 64, 4), (16, 4, 128, 8), (32, 6, 128, 8), (64, 8, 256, 16)],
    )
    def test_matches_reference_all_local(self, max_tokens, topk, K, num_experts):
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        torch.manual_seed(42)
        input = torch.randn(max_tokens * topk, K, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")

        result = _moe_sum(
            input, topk_weights, max_tokens, topk, K, _vt(max_tokens), routing_map, 0, num_experts
        )
        expected = self._ref_moe_sum(
            input, topk_weights, max_tokens, topk, K, routing_map, 0, num_experts
        )

        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "local_start,num_local,num_experts", [(0, 4, 8), (4, 4, 8), (0, 1, 8), (2, 3, 8)]
    )
    def test_matches_reference_expert_subset(self, local_start, num_local, num_experts):
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        max_tokens, topk, K = 32, 4, 128
        torch.manual_seed(42)
        input = torch.randn(max_tokens * topk, K, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")

        result = _moe_sum(
            input,
            topk_weights,
            max_tokens,
            topk,
            K,
            _vt(max_tokens),
            routing_map,
            local_start,
            num_local,
        )
        expected = self._ref_moe_sum(
            input, topk_weights, max_tokens, topk, K, routing_map, local_start, num_local
        )

        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_writes_to_provided_output_buffer(self):
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        max_tokens, topk, K, num_experts = 8, 2, 64, 4
        torch.manual_seed(42)
        input = torch.randn(max_tokens * topk, K, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")

        out_buf = torch.empty(max_tokens, K, dtype=torch.float32, device="cuda")
        result = _moe_sum(
            input,
            topk_weights,
            max_tokens,
            topk,
            K,
            _vt(max_tokens),
            routing_map,
            0,
            num_experts,
            out=out_buf,
        )

        assert result.data_ptr() == out_buf.data_ptr()

    @pytest.mark.parametrize("K", [32, 64, 128, 256, 512, 1024, 2688])
    def test_various_hidden_dims(self, K):
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        max_tokens, topk, num_experts = 8, 4, 4
        torch.manual_seed(42)
        input = torch.randn(max_tokens * topk, K, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")

        result = _moe_sum(
            input, topk_weights, max_tokens, topk, K, _vt(max_tokens), routing_map, 0, num_experts
        )
        expected = self._ref_moe_sum(
            input, topk_weights, max_tokens, topk, K, routing_map, 0, num_experts
        )

        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("K", [128, 2688])
    @pytest.mark.parametrize("apply_weights,acc_fp64", [(True, False), (False, True)])
    def test_skip_unrouted_rows(self, K, apply_weights, acc_fp64):
        """write_unrouted_rows=False leaves rows with no local expert untouched and writes
        every other valid row bit-identically to the default."""
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        max_tokens, topk, num_experts, local_start, num_local = 64, 3, 16, 4, 2
        valid = 56
        torch.manual_seed(42)
        input = torch.randn(max_tokens * topk, K, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        routing_map[::7] = -1  # padding tokens route nowhere
        local = (routing_map >= local_start) & (routing_map < local_start + num_local)
        routed = local.any(dim=1)
        routed[valid:] = False
        assert routed.any() and not routed[:valid].all()

        def run(write_unrouted_rows):
            out = torch.full((max_tokens, K), float("nan"), device="cuda", dtype=torch.float32)
            return _moe_sum(
                input,
                topk_weights,
                max_tokens,
                topk,
                K,
                _vt(valid),
                routing_map,
                local_start,
                num_local,
                out=out,
                apply_weights=apply_weights,
                acc_fp64=acc_fp64,
                write_unrouted_rows=write_unrouted_rows,
            )

        full, skipped = run(True), run(False)
        assert torch.equal(skipped[routed], full[routed])
        assert torch.isnan(skipped[~routed]).all()
        assert not torch.isnan(full[:valid]).any()


    @staticmethod
    def _bit_exact_inputs(max_tokens, topk, K, num_experts, padding_every=0):
        torch.manual_seed(1234)
        input = torch.randn(max_tokens * topk, K, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        # Distinct experts per token, as the router produces.
        routing_map = torch.rand(max_tokens, num_experts, device="cuda").topk(topk).indices
        if padding_every:
            routing_map[::padding_every] = -1
        return input, topk_weights, routing_map

    @pytest.mark.parametrize("K", [128, 1000, 2688])
    @pytest.mark.parametrize(
        "local_start,num_local,padding_every", [(0, 64, 0), (8, 8, 0), (24, 8, 5)]
    )
    @pytest.mark.parametrize(
        "apply_weights,acc_fp64", [(True, False), (False, False), (False, True)]
    )
    @pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
    def test_bit_exact_vs_one_token_per_program(
        self, K, local_start, num_local, padding_every, apply_weights, acc_fp64, out_dtype
    ):
        """The tiled kernel sums the same slots in the same order as the previous kernel,
        so every written element is bit-identical; rows >= valid_tokens stay untouched."""
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        max_tokens, topk, num_experts, valid = 300, 22, 64, 287
        input, topk_weights, routing_map = self._bit_exact_inputs(
            max_tokens, topk, K, num_experts, padding_every
        )

        def fresh_out():
            return torch.full((max_tokens, K), float("nan"), device="cuda", dtype=out_dtype)

        expected = _one_token_per_program_moe_sum(
            input,
            topk_weights,
            K,
            _vt(valid),
            routing_map,
            local_start,
            num_local,
            fresh_out(),
            apply_weights,
            acc_fp64,
        )
        result = _moe_sum(
            input,
            topk_weights,
            max_tokens,
            topk,
            K,
            _vt(valid),
            routing_map,
            local_start,
            num_local,
            out=fresh_out(),
            apply_weights=apply_weights,
            acc_fp64=acc_fp64,
        )
        assert torch.equal(result[:valid], expected[:valid])
        assert torch.isnan(result[valid:]).all()

    @pytest.mark.parametrize(
        "config",
        [
            {"BLOCK_T": 2, "BLOCK_K": 1024, "programs_per_sm": 1, "num_warps": 4},
            {"BLOCK_T": 8, "BLOCK_K": 128, "programs_per_sm": 16, "num_warps": 4},
            {"BLOCK_T": 32, "BLOCK_K": 512, "programs_per_sm": 2, "num_warps": 8},
            {"BLOCK_T": 8, "BLOCK_K": 256, "programs_per_sm": 8, "num_warps": 8},
        ],
    )
    def test_result_independent_of_launch_config(self, config):
        """Tile sizes and program counts change only who computes an element, not how."""
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        max_tokens, topk, K, num_experts, valid = 300, 22, 1000, 64, 291
        input, topk_weights, routing_map = self._bit_exact_inputs(
            max_tokens, topk, K, num_experts, padding_every=7
        )
        args = (input, topk_weights, max_tokens, topk, K, _vt(valid), routing_map, 8, 8)
        default = _moe_sum(*args, out=torch.zeros(max_tokens, K, device="cuda"))
        tuned = _moe_sum(*args, out=torch.zeros(max_tokens, K, device="cuda"), config=config)
        assert torch.equal(default, tuned)

    def test_zero_valid_tokens_writes_nothing(self):
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        max_tokens, topk, K, num_experts = 64, 22, 256, 64
        input, topk_weights, routing_map = self._bit_exact_inputs(max_tokens, topk, K, num_experts)
        out = torch.full((max_tokens, K), float("nan"), device="cuda")
        _moe_sum(input, topk_weights, max_tokens, topk, K, _vt(0), routing_map, 0, 8, out=out)
        assert torch.isnan(out).all()


# ──────────────────────────────────────────────────────────────────────
# vllm_fused_moe (end-to-end)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.internal
class TestVllmFusedMoe:

    @pytest.mark.parametrize(
        "max_tokens,hidden_size,ffn_hidden,topk,num_experts",
        [
            (1, 64, 64, 1, 4),
            (4, 64, 64, 2, 4),
            (8, 128, 128, 2, 8),
            (16, 128, 128, 4, 8),
            (32, 128, 128, 6, 8),
            (64, 128, 256, 4, 8),
            (128, 64, 128, 8, 16),
        ],
    )
    def test_matches_reference(self, max_tokens, hidden_size, ffn_hidden, topk, num_experts):
        """vllm_fused_moe output matches sequential per-token reference."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )

        result = vllm_fused_moe(
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(max_tokens),
            routing_map,
        )
        expected = _ref_sequential_moe(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_experts, 0, max_tokens
        )

        assert result.shape == (max_tokens, hidden_size)
        assert result.dtype == torch.float32
        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize(
        "max_tokens,hidden_size,ffn_hidden,topk,num_experts",
        [(4, 64, 64, 2, 4), (16, 128, 128, 4, 8), (32, 128, 256, 6, 8), (128, 64, 128, 8, 16)],
    )
    def test_matches_reference_swiglu(self, max_tokens, hidden_size, ffn_hidden, topk, num_experts):
        """SwiGLU (SiLU(gate)*up) output matches the sequential per-token reference."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_swiglu_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )

        result = vllm_fused_moe(
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SWIGLU,
            num_experts,
            0,
            _vt(max_tokens),
            routing_map,
        )
        expected = _ref_sequential_moe_swiglu(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_experts, 0, max_tokens
        )

        assert result.shape == (max_tokens, hidden_size)
        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize(
        "local_start,num_local,num_experts", [(0, 4, 8), (4, 4, 8), (0, 2, 8), (6, 2, 8)]
    )
    def test_expert_subset(self, local_start, num_local, num_experts):
        """Correct output when only a subset of experts are local."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk = 32, 128, 128, 4
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )
        fc1_local = fc1_weight[local_start : local_start + num_local].contiguous()
        fc2_local = fc2_weight[local_start : local_start + num_local].contiguous()

        result = vllm_fused_moe(
            hidden,
            probs,
            fc1_local,
            fc2_local,
            ActivationType.SQUARED_RELU,
            num_local,
            local_start,
            _vt(max_tokens),
            routing_map,
        )
        expected = _ref_sequential_moe(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_local, local_start, max_tokens
        )

        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("valid_tokens", [1, 4, 8, 15])
    def test_partial_valid_tokens(self, valid_tokens):
        """Only the first valid_tokens rows have meaningful output."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 16, 128, 128, 4, 8
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )

        result = vllm_fused_moe(
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(valid_tokens),
            routing_map,
        )
        expected = _ref_sequential_moe(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_experts, 0, valid_tokens
        )

        torch.testing.assert_close(
            result[:valid_tokens], expected[:valid_tokens], atol=5e-2, rtol=5e-2
        )

    def test_output_buffer_reuse(self):
        """Output is written to provided buffer when out= is specified."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 8, 128, 128, 2, 4
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )

        out_buf = torch.empty(max_tokens, hidden_size, dtype=torch.float32, device="cuda")
        result = vllm_fused_moe(
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(max_tokens),
            routing_map,
            out=out_buf,
        )

        assert result.data_ptr() == out_buf.data_ptr()

    def test_rejects_non_bf16_input(self):
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 4, 64, 64, 2, 4
        _, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )
        hidden_fp32 = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.float32)

        with pytest.raises(AssertionError, match="bf16"):
            vllm_fused_moe(
                hidden_fp32,
                probs,
                fc1_weight,
                fc2_weight,
                ActivationType.SQUARED_RELU,
                num_experts,
                0,
                _vt(max_tokens),
                routing_map,
            )

    @pytest.mark.parametrize("seed", [0, 7, 42, 123, 999])
    def test_deterministic_across_seeds(self, seed):
        """Same inputs produce the same output regardless of seed."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 16, 128, 128, 4, 8
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts, seed=seed
        )

        result = vllm_fused_moe(
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(max_tokens),
            routing_map,
        )
        expected = _ref_sequential_moe(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_experts, 0, max_tokens
        )

        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)

    def test_num_tokens_hint(self):
        """num_tokens_hint selects a potentially different block size but result is still correct."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 16, 128, 128, 4, 8
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )

        result = vllm_fused_moe(
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(max_tokens),
            routing_map,
            num_tokens_hint=4,
        )
        expected = _ref_sequential_moe(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_experts, 0, max_tokens
        )

        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)


# ──────────────────────────────────────────────────────────────────────
# CUDA graph capture + replay
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.internal
class TestVllmFusedMoeCudaGraph:

    @pytest.mark.parametrize(
        "max_tokens,hidden_size,ffn_hidden,topk,num_experts",
        [(8, 128, 128, 2, 4), (16, 128, 128, 4, 8), (32, 128, 128, 6, 8), (64, 128, 256, 4, 8)],
    )
    def test_capture_and_replay(self, max_tokens, hidden_size, ffn_hidden, topk, num_experts):
        """vllm_fused_moe can be captured in a CUDA graph and replayed correctly."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        torch.manual_seed(42)
        static_hidden = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        static_probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        static_routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        static_vt = _vt(max_tokens)
        fc1 = (
            torch.randn(num_experts, ffn_hidden, hidden_size, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        fc2 = (
            torch.randn(num_experts, hidden_size, ffn_hidden, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        static_out = torch.empty(max_tokens, hidden_size, dtype=torch.float32, device="cuda")

        # Warmup to trigger Triton autotuning
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(3):
                vllm_fused_moe(
                    static_hidden,
                    static_probs,
                    fc1,
                    fc2,
                    ActivationType.SQUARED_RELU,
                    num_experts,
                    0,
                    static_vt,
                    static_routing_map,
                    out=static_out,
                )
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            vllm_fused_moe(
                static_hidden,
                static_probs,
                fc1,
                fc2,
                ActivationType.SQUARED_RELU,
                num_experts,
                0,
                static_vt,
                static_routing_map,
                out=static_out,
            )

        graph.replay()

        expected = _ref_sequential_moe(
            static_hidden, static_probs, fc1, fc2, static_routing_map, num_experts, 0, max_tokens
        )
        torch.testing.assert_close(static_out, expected, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize(
        "max_tokens,valid_tokens_list",
        [(16, [16, 8, 1, 4, 16]), (32, [32, 1, 16, 8, 32]), (64, [64, 32, 4, 1, 48])],
    )
    def test_replay_with_varying_valid_tokens(self, max_tokens, valid_tokens_list):
        """Replaying with different valid_tokens produces correct results each time.

        This is the core decode use case: the buffer is max-sized but only a
        varying prefix is valid each iteration. The graph is captured once and
        replayed with updated valid_tokens.
        """
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        hidden_size, ffn_hidden, topk, num_experts = 128, 128, 4, 8

        torch.manual_seed(42)
        static_hidden = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        static_probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        static_routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        static_vt = _vt(max_tokens)
        fc1 = (
            torch.randn(num_experts, ffn_hidden, hidden_size, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        fc2 = (
            torch.randn(num_experts, hidden_size, ffn_hidden, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        static_out = torch.empty(max_tokens, hidden_size, dtype=torch.float32, device="cuda")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(3):
                vllm_fused_moe(
                    static_hidden,
                    static_probs,
                    fc1,
                    fc2,
                    ActivationType.SQUARED_RELU,
                    num_experts,
                    0,
                    static_vt,
                    static_routing_map,
                    out=static_out,
                )
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            vllm_fused_moe(
                static_hidden,
                static_probs,
                fc1,
                fc2,
                ActivationType.SQUARED_RELU,
                num_experts,
                0,
                static_vt,
                static_routing_map,
                out=static_out,
            )

        for vt in valid_tokens_list:
            static_vt.fill_(vt)
            graph.replay()

            expected = _ref_sequential_moe(
                static_hidden, static_probs, fc1, fc2, static_routing_map, num_experts, 0, vt
            )
            torch.testing.assert_close(
                static_out[:vt],
                expected[:vt],
                atol=5e-2,
                rtol=5e-2,
                msg=f"Mismatch with valid_tokens={vt}",
            )

    def test_replay_with_new_inputs(self):
        """Replaying after mutating hidden/probs/routing produces correct results.

        After graph capture, we overwrite the static input buffers with entirely
        new data and replay. The graph re-reads from the same device pointers,
        so it picks up the new values.
        """
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 16, 128, 128, 4, 8

        torch.manual_seed(42)
        static_hidden = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        static_probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        static_routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        static_vt = _vt(max_tokens)
        fc1 = (
            torch.randn(num_experts, ffn_hidden, hidden_size, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        fc2 = (
            torch.randn(num_experts, hidden_size, ffn_hidden, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        static_out = torch.empty(max_tokens, hidden_size, dtype=torch.float32, device="cuda")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(3):
                vllm_fused_moe(
                    static_hidden,
                    static_probs,
                    fc1,
                    fc2,
                    ActivationType.SQUARED_RELU,
                    num_experts,
                    0,
                    static_vt,
                    static_routing_map,
                    out=static_out,
                )
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            vllm_fused_moe(
                static_hidden,
                static_probs,
                fc1,
                fc2,
                ActivationType.SQUARED_RELU,
                num_experts,
                0,
                static_vt,
                static_routing_map,
                out=static_out,
            )

        for seed in [7, 123, 999]:
            torch.manual_seed(seed)
            static_hidden.copy_(
                torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
            )
            static_probs.copy_(torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32))
            static_routing_map.copy_(
                torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
            )

            graph.replay()

            expected = _ref_sequential_moe(
                static_hidden,
                static_probs,
                fc1,
                fc2,
                static_routing_map,
                num_experts,
                0,
                max_tokens,
            )
            torch.testing.assert_close(
                static_out,
                expected,
                atol=5e-2,
                rtol=5e-2,
                msg=f"Mismatch after replaying with seed={seed}",
            )

    def test_replay_matches_eager(self):
        """Graph replay produces the same output as eager execution on identical inputs."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 32, 128, 128, 4, 8

        torch.manual_seed(42)
        static_hidden = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        static_probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        static_routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        static_vt = _vt(max_tokens)
        fc1 = (
            torch.randn(num_experts, ffn_hidden, hidden_size, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        fc2 = (
            torch.randn(num_experts, hidden_size, ffn_hidden, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        static_out = torch.empty(max_tokens, hidden_size, dtype=torch.float32, device="cuda")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(3):
                vllm_fused_moe(
                    static_hidden,
                    static_probs,
                    fc1,
                    fc2,
                    ActivationType.SQUARED_RELU,
                    num_experts,
                    0,
                    static_vt,
                    static_routing_map,
                    out=static_out,
                )
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            vllm_fused_moe(
                static_hidden,
                static_probs,
                fc1,
                fc2,
                ActivationType.SQUARED_RELU,
                num_experts,
                0,
                static_vt,
                static_routing_map,
                out=static_out,
            )

        for seed in [0, 7, 42]:
            torch.manual_seed(seed)
            new_hidden = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
            new_probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
            new_routing = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
            new_vt = torch.randint(1, max_tokens + 1, (1,)).item()

            static_hidden.copy_(new_hidden)
            static_probs.copy_(new_probs)
            static_routing_map.copy_(new_routing)
            static_vt.fill_(new_vt)

            graph.replay()
            graph_result = static_out[:new_vt].clone()

            eager_out = torch.empty(max_tokens, hidden_size, dtype=torch.float32, device="cuda")
            vllm_fused_moe(
                new_hidden,
                new_probs,
                fc1,
                fc2,
                ActivationType.SQUARED_RELU,
                num_experts,
                0,
                _vt(new_vt),
                new_routing,
                out=eager_out,
            )
            eager_result = eager_out[:new_vt]

            torch.testing.assert_close(
                graph_result,
                eager_result,
                atol=0,
                rtol=0,
                msg=f"Graph/eager mismatch at seed={seed}, valid_tokens={new_vt}",
            )


# ──────────────────────────────────────────────────────────────────────
# Cross-backend: vllm vs mcore_fused_moe (BF16)
# ──────────────────────────────────────────────────────────────────────


requires_torch_grouped_mm = pytest.mark.skipif(
    not hasattr(torch.nn.functional, "grouped_mm"),
    reason="Requires torch.nn.functional.grouped_mm (PyTorch 2.10+)",
)


@pytest.mark.internal
@requires_torch_grouped_mm
class TestVllmVsMcoreFusedMoe:

    @pytest.mark.parametrize(
        "max_tokens,hidden_size,ffn_hidden,topk,num_experts",
        [(4, 64, 64, 2, 4), (16, 128, 128, 4, 8), (32, 128, 128, 6, 8), (64, 128, 256, 4, 8)],
    )
    def test_vllm_matches_mcore(self, max_tokens, hidden_size, ffn_hidden, topk, num_experts):
        """vllm_fused_moe and mcore_fused_moe produce equivalent results on BF16."""
        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )
        vt = _vt(max_tokens)

        vllm_out = vllm_fused_moe(
            hidden.clone(),
            probs.clone(),
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            vt,
            routing_map.clone(),
        )
        mcore_out = mcore_fused_moe(
            hidden.clone(),
            probs.clone(),
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            vt,
            routing_map.clone(),
        )

        torch.testing.assert_close(vllm_out, mcore_out, atol=5e-2, rtol=5e-2)


CLAMP_SCALE = 16.0


def _make_saturating_moe_inputs(max_tokens, hidden_size, ffn_hidden, topk, num_experts, seed=42):
    """Like _make_moe_inputs, but with FC1 weights large enough to saturate the tanh clamp.

    The default 0.01 weight scale keeps the FC1 pre-activation well inside the clamp's
    linear region, where a dropped clamp would still pass.
    """
    hidden, probs, routing_map, _, fc2_weight = _make_moe_inputs(
        max_tokens, hidden_size, ffn_hidden, topk, num_experts, seed=seed
    )
    fc1_weight = (
        torch.randn(num_experts, ffn_hidden, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.5
    )
    return hidden, probs, routing_map, fc1_weight, fc2_weight


def _ref_sequential_moe_clamped(
    hidden_states, probs, fc1_weight, fc2_weight, routing_map, num_local_experts, clamp_scale
):
    """Sequential per-token reference with the tanh soft-clamped squared-ReLU activation.

    Mirrors _ref_sequential_moe with ``squared_relu(s * tanh(x / s))`` in place of
    ``squared_relu(x)``, but also models the two BF16 roundings the kernel performs inside
    the activation. Unlike the unclamped reference — whose 0.01-scale weights keep every
    intermediate small enough for atol to swallow the rounding — the clamp only bites when
    the pre-activation is large, and at that scale the roundings move the result by more
    than the tolerance:

      1. the clamped pre-activation stays in FP32 through the square, matching training's
         fused ``weighted_clamped_squared_relu``;
      2. the squared activation is rounded to BF16 on its way into the FC1 output buffer,
         which is what FC2 actually reads.

    The FC2 output is likewise BF16 before ``_moe_sum`` applies the FP32 routing
    probability, so that rounding is modelled too.
    """
    max_tokens, topk = routing_map.shape
    hidden_size = hidden_states.shape[1]
    out = torch.zeros(max_tokens, hidden_size, device="cuda", dtype=torch.float32)
    for t in range(max_tokens):
        for k in range(topk):
            lid = routing_map[t, k].item()
            if not 0 <= lid < num_local_experts:
                continue
            fc1_out = hidden_states[t].float() @ fc1_weight[lid].float().T
            clamped = clamp_scale * torch.tanh(fc1_out / clamp_scale)
            activated = (torch.clamp(clamped, min=0.0) ** 2).to(torch.bfloat16)
            fc2_out = (activated.float() @ fc2_weight[lid].float().T).to(torch.bfloat16)
            out[t] += probs[t, k].item() * fc2_out.float()
    return out


def _assert_close_to_bf16_ulp(actual, expected):
    """Compare two MoE outputs to the precision a BF16 MoE can carry.

    Both the kernel and the sequential reference round the FC1 and FC2 outputs to BF16, but
    they reach those roundings through different fp32 accumulation orders: grouped_mm tiles
    the K loop while the reference does a per-token matmul. A value sitting within one fp32
    ULP of a BF16 rounding boundary rounds one way in the kernel and the other way in the
    reference, and that ``2 ** -8`` relative step then propagates through FC2. Exact
    agreement is therefore unattainable here, unlike in the activation-only tests where no
    GEMM is re-derived; the achievable bound is one BF16 ULP.

    Scaling ``atol`` by the reference's own magnitude keeps that bound meaningful across
    input regimes: saturating weights produce outputs orders of magnitude larger than
    small-weight inputs do, and a fixed absolute tolerance is either vacuous or
    accidentally tight depending on which regime it was chosen for. A dropped or
    mis-scaled activation clamp moves the result by orders of magnitude, well outside
    this bound.
    """
    bf16_ulp = 2.0**-8
    torch.testing.assert_close(
        actual, expected, atol=bf16_ulp * expected.abs().max().item(), rtol=bf16_ulp
    )


@pytest.mark.internal
class TestFusedMoeActivationClamp:
    """config.activation_func_tanh_clamp_scale fused into the squared-ReLU activation.

    The clamp preconditions the FC1 output with ``s * tanh(x / s)`` before the square,
    bounding the activation by ``s ** 2``. It rides the vLLM grouped-GEMM epilogue and the
    torch-backend activation kernels; these pin all of them to one PyTorch reference.
    """

    @pytest.mark.parametrize(
        "max_tokens,hidden_size,ffn_hidden,topk,num_experts",
        [(4, 64, 64, 2, 4), (16, 128, 128, 4, 8), (64, 128, 256, 4, 8)],
    )
    def test_vllm_matches_reference(self, max_tokens, hidden_size, ffn_hidden, topk, num_experts):
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_saturating_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )

        result = vllm_fused_moe(
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(max_tokens),
            routing_map,
            activation_clamp_scale=CLAMP_SCALE,
        )
        expected = _ref_sequential_moe_clamped(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_experts, CLAMP_SCALE
        )

        assert result.shape == (max_tokens, hidden_size)
        _assert_close_to_bf16_ulp(result, expected)

    def test_clamp_changes_the_result(self):
        """Guard against the clamp being silently dropped: it must alter the output."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 16, 128, 128, 4, 8
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_saturating_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )
        args = (
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(max_tokens),
            routing_map,
        )
        unclamped = vllm_fused_moe(*args).clone()
        clamped = vllm_fused_moe(*args, activation_clamp_scale=CLAMP_SCALE).clone()
        assert not torch.allclose(unclamped, clamped, atol=1e-2, rtol=1e-2)

    def test_swiglu_with_clamp_is_rejected(self):
        """The gated (SiTU-GLU) form of the clamp has no inference kernel yet."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 4, 64, 64, 2, 4
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_swiglu_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )
        with pytest.raises(AssertionError, match="only implemented for squared ReLU"):
            vllm_fused_moe(
                hidden,
                probs,
                fc1_weight,
                fc2_weight,
                ActivationType.SWIGLU,
                num_experts,
                0,
                _vt(max_tokens),
                routing_map,
                activation_clamp_scale=CLAMP_SCALE,
            )

    @requires_torch_grouped_mm
    def test_mcore_matches_reference(self):
        """The torch grouped_mm backend applies the same clamp as the vLLM backend."""
        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 16, 128, 128, 4, 8
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_saturating_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )

        result = mcore_fused_moe(
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(max_tokens),
            routing_map,
            activation_clamp_scale=CLAMP_SCALE,
        )
        expected = _ref_sequential_moe_clamped(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_experts, CLAMP_SCALE
        )
        _assert_close_to_bf16_ulp(result.float(), expected)

    def test_mxfp8_fused_quant_kernel_receives_clamp_scale(self):
        """The MXFP8 route binds the clamp onto the fused activation+quantize kernel.

        The kernel's numerics are covered in test_mxfp8_utils.py; this pins the dispatch,
        which is the only place the MXFP8 path could silently drop the clamp.
        """
        from functools import partial

        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8
        from megatron.core.inference.moe.fused_moe import ActivationType, _get_activation_func

        bound = _get_activation_func(
            ActivationType.SQUARED_RELU,
            fused_quant=True,
            activation_kwargs={"clamp_scale": CLAMP_SCALE},
        )
        assert isinstance(bound, partial)
        assert bound.func is squared_relu_and_quantize_mxfp8
        assert bound.keywords == {"clamp_scale": CLAMP_SCALE}

        # An unset clamp must leave the kernel unwrapped so the default path is unchanged.
        assert (
            _get_activation_func(
                ActivationType.SQUARED_RELU,
                fused_quant=True,
                activation_kwargs={"clamp_scale": None},
            )
            is squared_relu_and_quantize_mxfp8
        )
