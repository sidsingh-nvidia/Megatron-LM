# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Config validation for inference_moe_combine_precision and the 'nvl_a2a' dispatcher."""

import pytest

from megatron.core.activations import squared_relu
from megatron.core.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_ffn_hidden_size=128,
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        inference_grouped_gemm_backend="vllm",
        inference_moe_token_dispatcher_type="nvls",
        activation_func=squared_relu,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_combine_precision_defaults_to_fp32():
    assert _make_config().inference_moe_combine_precision == "fp32"


@pytest.mark.parametrize("dispatcher", ["nvls", "nvl_a2a"])
@pytest.mark.parametrize("backend", ["vllm", "flashinfer"])
def test_bf16_combine_accepted(dispatcher, backend):
    config = _make_config(
        inference_moe_token_dispatcher_type=dispatcher,
        inference_grouped_gemm_backend=backend,
        inference_moe_combine_precision="bf16",
    )
    assert config.inference_moe_combine_precision == "bf16"


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"inference_moe_token_dispatcher_type": "nvl_a2a"}, "requires.*'bf16'"),
        (
            {
                "inference_moe_token_dispatcher_type": "nccl",
                "inference_moe_combine_precision": "bf16",
            },
            "only to the 'nvls' and 'nvl_a2a'",
        ),
        (
            {"inference_grouped_gemm_backend": "torch", "inference_moe_combine_precision": "bf16"},
            "not supported by the torch",
        ),
        (
            {
                "inference_moe_combine_precision": "bf16",
                "fp8": "hybrid",
                "fp8_recipe": "mxfp8",
                "fp8_param": True,
            },
            "MXFP8 experts on the vLLM backend",
        ),
    ],
)
def test_invalid_combine_configuration_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_config(**overrides)
