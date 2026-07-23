from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=[])


import pytest
import torch
import yaml
from scripts.run_deepseek_v4 import _DSV4_NVFP4_TE_PRECISION_CONFIG
from tools.convert_hf_to_nvfp4_blockfp8 import _SkipRules as MixedSkipRules
from tools.convert_hf_to_nvfp4_blockfp8 import fp32_pow2_scale_to_e8m0
from tools.convert_hf_to_nvfp4_blockfp8 import quantize_blockfp8_ue8m0
from tools.convert_hf_to_nvfp4_blockfp8 import should_quantize_blockfp8
from tools.convert_hf_to_nvfp4_blockfp8 import should_quantize_nvfp4 as mixed_should_quantize_nvfp4
from tools.convert_hf_to_nvfp4 import quantize_nvfp4 as tool_quantize_nvfp4
from tools.convert_hf_to_nvfp4 import should_quantize as tool_should_quantize_nvfp4

try:
    # TE renamed the reference module quantization_nvfp4 -> quantization_ref_nvfp4.
    from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import NVFP4QuantizerRef
except ImportError:
    from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import NVFP4QuantizerRef

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import (
    NVFP4_GROUP_SIZE,
    _nvfp4_global_decode_scale_te,
)
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import (
    quantize_nvfp4 as processor_quantize_nvfp4,
)
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import quantize_params_nvfp4
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4_blockfp8 import (
    quantize_params_nvfp4_blockfp8,
)

NVFP4_SHAPES = [
    (1, 64),
    (1, 1024),
    (3, 128),
    (16, 64),
    (64, 128),
    (128, 64),
    (256, 128),
    (512, 256),
    (128, 1024),
    (1024, 2048),
    (7168, 2048),
    (2048, 7168),
    (128, 16384),
]

MIXED_QUANTIZATION_CONFIG = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
    "scale_fmt": "ue8m0",
    "quant_algo": "MIXED_PRECISION",
    "moe_quant_algo": "NVFP4",
    "group_size": 16,
    "ignore": [],
}


def _make_weight(init_data: str, dtype: torch.dtype, shape: tuple[int, int], device: str) -> torch.Tensor:
    m, n = shape
    if init_data == "random":
        return torch.randn((m, n), dtype=dtype, device=device)
    if init_data == "boundary":
        base = torch.linspace(-12.0, 12.0, steps=n // 2, dtype=torch.float32, device=device)
        eps = torch.full_like(base, 1e-3)
        eps = torch.maximum(eps, 1e-4 * torch.ones_like(base))
        row = torch.empty(n, dtype=torch.float32, device=device)
        row[0::2] = base - eps
        row[1::2] = base + eps
        return row.unsqueeze(0).repeat(m, 1).to(dtype=dtype)
    if init_data == "zeros":
        return torch.zeros((m, n), dtype=dtype, device=device)
    if init_data == "maxes":
        return torch.full((m, n), torch.finfo(dtype).max, dtype=dtype, device=device)
    raise ValueError(f"Unknown init_data: {init_data}")


def _te_nvfp4_reference(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = weight.contiguous()
    global_amax = torch.max(torch.abs(weight.to(torch.float32)))
    qweight, block_scale = NVFP4QuantizerRef._quantize_blockwise_reference(
        weight,
        global_amax,
        NVFP4_GROUP_SIZE,
        1,
        pow_2_scales=False,
        eps=0.0,
    )
    return qweight, block_scale, _nvfp4_global_decode_scale_te(global_amax)


def test_nvfp4_quantize_params_requires_complete_gated_pair():
    weight = torch.randn((4, NVFP4_GROUP_SIZE), dtype=torch.float32)
    with pytest.raises(ValueError, match="requires gate/up tensors to be quantized together"):
        quantize_params_nvfp4(
            args=None,
            megatron_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
            converted_named_params=[
                ("model.layers.0.mlp.experts.0.gate_proj.weight", weight),
            ],
            quantization_config={"quant_method": "nvfp4"},
        )


def test_nvfp4_quantize_params_respects_extra_high_precision_layers_megatron():
    weight = torch.randn((4, NVFP4_GROUP_SIZE), dtype=torch.bfloat16)
    converted_named_params = [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", weight),
        ("model.layers.0.mlp.experts.0.up_proj.weight", weight),
    ]
    args = type("Args", (), {"extra_high_precision_layers_megatron": ("linear_fc1",)})()

    out = quantize_params_nvfp4(
        args=args,
        megatron_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
        converted_named_params=converted_named_params,
        quantization_config={"quant_method": "nvfp4"},
    )

    assert out is converted_named_params


@pytest.mark.parametrize("layer_idx", [0, 3])
def test_nvfp4_quantize_params_respects_first_last_layers_bf16(layer_idx):
    weight = torch.randn((4, NVFP4_GROUP_SIZE), dtype=torch.bfloat16)
    converted_named_params = [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", weight),
        ("model.layers.0.mlp.experts.0.up_proj.weight", weight),
    ]
    args = type(
        "Args",
        (),
        {
            "first_last_layers_bf16": True,
            "num_layers": 4,
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
        },
    )()

    out = quantize_params_nvfp4(
        args=args,
        megatron_name=f"decoder.layers.{layer_idx}.mlp.experts.linear_fc1.weight0",
        converted_named_params=converted_named_params,
        quantization_config={"quant_method": "nvfp4"},
    )

    assert out is converted_named_params


def test_nvfp4_hf_should_quantize_respects_extra_high_precision_layers_hf():
    weight = torch.randn((4, NVFP4_GROUP_SIZE), dtype=torch.bfloat16)

    assert not tool_should_quantize_nvfp4(
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        weight,
        skip_weight_substrings=("mlp.experts.0",),
    )
    assert tool_should_quantize_nvfp4(
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        weight,
        skip_weight_substrings=("mlp.experts.1",),
    )


def test_deepseek_v4_mixed_checkpoint_dtype_contract():
    weight = torch.empty((128, 128), dtype=torch.bfloat16)
    skip_rules = MixedSkipRules(
        num_hidden_layers=4,
        num_layers_at_start_in_bf16=0,
        num_layers_at_end_in_bf16=0,
        extra_high_precision_layers_hf=(),
    )

    nvfp4_names = (
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.0.up_proj.weight",
        "model.layers.1.mlp.experts.0.down_proj.weight",
    )
    blockfp8_names = (
        "model.layers.1.self_attn.wq_a.weight",
        "model.layers.1.self_attn.wq_b.weight",
        "model.layers.1.self_attn.wkv.weight",
        "model.layers.1.self_attn.wo_b.weight",
        "model.layers.1.self_attn.indexer.wq_b.weight",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.layers.1.mlp.shared_experts.up_proj.weight",
        "model.layers.1.mlp.shared_experts.down_proj.weight",
    )
    high_precision_names = (
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
        "model.layers.1.self_attn.q_norm.weight",
        "model.layers.1.self_attn.kv_norm.weight",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.self_attn.wo_a.weight",
        "model.layers.1.self_attn.compressor.wkv.weight",
        "model.layers.1.self_attn.compressor.wgate.weight",
        "model.layers.1.self_attn.indexer.compressor.wkv.weight",
        "model.layers.1.self_attn.indexer.compressor.wgate.weight",
        "model.layers.1.self_attn.indexer.weights_proj.weight",
        "model.layers.1.self_attn.attn_sink",
        "model.layers.1.self_attn.compressor.ape",
        "model.layers.1.self_attn.indexer.compressor.ape",
        "model.layers.1.hc_attn_base",
        "model.layers.1.hc_attn_fn",
        "model.layers.1.hc_attn_scale",
        "model.layers.1.hc_ffn_base",
        "model.layers.1.hc_ffn_fn",
        "model.layers.1.hc_ffn_scale",
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
    )

    for name in nvfp4_names:
        assert mixed_should_quantize_nvfp4(name, weight, skip_rules)
        assert not should_quantize_blockfp8(name, weight, skip_rules)
    for name in blockfp8_names:
        assert should_quantize_blockfp8(name, weight, skip_rules)
        assert not mixed_should_quantize_nvfp4(name, weight, skip_rules)
    for name in high_precision_names:
        assert not should_quantize_blockfp8(name, weight, skip_rules)
        assert not mixed_should_quantize_nvfp4(name, weight, skip_rules)


def test_deepseek_v4_te_precision_config_keeps_shared_experts_blockfp8():
    config = yaml.safe_load(_DSV4_NVFP4_TE_PRECISION_CONFIG)

    assert config["configs"]["blockfp8"]["training_recipe"] == {
        "fp8_quantization_recipe": "blockwise"
    }
    assert config["matchers"]["shared_experts_fc1_blockfp8"] == {
        "type": "glob",
        "enabled": True,
        "pattern": "*.mlp.shared_experts.linear_fc1",
        "config": "blockfp8",
    }
    assert config["matchers"]["shared_experts_fc2_blockfp8"] == {
        "type": "glob",
        "enabled": True,
        "pattern": "*.mlp.shared_experts.linear_fc2",
        "config": "blockfp8",
    }
    assert config["matchers"]["routed_experts_fc1_nvfp4"]["config"] == "nvfp4"
    assert config["matchers"]["routed_experts_fc2_nvfp4"]["config"] == "nvfp4"
    assert config["matchers"]["dsa_indexer_weights_proj_bf16"]["config"] == "bf16"


def test_deepseek_v4_cold_blockfp8_storage_dtype():
    weight = torch.randn((128, 128), dtype=torch.bfloat16, device="cuda")
    qweight, scale = quantize_blockfp8_ue8m0(weight)
    packed_scale = fp32_pow2_scale_to_e8m0(scale)

    assert qweight.dtype == torch.float8_e4m3fn
    assert packed_scale.dtype == torch.float8_e8m0fnu
    assert packed_scale.shape == (1, 1)


def test_deepseek_v4_online_update_keeps_shared_experts_blockfp8():
    megatron_name = "module.module.decoder.layers.1.mlp.shared_experts.linear_fc2.weight"
    converted_name = "model.layers.1.mlp.shared_experts.down_proj.weight"
    weight = torch.randn((128, 128), dtype=torch.bfloat16, device="cuda")
    output = quantize_params_nvfp4_blockfp8(
        args=None,
        megatron_name=megatron_name,
        converted_named_params=[(converted_name, weight)],
        quantization_config=MIXED_QUANTIZATION_CONFIG,
    )
    output = dict(output)

    assert output[converted_name].dtype == torch.float8_e4m3fn
    scale_name = converted_name.replace(".weight", ".weight_scale_inv")
    # Cold checkpoints store one E8M0 byte per 128x128 block. Online updates
    # send the same exponents in SGLang/DeepGEMM's TMA-packed int32 layout.
    assert output[scale_name].dtype == torch.int32
    assert converted_name.replace(".weight", ".weight_scale_2") not in output
    assert converted_name.replace(".weight", ".input_scale") not in output


def test_deepseek_v4_online_update_keeps_routed_experts_nvfp4():
    gate = torch.randn((128, 128), dtype=torch.bfloat16, device="cuda")
    up = torch.randn((128, 128), dtype=torch.bfloat16, device="cuda")
    gate_name = "model.layers.1.mlp.experts.0.gate_proj.weight"
    up_name = "model.layers.1.mlp.experts.0.up_proj.weight"
    output = dict(
        quantize_params_nvfp4_blockfp8(
            args=None,
            megatron_name="module.module.decoder.layers.1.mlp.experts.linear_fc1.weight0",
            converted_named_params=[(gate_name, gate), (up_name, up)],
            quantization_config=MIXED_QUANTIZATION_CONFIG,
        )
    )

    for name in (gate_name, up_name):
        assert output[name].dtype == torch.uint8
        assert output[name.replace(".weight", ".weight_scale")].dtype == torch.float8_e4m3fn
        assert output[name.replace(".weight", ".weight_scale_2")].dtype == torch.float32
        assert output[name.replace(".weight", ".input_scale")].dtype == torch.float32
        assert name.replace(".weight", ".weight_scale_inv") not in output


@pytest.mark.parametrize(
    ("megatron_name", "converted_name"),
    [
        (
            "module.module.decoder.layers.1.self_attention.indexer.linear_weights_proj.weight",
            "model.layers.1.self_attn.indexer.weights_proj.weight",
        ),
        (
            "module.module.decoder.layers.1.self_attention.wo_a.weight",
            "model.layers.1.self_attn.wo_a.weight",
        ),
        (
            "module.module.decoder.layers.1.self_attention.compressor.wkv.weight",
            "model.layers.1.self_attn.compressor.wkv.weight",
        ),
        (
            "module.module.decoder.layers.1.self_attention.compressor.wgate.weight",
            "model.layers.1.self_attn.compressor.wgate.weight",
        ),
        (
            "module.module.decoder.layers.1.self_attention.indexer.compressor.wkv.weight",
            "model.layers.1.self_attn.indexer.compressor.wkv.weight",
        ),
        (
            "module.module.decoder.layers.1.self_attention.indexer.compressor.wgate.weight",
            "model.layers.1.self_attn.indexer.compressor.wgate.weight",
        ),
        (
            "module.module.decoder.layers.1.mlp.router.weight",
            "model.layers.1.mlp.gate.weight",
        ),
    ],
    ids=[
        "indexer_weights_proj",
        "wo_a",
        "compressor_wkv",
        "compressor_wgate",
        "indexer_compressor_wkv",
        "indexer_compressor_wgate",
        "router",
    ],
)
def test_deepseek_v4_online_update_preserves_high_precision_targets(
    megatron_name, converted_name
):
    weight = torch.randn((128, 128), dtype=torch.bfloat16, device="cuda")
    converted_named_params = [(converted_name, weight)]
    output = quantize_params_nvfp4_blockfp8(
        args=None,
        megatron_name=megatron_name,
        converted_named_params=converted_named_params,
        quantization_config=MIXED_QUANTIZATION_CONFIG,
    )

    assert output is converted_named_params
    assert output[0][0] == converted_name
    assert output[0][1] is weight


@pytest.mark.parametrize(
    "quantize_fn",
    [processor_quantize_nvfp4, tool_quantize_nvfp4],
    ids=["processor", "convert_tool"],
)
@pytest.mark.parametrize("shape", NVFP4_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
def test_nvfp4_quantize_matches_te_reference_bitwise(quantize_fn, shape, dtype, init_data):
    device = "cuda"
    torch.manual_seed(42)

    weight = _make_weight(init_data, dtype, shape, device)
    qweight, block_scale, global_scale = quantize_fn(weight)
    qweight_ref, block_scale_ref, global_scale_ref = _te_nvfp4_reference(weight)

    torch.testing.assert_close(qweight, qweight_ref, rtol=0, atol=0)
    torch.testing.assert_close(block_scale.view(torch.uint8), block_scale_ref.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(global_scale, global_scale_ref, rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
