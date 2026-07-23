"""Mixed NVFP4 + blockwise-FP8 online weight-update quantizer (DeepSeek-V4).

Checkpoint contract (sglang HybridFp8NvFp4Config, see
tools/convert_hf_to_nvfp4_blockfp8.py for the cold-start counterpart):
  - routed experts (mlp.experts.linear_fc1/linear_fc2) -> NVFP4 modelopt layout
    (.weight uint8 packed / .weight_scale e4m3 / .weight_scale_2 fp32 /
     .input_scale fp32), gate/up sharing one global amax.
  - everything else that the fp8 pipeline quantizes (attention linears, shared
    experts, MTP) -> blockwise FP8 with UE8M0 scales, delegated verbatim to
    quantize_params_fp8 so it cannot drift from the plain-FP8 pipeline.

quantization_config comes from the mixed checkpoint's config.json:
  {quant_method: "fp8", weight_block_size: [128, 128], scale_fmt: "ue8m0",
   quant_algo: "MIXED_PRECISION", moe_quant_algo: "NVFP4", group_size: 16,
   ignore: [...]}
"""

import re

from .quantizer_fp8 import quantize_params_fp8
from .quantizer_nvfp4 import _get_ignore_rules, _quantize_moe_params


def is_nvfp4_blockfp8_quantization_config(quantization_config) -> bool:
    if not isinstance(quantization_config, dict):
        return False
    return (
        quantization_config.get("quant_method") == "fp8"
        and str(quantization_config.get("quant_algo", "")).upper() == "MIXED_PRECISION"
        and str(quantization_config.get("moe_quant_algo", "")).upper() == "NVFP4"
    )


def quantize_params_nvfp4_blockfp8(args, megatron_name, converted_named_params, quantization_config):
    assert is_nvfp4_blockfp8_quantization_config(quantization_config)
    group_size = int(quantization_config.get("group_size", 16))
    assert group_size == 16, f"NVFP4 requires group_size 16, got {group_size}"
    weight_block_size = quantization_config.get("weight_block_size")
    assert list(weight_block_size or ()) == [128, 128], (
        f"DeepSeek-V4 blockwise FP8 requires weight_block_size [128, 128], "
        f"got {weight_block_size}"
    )
    assert quantization_config.get("fmt", "e4m3") == "e4m3"
    assert quantization_config.get("scale_fmt") == "ue8m0"
    assert quantization_config.get("activation_scheme") == "dynamic"

    if getattr(args, "extra_high_precision_layers_megatron", False):
        for layer_name in getattr(args, "extra_high_precision_layers_megatron", ()):
            if layer_name in megatron_name:
                return converted_named_params

    decoder_layers_pattern = r"decoder\.layers\.(\d+)\.(.+)"
    match = re.search(decoder_layers_pattern, megatron_name)
    is_mtp = False
    if not match:
        mtp_layer_pattern = r"mtp\.layers\.(\d+)\.(.+)"
        match = re.search(mtp_layer_pattern, megatron_name)
        if not match:
            return converted_named_params
        is_mtp = True
        layer_idx, rest = match.groups()
        rest = rest.replace("transformer_layer.", "")
    else:
        layer_idx, rest = match.groups()

    # Skip quantization for BF16 head/tail of the main decoder layers
    # (must match --num-layers-at-start/end-in-bf16 used at checkpoint
    # conversion time so the rollout keeps those layers in BF16).
    if getattr(args, "first_last_layers_bf16", False):
        num_layers = int(args.num_layers)
        num_layers_at_start_in_bf16 = int(getattr(args, "num_layers_at_start_in_bf16", 0))
        num_layers_at_end_in_bf16 = int(getattr(args, "num_layers_at_end_in_bf16", 0))
        head_end_idx = num_layers_at_start_in_bf16
        tail_start_idx = num_layers - num_layers_at_end_in_bf16
        if int(layer_idx) < head_end_idx or int(layer_idx) >= tail_start_idx:
            return converted_named_params

    # Routed experts of the main decoder -> NVFP4. MTP experts stay on the FP8
    # path below: sglang excludes NextN MoE from NVFP4 (model.decoder.*).
    if not is_mtp:
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        expert_match = re.match(expert_pattern, rest)
        if expert_match and expert_match.groups()[0] in ("linear_fc1", "linear_fc2"):
            return _quantize_moe_params(converted_named_params, _get_ignore_rules(quantization_config))

    # Selected attention linears, shared experts, dense MLP, and MTP follow the
    # existing blockwise-FP8 path. High-precision carve-outs such as wo_a,
    # compressors, routers, and norms are deliberately passed through by it.
    return quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config)
