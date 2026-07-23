from .padding_remover import remove_padding
from .quantizer_compressed_tensors import quantize_params_compressed_tensors
from .quantizer_fp8 import quantize_params_fp8
from .quantizer_mxfp8 import quantize_params_mxfp8
from .quantizer_nvfp4 import quantize_params_nvfp4
from .quantizer_nvfp4_blockfp8 import (
    is_nvfp4_blockfp8_quantization_config,
    quantize_params_nvfp4_blockfp8,
)

__all__ = [
    "remove_padding",
    "quantize_param",
    "quantize_params_fp8",
    "quantize_params_mxfp8",
    "quantize_params_nvfp4",
    "quantize_params_nvfp4_blockfp8",
    "quantize_params_compressed_tensors",
]


def quantize_params(args, megatron_name, converted_named_params, quantization_config):
    if quantization_config is None:
        return converted_named_params
    elif is_nvfp4_blockfp8_quantization_config(quantization_config):
        # DeepSeek-V4 mixed checkpoint: NVFP4 routed experts + blockwise FP8.
        # Must be checked before plain fp8 (it also declares quant_method=fp8).
        return quantize_params_nvfp4_blockfp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "fp8":
        return quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "mxfp8":
        return quantize_params_mxfp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] in ("nvfp4",) or quantization_config.get("quant_algo") == "NVFP4":
        # Pure NVFP4 checkpoints (tools/convert_hf_to_nvfp4.py, modelopt).
        return quantize_params_nvfp4(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "compressed-tensors":
        # only int4 at the moment.
        return quantize_params_compressed_tensors(converted_named_params, quantization_config)
