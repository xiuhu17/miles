"""
python tools/convert_hf_to_nvfp4_blockfp8.py [-h] --model-dir MODEL_DIR --save-dir SAVE_DIR
                                             [--device DEVICE]
                                             [--num-layers-at-start-in-bf16 N]
                                             [--num-layers-at-end-in-bf16 M]
                                             [--extra-high-precision-layers-hf ...]

Convert a BF16/FP16/FP32 DeepSeek-V4 HF safetensors checkpoint (e.g. the output
of tools/fp8_cast_bf16.py) into the mixed NVFP4 + blockwise-FP8 rollout format:

  - routed expert weights (model.layers.N.mlp.experts.M.{gate,up,down}_proj.weight)
      -> NVFP4 (modelopt layout): .weight (uint8 packed E2M1, [N, K/2])
                                  .weight_scale (float8_e4m3fn, [N, K/16])
                                  .weight_scale_2 (float32 scalar)
                                  .input_scale (float32 scalar, = 1.0)
      gate_proj/up_proj of the same expert share one global amax so sglang's
      modelopt MoE loader sees identical w1/w3 scales (no dequant/requant).
  - attention linears (wq_a, wq_b, wkv, wo_b, indexer.wq_b, indexer.wk)
    and shared expert projections
      -> blockwise FP8: .weight (float8_e4m3fn, [N, K])
                        .weight_scale_inv (float8_e8m0fnu power-of-2 scale,
                                           [ceil(N/128), ceil(K/128)])
      Quantized with sglang's quant_weight_ue8m0 (DeepGEMM-style ceil-to-pow2),
      the same kernel the miles weight-update quantizer uses, so the cold-start
      bytes match what the first weight update would produce.
  - everything else (wo_a, both compressors' wkv/wgate, norms, router/FFN
    gate, embeddings, lm_head, attn_sink, ape, hc_*, indexer weights_proj,
    ...) stays high precision.

The output config.json quantization_config declares the sglang hybrid contract
(quant_method=fp8 + quant_algo=MIXED_PRECISION + moe_quant_algo=NVFP4), which
routes routed-expert FusedMoE layers through ModelOptNvFp4FusedMoEMethod and
everything else through the blockwise-FP8 path (HybridFp8NvFp4Config).

NVFP4 quantization follows the TransformerEngine reference implementation
(NVFP4QuantizerRef, 1D 1x16 tiles, no RHT / no stochastic rounding / no 4over6),
matching processors/quantizer_nvfp4.py used for online weight updates.
Run with NVTE_USE_FAST_MATH=0 (TE default) for bitwise-stable quantization.
"""

import argparse
import gc
import json
import os
import re
import shutil

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm

FP8_BLOCK_SIZE = [128, 128]
NVFP4_GROUP_SIZE = 16

# Routed experts only: shared experts stay blockwise FP8 in DeepSeek-V4.
ROUTED_EXPERT_WEIGHT_RE = re.compile(
    r"\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")

GATED_PAIR_SUFFIXES = {
    ".gate_proj.weight": "gate",
    ".up_proj.weight": "up",
}

# Blockwise-FP8 targets after Megatron->HF renaming (deepseekv4.py), matching
# the DeepSeek-V4 mixed-precision contract. Shared experts are deliberately
# blockwise FP8; wo_a and the other high-precision carve-outs are absent.
BLOCKFP8_WEIGHT_SUFFIXES = (
    ".self_attn.wq_a.weight",
    ".self_attn.wq_b.weight",
    ".self_attn.wkv.weight",
    ".self_attn.wo_b.weight",
    ".self_attn.indexer.wq_b.weight",
    ".self_attn.indexer.wk.weight",
    ".mlp.shared_experts.gate_proj.weight",
    ".mlp.shared_experts.up_proj.weight",
    ".mlp.shared_experts.down_proj.weight",
)

HIGH_PRECISION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def quantize_nvfp4(weight, global_amax=None):
    # Shared with the online weight-update quantizer so cold start and weight
    # update produce identical NVFP4 bytes (TE reference quantizer).
    from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import (
        quantize_nvfp4 as _quantize_nvfp4,
    )

    return _quantize_nvfp4(weight, global_amax=global_amax)


def quantize_blockfp8_ue8m0(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 -> FP8-E4M3 with 128x128 power-of-2 (UE8M0) scales.

    Uses sglang's quant_weight_ue8m0 (DeepGEMM ceil-to-pow2 semantics), the same
    function processors/quantizer_fp8.py uses for online weight updates.
    Returns (qweight float8_e4m3fn [N, K], scale float32 pow2 [N/128, K/128]).
    """
    from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0

    qweight, scale = quant_weight_ue8m0(weight, weight_block_size=FP8_BLOCK_SIZE)
    return qweight, scale


def fp32_pow2_scale_to_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Pack an FP32 power-of-2 scale into FP8-E8M0 storage (exact).

    E8M0 stores only the biased FP32 exponent, so `bits >> 23` is the exact
    encoding; falls back to FP32 on torch builds without float8_e8m0fnu
    (both dtypes load correctly into sglang's fp32 weight_scale_inv buffer).
    """
    if not hasattr(torch, "float8_e8m0fnu"):
        return scale
    bits = scale.contiguous().view(torch.int32)
    if torch.any(bits & 0x007FFFFF) or torch.any(bits < 0):
        raise ValueError("blockwise FP8 scale is not a positive power of 2; cannot pack to E8M0")
    return ((bits >> 23) & 0xFF).to(torch.uint8).view(torch.float8_e8m0fnu)


def _strip_weight_suffix(weight_key: str) -> str:
    if not weight_key.endswith(".weight"):
        raise ValueError(f"Expected key ending with '.weight', got: {weight_key}")
    return weight_key[: -len(".weight")]


def _get_layer_idx(name: str) -> int | None:
    match = LAYER_IDX_RE.search(name)
    return int(match.group(1)) if match else None


def _split_gated_pair_name(name: str) -> tuple[str | None, str | None]:
    for suffix, role in GATED_PAIR_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], role
    return None, None


class _SkipRules:
    """Shared skip logic so both passes and the config lists stay aligned."""

    def __init__(
        self,
        num_hidden_layers: int,
        num_layers_at_start_in_bf16: int,
        num_layers_at_end_in_bf16: int,
        extra_high_precision_layers_hf: tuple[str, ...],
    ):
        self.num_hidden_layers = num_hidden_layers
        head_end_idx = num_layers_at_start_in_bf16
        tail_start_idx = num_hidden_layers - num_layers_at_end_in_bf16
        self.bf16_layer_indices = set(range(0, head_end_idx)) | set(
            range(tail_start_idx, num_hidden_layers)
        )
        self.extra_high_precision_layers_hf = extra_high_precision_layers_hf

    def is_skipped(self, name: str) -> bool:
        if any(substr in name for substr in self.extra_high_precision_layers_hf):
            return True
        layer_idx = _get_layer_idx(name)
        if layer_idx is not None and layer_idx in self.bf16_layer_indices:
            return True
        return False

    def is_mtp_layer(self, name: str) -> bool:
        # NextN/MTP tensors live at model.layers.{num_hidden_layers}+; sglang
        # excludes them from NVFP4 (model.decoder.*), so keep them blockwise FP8.
        layer_idx = _get_layer_idx(name)
        return layer_idx is not None and layer_idx >= self.num_hidden_layers


def _is_routed_expert_weight(name: str) -> bool:
    return ROUTED_EXPERT_WEIGHT_RE.search(name) is not None


def _check_quantizable(name: str, weight: torch.Tensor, group_k: int) -> None:
    if weight.dtype not in HIGH_PRECISION_DTYPES:
        raise ValueError(
            f"{name} has dtype {weight.dtype}; expected a BF16/FP16/FP32 checkpoint. "
            "Run tools/fp8_cast_bf16.py first to dequantize the FP8 source checkpoint."
        )
    if weight.dim() != 2:
        raise ValueError(f"{name} must be 2D to quantize, got shape {tuple(weight.shape)}")
    if weight.shape[-1] % group_k != 0:
        raise ValueError(f"{name} last dim {weight.shape[-1]} not divisible by {group_k}")


def should_quantize_nvfp4(name: str, weight: torch.Tensor, skip_rules: _SkipRules) -> bool:
    if not _is_routed_expert_weight(name):
        return False
    if skip_rules.is_skipped(name) or skip_rules.is_mtp_layer(name):
        return False
    _check_quantizable(name, weight, NVFP4_GROUP_SIZE)
    return True


def should_quantize_blockfp8(name: str, weight: torch.Tensor, skip_rules: _SkipRules) -> bool:
    is_blockfp8_target = name.endswith(BLOCKFP8_WEIGHT_SUFFIXES) or (
        skip_rules.is_mtp_layer(name) and _is_routed_expert_weight(name)
    )
    if not is_blockfp8_target:
        return False
    if skip_rules.is_skipped(name):
        return False
    _check_quantizable(name, weight, 1)
    return True


def _collect_shared_global_amax(
    input_path: str,
    safetensors_files: list[str],
    device: str,
    skip_rules: _SkipRules,
) -> dict[str, torch.Tensor]:
    """Collect per-expert gate/up shared amax across all shards (w1/w3 must
    share one global scale for sglang's modelopt NVFP4 MoE loader)."""
    gate_amax: dict[str, torch.Tensor] = {}
    up_amax: dict[str, torch.Tensor] = {}
    for filename in safetensors_files:
        with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device=device) as f:
            for key in f.keys():
                base, role = _split_gated_pair_name(key)
                if base is None or role is None:
                    continue
                tensor = f.get_tensor(key)
                if not should_quantize_nvfp4(key, tensor, skip_rules):
                    continue
                amax = tensor.abs().max().to(torch.float32)
                target = gate_amax if role == "gate" else up_amax
                prev = target.get(base)
                target[base] = amax if prev is None else torch.max(prev, amax)

    shared_global_amax: dict[str, torch.Tensor] = {}
    for base in gate_amax.keys() & up_amax.keys():
        shared_global_amax[base] = torch.max(gate_amax[base], up_amax[base])
    for base in gate_amax.keys() ^ up_amax.keys():
        raise ValueError(f"Incomplete gate/up pair for {base}; cannot share NVFP4 global amax.")
    return shared_global_amax


class ConversionResult:
    def __init__(self) -> None:
        self.weight_map: dict[str, str] = {}
        self.total_size: int = 0
        self.modules_to_not_convert: list[str] = []
        self.nvfp4_ignore: set[str] = set()

    def add_file(self, filename: str, q_weights: dict[str, torch.Tensor], module_names: list[str]) -> None:
        for key, tensor in q_weights.items():
            self.weight_map[key] = filename
            self.total_size += tensor.numel() * tensor.element_size()
        self.modules_to_not_convert.extend(module_names)


def process_file(
    input_path: str,
    output_path: str,
    filename: str,
    result_collector: ConversionResult,
    device: str,
    skip_rules: _SkipRules,
    shared_global_amax: dict[str, torch.Tensor],
) -> None:
    q_weights: dict[str, torch.Tensor] = {}
    modules_to_not_convert: list[str] = []

    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device=device) as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if not key.endswith(".weight"):
                q_weights[key] = tensor
                continue

            if should_quantize_nvfp4(key, tensor, skip_rules):
                base, _role = _split_gated_pair_name(key)
                global_amax = shared_global_amax.get(base) if base else None
                qweight, block_scale, weight_scale_2 = quantize_nvfp4(tensor, global_amax=global_amax)
                q_weights[key] = qweight
                q_weights[key.replace(".weight", ".weight_scale")] = block_scale
                q_weights[key.replace(".weight", ".weight_scale_2")] = weight_scale_2
                q_weights[key.replace(".weight", ".input_scale")] = torch.ones_like(
                    weight_scale_2, dtype=torch.float32
                )
            elif should_quantize_blockfp8(key, tensor, skip_rules):
                qweight, scale = quantize_blockfp8_ue8m0(tensor)
                q_weights[key] = qweight
                q_weights[key.replace(".weight", ".weight_scale_inv")] = fp32_pow2_scale_to_e8m0(scale)
            else:
                if _is_routed_expert_weight(key) and not skip_rules.is_mtp_layer(key):
                    # NVFP4-skipped routed experts must be excluded from the
                    # modelopt NVFP4 loader via quantization_config["ignore"].
                    expert_module = key[: key.rindex(".mlp.experts.")] + ".mlp.experts"
                    result_collector.nvfp4_ignore.add(expert_module)
                elif ".experts." not in key:
                    modules_to_not_convert.append(_strip_weight_suffix(key))
                q_weights[key] = tensor

    safetensors.torch.save_file(q_weights, os.path.join(output_path, filename), metadata={"format": "pt"})
    result_collector.add_file(filename, q_weights, modules_to_not_convert)


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.findall(r"\d+|\D+", s)]


def convert_nvfp4_blockfp8(
    model_dir: str,
    save_dir: str,
    device: str,
    num_layers_at_start_in_bf16: int = 0,
    num_layers_at_end_in_bf16: int = 0,
    extra_high_precision_layers_hf: tuple[str, ...] = (),
) -> None:
    input_path = os.path.abspath(model_dir)
    output_path = os.path.abspath(save_dir)
    os.makedirs(output_path, exist_ok=True)

    config_path = os.path.join(input_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    num_hidden_layers = int(cfg["num_hidden_layers"])

    skip_rules = _SkipRules(
        num_hidden_layers=num_hidden_layers,
        num_layers_at_start_in_bf16=num_layers_at_start_in_bf16,
        num_layers_at_end_in_bf16=num_layers_at_end_in_bf16,
        extra_high_precision_layers_hf=extra_high_precision_layers_hf,
    )

    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            shutil.copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))

    index_path = os.path.join(input_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    safetensors_files = sorted(set(weight_map.values()))

    shared_global_amax = _collect_shared_global_amax(input_path, safetensors_files, device, skip_rules)

    result_collector = ConversionResult()
    for filename in tqdm(safetensors_files, desc="Processing files"):
        process_file(
            input_path,
            output_path,
            filename,
            result_collector,
            device,
            skip_rules,
            shared_global_amax,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # This tool consumes the HF-named checkpoint produced by fp8_cast_bf16
    # (model.layers.N.self_attn... / .mlp.experts.N.gate_proj...). A native-named
    # DeepSeek-V4 checkpoint (layers.N.attn... / .ffn.experts.N.w1...) would
    # silently match nothing; fail loudly instead of writing a bogus mixed ckpt.
    produced_nvfp4 = any(k.endswith(".weight_scale_2") for k in result_collector.weight_map)
    produced_blockfp8 = any(k.endswith(".weight_scale_inv") for k in result_collector.weight_map)
    if not produced_nvfp4 and not produced_blockfp8 and not skip_rules.bf16_layer_indices:
        raise ValueError(
            "No tensor was quantized: the input does not look like an HF-named "
            "DeepSeek-V4 BF16 checkpoint (expected model.layers.N.self_attn.* / "
            "model.layers.N.mlp.experts.N.{gate,up,down}_proj.weight). Native-named "
            "checkpoints (layers.N.attn.* / .ffn.experts.N.w1...) are not supported; "
            "run tools/fp8_cast_bf16.py first."
        )

    # sglang excludes NVFP4 per FusedMoE layer; a layer with only some experts
    # quantized cannot be represented.
    for module in result_collector.nvfp4_ignore:
        prefix = module + "."
        for key in result_collector.weight_map:
            if key.startswith(prefix) and key.endswith(".weight_scale_2"):
                raise ValueError(
                    f"Layer {module} has both NVFP4-quantized and skipped routed experts "
                    f"(e.g. {key}); adjust --extra-high-precision-layers-hf to cover whole layers."
                )

    # Blockwise-FP8 base (Fp8Config) + NVFP4 MoE markers: sglang detects
    # quant_algo=MIXED_PRECISION + moe_quant_algo=NVFP4 + group_size and wraps
    # the fp8 config into HybridFp8NvFp4Config (routed FusedMoE -> modelopt fp4).
    quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": FP8_BLOCK_SIZE,
        "scale_fmt": "ue8m0",
        "quant_algo": "MIXED_PRECISION",
        "moe_quant_algo": "NVFP4",
        "group_size": NVFP4_GROUP_SIZE,
        "ignore": sorted(result_collector.nvfp4_ignore, key=_natural_key),
    }
    if result_collector.modules_to_not_convert:
        quantization_config["modules_to_not_convert"] = sorted(
            set(result_collector.modules_to_not_convert), key=_natural_key
        )

    cfg["quantization_config"] = quantization_config
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    index_dict = {
        "weight_map": result_collector.weight_map,
        "metadata": {"total_size": result_collector.total_size},
    }
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index_dict, f, indent=2)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the BF16 HF safetensors model.")
    parser.add_argument("--save-dir", type=str, required=True, help="Path to save the converted model.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to run quantization on (default: cuda).",
    )
    parser.add_argument(
        "--num-layers-at-start-in-bf16",
        type=int,
        default=0,
        help="Keep first N decoder layers unquantized.",
    )
    parser.add_argument(
        "--num-layers-at-end-in-bf16",
        type=int,
        default=0,
        help="Keep last N decoder layers unquantized.",
    )
    parser.add_argument(
        "--extra-high-precision-layers-hf",
        type=str,
        nargs="*",
        default=(),
        help="Extra substrings for HF weight names to skip quantization.",
    )
    args, _ = parser.parse_known_args()

    if isinstance(args.device, str) and args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot run NVFP4+blockFP8 quantization.")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    if not os.path.exists(args.save_dir):
        print(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise ValueError("The save_dir should be a directory.")

    convert_nvfp4_blockfp8(
        args.model_dir,
        args.save_dir,
        str(device),
        num_layers_at_start_in_bf16=args.num_layers_at_start_in_bf16,
        num_layers_at_end_in_bf16=args.num_layers_at_end_in_bf16,
        extra_high_precision_layers_hf=tuple(
            s.strip() for s in args.extra_high_precision_layers_hf if isinstance(s, str) and s.strip()
        ),
    )


if __name__ == "__main__":
    main()
