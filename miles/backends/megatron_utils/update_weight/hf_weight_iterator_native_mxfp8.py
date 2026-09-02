from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core, convert_to_hf
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp8 import should_quantize_param_mxfp8
from miles.backends.megatron_utils.quantized_storage import MXFP8PublishComponents, TEQuantizedStorageAdapter
from miles.utils.native_param_storage import native_format_name
from miles.utils.types import ParamInfo

from .hf_weight_iterator_direct import HfWeightIteratorDirect

logger = logging.getLogger(__name__)

_STORAGE_SCHEMA_ATTR = "native_publish_storage_schema"
_STORAGE_TYPE_ATTR = "native_publish_storage_type"
_STORAGE_FORMAT_ATTR = "native_publish_storage_format"
_STORAGE_ERROR_ATTR = "native_publish_storage_error"
_UNSUPPORTED_QUANTIZED_FORMATS = frozenset({"FP8", "FP8_GROUPED", "MXFP8_GROUPED"})
_COMPONENT_SAFE_MODEL_MARKERS = (
    "glm4moelite",
    "deepseekv3",
    "glmmoedsa",
    "glm_moe_dsa",
    "deepseekv4",
)


def _has_component_safe_converter(model_name: str) -> bool:
    return any(marker in model_name.lower() for marker in _COMPONENT_SAFE_MODEL_MARKERS)


class HfWeightIteratorNativeMXFP8(HfWeightIteratorDirect):
    """Publish TE MXFP8 primary bytes through the shared direct iterator.

    Only representation-specific extraction, allocation, partition validation,
    and HF naming live here. PP/EP broadcast, TP gather, bucketing, and chunk
    lifetime are inherited from :class:`HfWeightIteratorDirect`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.megatron_to_hf_mode != "raw":
            raise RuntimeError("native MXFP8 base publish is implemented only for --megatron-to-hf-mode raw")
        if not _has_component_safe_converter(self.model_name):
            raise RuntimeError(
                f"native MXFP8 publish is not validated for model converter {self.model_name!r}; "
                "its weight transform must preserve compact scale geometry before it can be enabled"
            )
        if self.quantization_config is None or self.quantization_config.get("quant_method") != "mxfp8":
            raise RuntimeError(
                "native MXFP8 publish requires an SGLang MXFP8 target " "(quantization_config.quant_method='mxfp8')"
            )

        all_infos = [info for bucket in self.megatron_local_param_info_buckets for info in bucket]
        for info in all_infos:
            if storage_error := info.attrs[_STORAGE_ERROR_ATTR]:
                raise RuntimeError(f"invalid native storage for {info.name!r}: {storage_error}")
            storage_format = info.attrs[_STORAGE_FORMAT_ATTR]
            if storage_format in _UNSUPPORTED_QUANTIZED_FORMATS:
                raise RuntimeError(
                    f"native MXFP8 publish does not support trainer storage format "
                    f"{storage_format} for {info.name!r}"
                )

        target_infos = [info for info in all_infos if should_quantize_param_mxfp8(self.args, info.name)]
        if not target_infos:
            raise RuntimeError("--fp8-param-gather produced zero native MXFP8 publish targets")
        for info in target_infos:
            if info.attrs[_STORAGE_SCHEMA_ATTR] is None:
                raise RuntimeError(
                    f"SGLang expects {info.name!r} in MXFP8, but trainer storage is "
                    f"{info.attrs[_STORAGE_TYPE_ATTR]}. Every native-publish target must be "
                    "a TE MXFP8 primary; no quantizer fallback is allowed."
                )

        if dist.get_rank() == 0:
            logger.info("native MXFP8 publish inventory: native=%d", len(target_infos))

    def _extra_param_info_attrs(self, _name: str, param: torch.Tensor) -> dict:
        storage_format = native_format_name(param)
        storage_error = None
        try:
            adapter = TEQuantizedStorageAdapter.from_tensor(param)
        except RuntimeError as exc:
            adapter = None
            storage_error = str(exc)
        return {
            _STORAGE_SCHEMA_ATTR: None if adapter is None else adapter.schema_digest,
            _STORAGE_TYPE_ATTR: f"{type(param).__module__}.{type(param).__qualname__}",
            _STORAGE_FORMAT_ATTR: storage_format,
            _STORAGE_ERROR_ATTR: storage_error,
        }

    def _materialize_local_value(self, info: ParamInfo, source, device: torch.device):
        adapter = TEQuantizedStorageAdapter.from_tensor(source)
        expected_schema = info.attrs[_STORAGE_SCHEMA_ATTR]
        current_schema = None if adapter is None else adapter.schema_digest
        if current_schema != expected_schema:
            raise RuntimeError(f"parameter storage schema changed after initialization for {info.name!r}")

        if should_quantize_param_mxfp8(self.args, info.name):
            if adapter is None:
                source_type = f"{type(source).__module__}.{type(source).__qualname__}"
                raise RuntimeError(
                    f"SGLang expects {info.name!r} in MXFP8, but publish storage is {source_type}; "
                    "refusing BF16 re-quantization fallback"
                )
            return adapter.compact_publish_components(device=device)

        if adapter is not None:
            # Explicit high-precision rollout exception. This is transient
            # staging, never a resident BF16 parameter backup.
            return adapter.dequantize(device=device, dtype=info.dtype)
        return super()._materialize_local_value(info, source, device)

    def _allocate_remote_value(self, info: ParamInfo, device: torch.device):
        if not should_quantize_param_mxfp8(self.args, info.name):
            return super()._allocate_remote_value(info, device)

        shape = tuple(info.shape)
        if len(shape) < 2 or shape[-1] % 32 != 0:
            raise RuntimeError(f"native MXFP8 publish requires rank >= 2 and K divisible by 32, got {shape}")
        return MXFP8PublishComponents(
            data=torch.empty(shape, dtype=torch.float8_e4m3fn, device=device),
            scale_inv=torch.empty((*shape[:-1], shape[-1] // 32), dtype=torch.uint8, device=device),
        )

    def _value_components(self, value: object) -> tuple[torch.Tensor, ...]:
        if isinstance(value, MXFP8PublishComponents):
            return value.data, value.scale_inv
        return super()._value_components(value)

    def _rebuild_value(
        self,
        old_value: object,
        components: tuple[torch.Tensor, ...],
    ):
        if not isinstance(old_value, MXFP8PublishComponents):
            return super()._rebuild_value(old_value, components)
        if len(components) != 2:
            raise RuntimeError(f"native MXFP8 publish value requires two components, got {len(components)}")
        return MXFP8PublishComponents(data=components[0], scale_inv=components[1])

    def _component_partition(
        self,
        info: ParamInfo,
        value: object,
        component_idx: int,
    ) -> tuple[int, int]:
        partition_stride, partition_dim = super()._component_partition(info, value, component_idx)
        if partition_dim < 0:
            raise RuntimeError(f"cannot gather {info.name!r}: invalid partition_dim={partition_dim}")
        if (
            isinstance(value, MXFP8PublishComponents)
            and component_idx == 1
            and partition_dim == len(info.shape) - 1
            and partition_stride != 1
        ):
            raise RuntimeError(
                f"MXFP8 K-partitioned scale gather does not support partition_stride={partition_stride} "
                f"for {info.name!r}"
            )
        return partition_stride, partition_dim

    def _convert_to_hf_named_tensors(self, values, infos: list[ParamInfo]):
        out: list[tuple[str, torch.Tensor]] = []
        ordinary_bucket = {
            info.name: value for info, value in zip(infos, values, strict=True) if isinstance(value, torch.Tensor)
        }
        for info, value in zip(infos, values, strict=True):
            target_mxfp8 = should_quantize_param_mxfp8(self.args, info.name)
            if isinstance(value, MXFP8PublishComponents):
                if not target_mxfp8:
                    raise RuntimeError(f"unexpected native MXFP8 payload for {info.name}")
                data_outputs = _convert_to_hf_core(self.args, self.model_name, info.name, value.data)
                scale_outputs = _convert_to_hf_core(self.args, self.model_name, info.name, value.scale_inv)
                if [name for name, _ in data_outputs] != [name for name, _ in scale_outputs]:
                    raise RuntimeError(
                        f"MXFP8 data/scale conversion name mismatch for {info.name}: "
                        f"data={[name for name, _ in data_outputs]}, "
                        f"scale={[name for name, _ in scale_outputs]}"
                    )
                for (hf_name, data), (_scale_hf_name, scale) in zip(data_outputs, scale_outputs, strict=True):
                    if not hf_name.endswith(".weight"):
                        raise RuntimeError(f"native MXFP8 converter produced a non-weight output: {hf_name}")
                    components = MXFP8PublishComponents(
                        data=data.contiguous(),
                        scale_inv=scale.contiguous(),
                    )
                    out.append((hf_name, components.data))
                    out.append((hf_name.removesuffix(".weight") + ".weight_scale_inv", components.scale_inv))
                continue

            if target_mxfp8:
                raise RuntimeError(
                    f"SGLang expects {info.name!r} in MXFP8, but its publish value is "
                    f"ordinary {type(value).__name__}; refusing BF16 re-quantization fallback"
                )
            out.extend(
                convert_to_hf(
                    self.args,
                    self.model_name,
                    info.name,
                    value,
                    quantization_config=None,
                    bucket=ordinary_bucket,
                )
            )
        return out
