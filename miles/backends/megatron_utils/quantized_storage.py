from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

from miles.utils.native_param_storage import iter_tensor_candidates


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _as_device(device: torch.device | int | str) -> torch.device:
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)


@dataclass(frozen=True)
class MXFP8PublishComponents:
    """SGLang loader representation of one logical MXFP8 weight."""

    data: torch.Tensor
    scale_inv: torch.Tensor

    def __post_init__(self) -> None:
        if self.data.dtype != torch.float8_e4m3fn:
            raise TypeError(f"MXFP8 publish data must be E4M3, got {self.data.dtype}")
        if self.scale_inv.dtype != torch.uint8:
            raise TypeError(f"MXFP8 publish scales must be UE8M0 uint8, got {self.scale_inv.dtype}")
        if self.data.ndim < 2 or self.scale_inv.ndim != self.data.ndim:
            raise ValueError(
                "MXFP8 data and compact scale must have matching rank >= 2: "
                f"data={tuple(self.data.shape)}, scale={tuple(self.scale_inv.shape)}"
            )
        if self.data.shape[:-1] != self.scale_inv.shape[:-1]:
            raise ValueError(
                "MXFP8 compact scale leading dimensions must match data: "
                f"data={tuple(self.data.shape)}, scale={tuple(self.scale_inv.shape)}"
            )
        if self.data.shape[-1] % 32 != 0 or self.scale_inv.shape[-1] != self.data.shape[-1] // 32:
            raise ValueError(
                "MXFP8 compact scale must contain one UE8M0 value per 32 data values: "
                f"data={tuple(self.data.shape)}, scale={tuple(self.scale_inv.shape)}"
            )


class TEQuantizedStorageAdapter:
    """Narrow, fail-closed view over TE's public tensor-flatten protocol.

    TransformerEngine QuantizedTensor wrappers have no useful outer storage.  This
    adapter is the only MILES module that interprets TE's inner component names.
    V1 intentionally supports only an unswizzled MXFP8 E4M3 rowwise representation.
    """

    _MXFP8_COMPONENTS = frozenset(
        {
            "_rowwise_data",
            "_rowwise_scale_inv",
            "_columnwise_data",
            "_columnwise_scale_inv",
        }
    )
    _REQUIRED_MXFP8_COMPONENTS = frozenset({"_rowwise_data", "_rowwise_scale_inv"})

    def __init__(self, carrier, component_names: tuple[str, ...], context: dict) -> None:
        self._carrier = carrier
        self._component_names = component_names
        self._context = context
        self._validate_mxfp8()

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TEQuantizedStorageAdapter | None:
        """Return an adapter for TE MXFP8 storage, or ``None`` for an ordinary tensor.

        Other quantized formats are rejected by the native-MXFP8 caller rather than
        being guessed here.  A class identifying itself as MXFP8 but lacking TE's
        flatten protocol is always an error.
        """
        saw_mxfp8 = False
        for candidate in iter_tensor_candidates(tensor):
            class_name = type(candidate).__name__.upper()
            if "MXFP8" not in class_name:
                continue
            saw_mxfp8 = True
            flatten = getattr(candidate, "__tensor_flatten__", None)
            if not callable(flatten):
                continue
            component_names, context = flatten()
            return cls(candidate, tuple(component_names), context)

        if saw_mxfp8:
            raise RuntimeError(
                "native MXFP8 publish requires TransformerEngine's " "QuantizedTensor __tensor_flatten__ protocol"
            )
        return None

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return tuple(self._carrier.size())

    @property
    def fake_dtype(self) -> torch.dtype:
        return self._carrier.dtype

    @property
    def publish_nbytes(self) -> int:
        rows = math.prod(self.logical_shape[:-1])
        cols = self.logical_shape[-1]
        return rows * cols + rows * (cols // 32)

    @property
    def schema_digest(self) -> str:
        h = hashlib.sha256()
        h.update(type(self._carrier).__module__.encode())
        h.update(type(self._carrier).__qualname__.encode())
        h.update(repr(self.logical_shape).encode())
        h.update(str(self.fake_dtype).encode())
        for name in self._component_names:
            component = getattr(self._carrier, name)
            h.update(name.encode())
            h.update(repr(tuple(component.shape)).encode())
            h.update(str(component.dtype).encode())
        return h.hexdigest()

    def compact_publish_components(
        self,
        *,
        device: torch.device | int | str,
        non_blocking: bool = True,
    ) -> MXFP8PublishComponents:
        """Copy only SGLang's rowwise data and unpadded UE8M0 scales."""
        device = _as_device(device)
        logical_shape = self.logical_shape
        rows = math.prod(logical_shape[:-1])
        cols = logical_shape[-1]

        rowwise_data = self._carrier._rowwise_data
        rowwise_scale_inv = self._carrier._rowwise_scale_inv

        data = rowwise_data.detach().to(device=device, non_blocking=non_blocking)
        data = data.view(torch.float8_e4m3fn).reshape(logical_shape).contiguous()

        # TE pads the physical scale matrix to [multiple-of-128 rows,
        # multiple-of-4 blocks]. SGLang's checkpoint/hot-loader ABI is compact.
        compact_scale = rowwise_scale_inv.detach()[:rows, : cols // 32]
        compact_scale = compact_scale.reshape(*logical_shape[:-1], cols // 32)
        compact_scale = compact_scale.to(device=device, non_blocking=non_blocking).contiguous()
        return MXFP8PublishComponents(data=data, scale_inv=compact_scale)

    def dequantize(self, *, device: torch.device | int | str, dtype: torch.dtype) -> torch.Tensor:
        """Explicit transient high-precision fallback for a non-quantized rollout layer."""
        device = _as_device(device)
        carrier = self._carrier
        if carrier.device != device:
            carrier = carrier.to(device=device, non_blocking=True)
        dequantize = getattr(carrier, "dequantize", None)
        if not callable(dequantize):
            raise RuntimeError(f"{type(carrier).__name__} does not expose dequantize()")
        return dequantize(dtype=dtype)

    def _validate_mxfp8(self) -> None:
        names = frozenset(self._component_names)
        if not self._REQUIRED_MXFP8_COMPONENTS.issubset(names):
            raise RuntimeError("TE MXFP8 storage is missing required rowwise components: " f"found={sorted(names)}")
        if not names.issubset(self._MXFP8_COMPONENTS):
            raise RuntimeError(
                "unrecognized TE MXFP8 component schema; refusing native publish: " f"found={sorted(names)}"
            )

        context = self._context
        if not isinstance(context, dict):
            raise RuntimeError("TE MXFP8 __tensor_flatten__ context must be a dict")
        nontensor = context.get("nontensor_kwargs")
        if not isinstance(nontensor, dict):
            raise RuntimeError("TE MXFP8 flatten context has no nontensor_kwargs")
        if nontensor.get("with_gemm_swizzled_scales"):
            raise RuntimeError("native publish does not accept GEMM-swizzled TE MXFP8 primary scales")
        fp8_dtype = str(nontensor.get("fp8_dtype"))
        if "E4M3" not in fp8_dtype.upper():
            raise RuntimeError(f"native MXFP8 publish supports E4M3 only, got {fp8_dtype}")

        logical_shape = self.logical_shape
        if len(logical_shape) < 2 or logical_shape[-1] % 32 != 0:
            raise RuntimeError(
                "native MXFP8 publish requires rank >= 2 and K divisible by 32, got " f"{logical_shape}"
            )
        rows = math.prod(logical_shape[:-1])
        cols = logical_shape[-1]
        data = self._carrier._rowwise_data
        scale = self._carrier._rowwise_scale_inv
        if data.dtype != torch.uint8 or tuple(data.shape) != logical_shape:
            raise RuntimeError(
                "unexpected TE MXFP8 rowwise data metadata: "
                f"shape={tuple(data.shape)} dtype={data.dtype}, logical={logical_shape}"
            )
        expected_scale_shape = (_round_up(rows, 128), _round_up(cols // 32, 4))
        if scale.dtype != torch.uint8 or tuple(scale.shape) != expected_scale_shape:
            raise RuntimeError(
                "unexpected TE MXFP8 rowwise scale metadata: "
                f"shape={tuple(scale.shape)} dtype={scale.dtype}, expected={expected_scale_shape}/torch.uint8"
            )
        if not data.is_contiguous() or not scale.is_contiguous():
            raise RuntimeError("native MXFP8 publish requires contiguous TE rowwise components")


def is_te_mxfp8_tensor(tensor: torch.Tensor) -> bool:
    return TEQuantizedStorageAdapter.from_tensor(tensor) is not None
