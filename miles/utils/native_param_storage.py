from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch


GROUPED_TENSOR_COMPONENT_FIELDS = (
    "rowwise_data",
    "columnwise_data",
    "scale_inv",
    "columnwise_scale_inv",
    "amax",
    "columnwise_amax",
    "scale",
    "first_dims",
    "last_dims",
    "tensor_offsets",
)


@dataclass(frozen=True)
class PhysicalStorageKey:
    device: str
    data_ptr: int
    nbytes: int


def iter_tensor_candidates(tensor: torch.Tensor) -> Iterator[object]:
    """Yield an outer tensor and its payload wrapper, if ``.data`` exposes one."""
    yield tensor
    data = getattr(tensor, "data", None)
    if data is not None and data is not tensor:
        yield data


def storage_wrapper_components(
    tensor: torch.Tensor,
) -> tuple[object | None, tuple[tuple[str, torch.Tensor], ...]]:
    """Return a native storage carrier and its physical tensor components."""
    for candidate in iter_tensor_candidates(tensor):
        flatten = getattr(candidate, "__tensor_flatten__", None)
        if callable(flatten):
            component_names, _context = flatten()
            components = tuple(
                (name, value)
                for name in component_names
                if isinstance((value := getattr(candidate, name, None)), torch.Tensor)
            )
            if components:
                return candidate, components

        if "GroupedTensor" in type(candidate).__name__:
            components = tuple(
                (name, value)
                for name in GROUPED_TENSOR_COMPONENT_FIELDS
                if isinstance((value := getattr(candidate, name, None)), torch.Tensor)
            )
            if components:
                return candidate, components

        if "MXFP8" in type(candidate).__name__.upper():
            raise RuntimeError(
                "native MXFP8 storage requires TransformerEngine's " "QuantizedTensor __tensor_flatten__ protocol"
            )
    return None, ()


def native_components(tensor: torch.Tensor) -> tuple[tuple[str, torch.Tensor], ...]:
    """Return native payload components, or an empty tuple for a plain tensor."""
    _carrier, components = storage_wrapper_components(tensor)
    return components


def iter_components(tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield physical payload tensors without dequantizing wrapper subclasses."""
    components = native_components(tensor)
    if components:
        yield from components
    else:
        yield "value", tensor


def physical_storage_key(tensor: torch.Tensor) -> PhysicalStorageKey:
    storage = tensor.untyped_storage()
    nbytes = storage.nbytes()
    data_ptr = storage.data_ptr() if nbytes else id(tensor)
    return PhysicalStorageKey(device=str(tensor.device), data_ptr=data_ptr, nbytes=nbytes)


def component_storage_keys(tensor: torch.Tensor) -> frozenset[PhysicalStorageKey]:
    return frozenset(physical_storage_key(component) for _name, component in iter_components(tensor))


def component_schema(tensor: torch.Tensor) -> tuple:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.dtype,
    )


def exact_alias_key(tensor: torch.Tensor) -> tuple[tuple, ...]:
    """Identify identical logical views, not merely views of the same storage."""
    return tuple(
        (
            name,
            physical_storage_key(component),
            component.storage_offset(),
            tuple(component.shape),
            tuple(component.stride()),
            component.dtype,
        )
        for name, component in iter_components(tensor)
    )


def tensor_schema(tensor: torch.Tensor) -> tuple:
    outer_shape = tuple(tensor.shape) if hasattr(tensor, "shape") else None
    outer_stride = tuple(tensor.stride()) if callable(getattr(tensor, "stride", None)) else None
    outer_dtype = str(getattr(tensor, "dtype", None))
    components = tuple((name, *component_schema(component)) for name, component in iter_components(tensor))
    return type(tensor), outer_shape, outer_stride, outer_dtype, components


def native_format_name(tensor: torch.Tensor) -> str:
    for candidate in iter_tensor_candidates(tensor):
        class_name = type(candidate).__name__.upper()
        if "MXFP8" in class_name:
            return "MXFP8"
        if "FLOAT8" in class_name or "FP8" in class_name:
            return "FP8"
        if "GROUPEDTENSOR" in class_name:
            quantizer_name = type(getattr(candidate, "quantizer", None)).__name__.upper()
            if "MXFP8" in quantizer_name:
                return "MXFP8_GROUPED"
            if "FLOAT8" in quantizer_name or "FP8" in quantizer_name:
                return "FP8_GROUPED"
    return str(tensor.dtype).removeprefix("torch.").upper()
