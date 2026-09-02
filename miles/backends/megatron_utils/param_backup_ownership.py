from __future__ import annotations

import logging
from argparse import Namespace
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum

import torch

from miles.backends.megatron_utils.lora_utils import is_lora_enabled
from miles.utils.native_param_storage import (
    PhysicalStorageKey,
    iter_components,
    native_format_name,
    physical_storage_key,
)

logger = logging.getLogger(__name__)

PRIMARY_NO_BACKUP_TMS_TAG = "primary_no_backup"
LORA_ADAPTER_NO_BACKUP_TMS_TAG = "lora_adapter"


class BackupOwner(str, Enum):
    """The only component allowed to retain recovery bytes for an allocation."""

    NONE = "none"
    PINNED = "pinned"
    TMS = "tms"


@dataclass(frozen=True)
class _LogicalTensorOwnership:
    name: str
    tensor: torch.Tensor
    owner: BackupOwner
    format_name: str


@dataclass(frozen=True)
class BackupOwnershipPlan:
    """Immutable actor parameter ownership compiled once after model setup."""

    entries: tuple[_LogicalTensorOwnership, ...]
    _owner_by_tensor_id: Mapping[int, BackupOwner] = field(init=False, repr=False, compare=False)
    _owner_by_storage: Mapping[PhysicalStorageKey, BackupOwner] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        owner_by_tensor_id: dict[int, BackupOwner] = {}
        owner_by_storage: dict[PhysicalStorageKey, BackupOwner] = {}
        logical_name_by_storage: dict[PhysicalStorageKey, str] = {}

        for entry in self.entries:
            old_tensor_owner = owner_by_tensor_id.setdefault(id(entry.tensor), entry.owner)
            if old_tensor_owner != entry.owner:
                raise RuntimeError(
                    f"tensor {entry.name!r} has conflicting backup owners: "
                    f"{old_tensor_owner.value} and {entry.owner.value}"
                )

            for component_name, component in iter_components(entry.tensor):
                storage_key = physical_storage_key(component)
                old_storage_owner = owner_by_storage.setdefault(storage_key, entry.owner)
                if old_storage_owner != entry.owner:
                    old_name = logical_name_by_storage[storage_key]
                    raise RuntimeError(
                        "one physical allocation cannot have two backup owners: "
                        f"{old_name!r}={old_storage_owner.value}, "
                        f"{entry.name}.{component_name}={entry.owner.value}"
                    )
                logical_name_by_storage.setdefault(storage_key, f"{entry.name}.{component_name}")

        object.__setattr__(self, "_owner_by_tensor_id", owner_by_tensor_id)
        object.__setattr__(self, "_owner_by_storage", owner_by_storage)

    def owner_of(self, tensor: torch.Tensor) -> BackupOwner:
        try:
            return self._owner_by_tensor_id[id(tensor)]
        except KeyError as exc:
            raise RuntimeError("actor parameter set changed after backup ownership was compiled") from exc

    def select(
        self,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
        owner: BackupOwner,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in named_tensors:
            if self.owner_of(tensor) == owner:
                yield name, tensor

    def expected_names(self, owner: BackupOwner) -> set[str]:
        return {entry.name for entry in self.entries if entry.owner == owner}

    def assert_pinned_coverage(self, pinned_names: Iterable[str]) -> None:
        expected = self.expected_names(BackupOwner.PINNED)
        actual = set(pinned_names)
        if missing := expected - actual:
            raise RuntimeError(f"PINNED owner has no TensorBackuper coverage: {sorted(missing)[:20]}")
        if unexpected := actual - expected:
            raise RuntimeError("TensorBackuper retained tensors owned by TMS/NONE: " f"{sorted(unexpected)[:20]}")

    def log_inventory(self) -> None:
        totals: dict[tuple[BackupOwner, str], int] = defaultdict(int)
        seen: set[PhysicalStorageKey] = set()
        for entry in self.entries:
            for _component_name, component in iter_components(entry.tensor):
                storage_key = physical_storage_key(component)
                if storage_key in seen:
                    continue
                seen.add(storage_key)
                totals[(entry.owner, entry.format_name)] += storage_key.nbytes

        for (owner, format_name), nbytes in sorted(totals.items(), key=lambda item: (item[0][0].value, item[0][1])):
            logger.info(
                "parameter backup inventory: owner=%s format=%s bytes=%d (%.2f GiB)",
                owner.value.upper(),
                format_name,
                nbytes,
                nbytes / 1024**3,
            )


def compile_actor_backup_ownership(
    args: Namespace,
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str = "actor",
    rematerializable_ids: set[int] | None,
    needs_pinned_actor_backup: bool,
    has_model_snapshots: bool,
) -> BackupOwnershipPlan:
    """Choose one stable recovery owner for every actor tensor."""
    if getattr(args, "offload_train", False):
        assert args.disable_grad_buffers_cpu_backup
        assert args.disable_param_buffers_cpu_backup

    lora = role == "actor" and is_lora_enabled(args)
    if lora and has_model_snapshots:
        raise NotImplementedError(
            "LoRA single-owner backup does not support Megatron-side ref/teacher/old-actor "
            "model switching. Frozen base weights are TMS-owned and cannot also keep a "
            "per-version pinned snapshot. Use an independent ref/teacher model."
        )

    entries = []
    rematerializable_ids = rematerializable_ids or set()
    for name, tensor in named_tensors:
        if lora:
            owner = (
                BackupOwner.TMS
                if getattr(args, "offload_train", False) and not tensor.requires_grad
                else BackupOwner.NONE
            )
        elif getattr(args, "rematerialize_param_from_master_weight", False):
            owner = BackupOwner.NONE if id(tensor) in rematerializable_ids else BackupOwner.PINNED
        else:
            owner = BackupOwner.PINNED if needs_pinned_actor_backup else BackupOwner.NONE
        entries.append(
            _LogicalTensorOwnership(
                name=name,
                tensor=tensor,
                owner=owner,
                format_name=native_format_name(tensor),
            )
        )

    plan = BackupOwnershipPlan(entries=tuple(entries))
    plan.log_inventory()
    return plan


def primary_model_allocation_region(args: Namespace, role: str):
    """Select the immutable TMS policy before primary storage is allocated."""
    if not getattr(args, "offload_train", False):
        return nullcontext()

    from torch_memory_saver import torch_memory_saver

    if role == "actor" and is_lora_enabled(args):
        if getattr(args, "offload_train_target", "cpu") == "disk":
            return torch_memory_saver.region(tag="default", enable_disk_backup=True)
        return torch_memory_saver.region(tag="default", enable_cpu_backup=True)

    return torch_memory_saver.region(
        tag=PRIMARY_NO_BACKUP_TMS_TAG,
        enable_cpu_backup=False,
    )


def lora_adapter_allocation_region(args: Namespace):
    """Keep adapter storage live while the TMS-owned frozen base is paused."""
    if not getattr(args, "offload_train", False):
        return nullcontext()

    from torch_memory_saver import torch_memory_saver

    return torch_memory_saver.region(
        tag=LORA_ADAPTER_NO_BACKUP_TMS_TAG,
        enable_cpu_backup=False,
    )
