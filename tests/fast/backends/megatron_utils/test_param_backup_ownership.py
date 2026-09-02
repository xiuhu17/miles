import sys
from argparse import Namespace
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from miles.backends.megatron_utils.param_backup_ownership import (
    LORA_ADAPTER_NO_BACKUP_TMS_TAG,
    PRIMARY_NO_BACKUP_TMS_TAG,
    BackupOwner,
    compile_actor_backup_ownership,
    lora_adapter_allocation_region,
    primary_model_allocation_region,
)


def _args(**overrides):
    values = dict(
        lora_rank=0,
        lora_adapter_path=None,
        offload_train=True,
        fp8_param_gather=True,
        rematerialize_param_from_master_weight=False,
        disable_grad_buffers_cpu_backup=True,
        disable_param_buffers_cpu_backup=True,
    )
    values.update(overrides)
    return Namespace(**values)


def _compile(args, tensors, *, role="actor", rematerializable_ids=None, backup=True, snapshots=False):
    return compile_actor_backup_ownership(
        args,
        tensors,
        role=role,
        rematerializable_ids=rematerializable_ids,
        needs_pinned_actor_backup=backup,
        has_model_snapshots=snapshots,
    )


class LegacyMXFP8Tensor:
    def __init__(self):
        self.requires_grad = True
        self.dtype = torch.bfloat16
        self._rowwise_data = torch.ones(8, dtype=torch.uint8)
        self._rowwise_scale_inv = torch.ones(2, dtype=torch.uint8)


def test_full_ft_remat_off_uses_only_pinned_owner():
    weight = torch.nn.Parameter(torch.ones(8))
    plan = _compile(_args(), [("weight", weight)])

    assert plan.owner_of(weight) == BackupOwner.PINNED
    assert list(plan.select([("weight", weight)], BackupOwner.PINNED)) == [("weight", weight)]
    plan.assert_pinned_coverage({"weight"})


def test_full_ft_remat_on_only_pins_non_rematerializable_extras():
    weight = torch.nn.Parameter(torch.ones(8))
    extra = torch.ones(2)
    args = _args(rematerialize_param_from_master_weight=True)
    plan = _compile(args, [("weight", weight), ("expert_bias", extra)], rematerializable_ids={id(weight)})

    assert plan.owner_of(weight) == BackupOwner.NONE
    assert plan.owner_of(extra) == BackupOwner.PINNED
    plan.assert_pinned_coverage({"expert_bias"})


def test_lora_frozen_base_is_tms_owned_and_adapter_has_no_backup():
    base = torch.nn.Parameter(torch.ones(8), requires_grad=False)
    adapter = torch.nn.Parameter(torch.ones(2), requires_grad=True)
    plan = _compile(_args(lora_rank=8), [("base", base), ("adapter", adapter)], backup=False)

    assert plan.owner_of(base) == BackupOwner.TMS
    assert plan.owner_of(adapter) == BackupOwner.NONE
    plan.assert_pinned_coverage(set())


def test_lora_args_do_not_make_critic_parameters_tms_owned():
    weight = torch.nn.Parameter(torch.ones(8))
    plan = _compile(_args(lora_rank=8), [("weight", weight)], role="critic")

    assert plan.owner_of(weight) == BackupOwner.PINNED


def test_lora_rejects_second_pinned_model_version_owner():
    base = torch.nn.Parameter(torch.ones(8), requires_grad=False)
    with pytest.raises(NotImplementedError, match="single-owner"):
        _compile(_args(lora_rank=8), [("base", base)], backup=True, snapshots=True)


def test_alias_with_two_semantic_owners_is_rejected_by_physical_storage():
    storage = torch.ones(8)
    frozen = torch.nn.Parameter(storage, requires_grad=False)
    adapter = torch.nn.Parameter(storage, requires_grad=True)

    with pytest.raises(RuntimeError, match="physical allocation"):
        _compile(_args(lora_rank=8), [("base", frozen), ("adapter", adapter)], backup=False)


def test_pinned_coverage_rejects_tms_or_none_tensor():
    base = torch.nn.Parameter(torch.ones(8), requires_grad=False)
    plan = _compile(_args(lora_rank=8), [("base", base)], backup=False)

    with pytest.raises(RuntimeError, match="TMS/NONE"):
        plan.assert_pinned_coverage({"base"})


def test_legacy_mxfp8_without_flatten_protocol_has_no_ownership_fallback():
    with pytest.raises(RuntimeError, match="__tensor_flatten__"):
        _compile(_args(offload_train=False), [("weight", LegacyMXFP8Tensor())])


@pytest.mark.parametrize(
    ("target", "expected_kwargs"),
    [
        ("cpu", {"tag": "default", "enable_cpu_backup": True}),
        ("disk", {"tag": "default", "enable_disk_backup": True}),
    ],
)
def test_lora_model_build_uses_explicit_tms_backed_region(monkeypatch, target, expected_kwargs):
    region = MagicMock(return_value=nullcontext())
    fake_module = ModuleType("torch_memory_saver")
    fake_module.torch_memory_saver = SimpleNamespace(region=region)
    monkeypatch.setitem(sys.modules, "torch_memory_saver", fake_module)

    with primary_model_allocation_region(_args(lora_rank=8, offload_train_target=target), role="actor"):
        pass

    region.assert_called_once_with(**expected_kwargs)


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_full_ft_model_build_uses_explicit_no_backup_region(monkeypatch, role):
    region = MagicMock(return_value=nullcontext())
    fake_module = ModuleType("torch_memory_saver")
    fake_module.torch_memory_saver = SimpleNamespace(region=region)
    monkeypatch.setitem(sys.modules, "torch_memory_saver", fake_module)

    with primary_model_allocation_region(_args(), role=role):
        pass

    region.assert_called_once_with(
        tag=PRIMARY_NO_BACKUP_TMS_TAG,
        enable_cpu_backup=False,
    )


def test_lora_adapter_build_overrides_backed_base_with_no_backup_region(monkeypatch):
    region = MagicMock(return_value=nullcontext())
    fake_module = ModuleType("torch_memory_saver")
    fake_module.torch_memory_saver = SimpleNamespace(region=region)
    monkeypatch.setitem(sys.modules, "torch_memory_saver", fake_module)

    with lora_adapter_allocation_region(_args(lora_rank=8)):
        pass

    region.assert_called_once_with(
        tag=LORA_ADAPTER_NO_BACKUP_TMS_TAG,
        enable_cpu_backup=False,
    )


def test_model_regions_are_disabled_without_train_offload(monkeypatch):
    region = MagicMock(return_value=nullcontext())
    fake_module = ModuleType("torch_memory_saver")
    fake_module.torch_memory_saver = SimpleNamespace(region=region)
    monkeypatch.setitem(sys.modules, "torch_memory_saver", fake_module)
    args = _args(offload_train=False, lora_rank=8)

    with primary_model_allocation_region(args, role="actor"):
        pass
    with lora_adapter_allocation_region(args):
        pass

    region.assert_not_called()
