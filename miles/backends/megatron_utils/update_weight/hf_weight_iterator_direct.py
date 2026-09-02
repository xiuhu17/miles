import dataclasses
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.types import ParamInfo

from ..megatron_to_hf import convert_to_hf
from ..sglang import monkey_patch_torch_reductions
from .common import (
    NamedUpdateUnit,
    _check_and_fix_partition,
    _gather_with_stride,
    get_atomic_update_groups,
    get_named_update_units,
    is_routed_expert_param,
    named_params_and_buffers,
)
from .hf_weight_iterator_base import HfWeightIteratorBase

_BASE_PARAM_INFO_ATTRS = frozenset(
    {
        "tensor_model_parallel",
        "partition_dim",
        "partition_stride",
        "parallel_mode",
    }
)


def _publish_metadata(info: ParamInfo) -> dict:
    return {key: value for key, value in info.attrs.items() if key not in _BASE_PARAM_INFO_ATTRS}


class HfWeightIteratorDirect(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.megatron_local_param_info_buckets = _get_megatron_local_param_info_buckets(
            self.args,
            self.model,
            self.model_name,
            extra_param_info_attrs=self._extra_param_info_attrs,
        )

    def _extra_param_info_attrs(self, _name: str, _param: torch.Tensor) -> dict:
        return {}

    def get_hf_weight_chunks(self, megatron_local_weights, weight_type="base"):
        rank = dist.get_rank()

        if weight_type == "lora":
            from miles_plugins.models.inkling.lora import export_inkling_lora_hf_named

            yield export_inkling_lora_hf_named(self.model)
            return

        for megatron_local_param_infos in tqdm(
            self.megatron_local_param_info_buckets, disable=rank != 0, desc="Update weights"
        ):
            megatron_full_values = self._get_megatron_full_values(
                megatron_local_param_infos,
                megatron_local_weights,
            )
            hf_named_tensors = self._convert_to_hf_named_tensors(megatron_full_values, megatron_local_param_infos)
            yield hf_named_tensors
            del megatron_full_values

    def _convert_to_hf_named_tensors(self, megatron_full_params: Sequence[torch.Tensor], param_infos: list[ParamInfo]):
        hf_named_tensors = []
        # A converter may need siblings from the same atomic update group — those are in this
        # bucket by construction, since the group is what keeps them together.
        bucket = {info.name: param for info, param in zip(param_infos, megatron_full_params, strict=True)}
        for info, param in zip(param_infos, megatron_full_params, strict=True):
            hf_named_tensors.extend(
                convert_to_hf(self.args, self.model_name, info.name, param, self.quantization_config, bucket=bucket)
            )
        return hf_named_tensors

    def _get_megatron_full_values(
        self,
        megatron_local_param_infos: Sequence[ParamInfo],
        megatron_local_weights,
    ) -> Sequence[object]:
        """Materialize and model-parallel gather logical publish values.

        A normal value has one tensor component. Quantized subclasses may keep
        multiple physical components in one logical value, provided they expose
        and rebuild those components through the hooks below. Communication and
        chunk lifetime stay shared across all representations.
        """
        monkey_patch_torch_reductions()
        device = torch.device("cuda", torch.cuda.current_device())
        rank = dist.get_rank()
        values = []
        for info in megatron_local_param_infos:
            if rank == info.src_rank:
                value = self._materialize_local_value(
                    info,
                    megatron_local_weights[info.name],
                    device,
                )
            else:
                value = self._allocate_remote_value(info, device)
            values.append(value)

        torch.cuda.synchronize()
        self._broadcast_values_across_pp_and_ep(megatron_local_param_infos, values)
        return self._all_gather_value_components_async(megatron_local_param_infos, values)

    def _materialize_local_value(
        self,
        _info: ParamInfo,
        source: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        return source.detach().to(device=device, non_blocking=True)

    def _allocate_remote_value(self, info: ParamInfo, device: torch.device) -> torch.Tensor:
        return torch.empty(info.shape, dtype=info.dtype, device=device)

    def _value_components(self, value: object) -> tuple[torch.Tensor, ...]:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"ordinary publish value must be a tensor, got {type(value).__name__}")
        return (value,)

    def _rebuild_value(
        self,
        _old_value: object,
        components: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if len(components) != 1:
            raise RuntimeError(f"ordinary publish value requires one component, got {len(components)}")
        return components[0]

    def _component_partition(
        self,
        info: ParamInfo,
        _value: object,
        _component_idx: int,
    ) -> tuple[int, int]:
        return _check_and_fix_partition(
            self.args,
            info.name,
            info.attrs.get("partition_stride", 1),
            info.attrs.get("partition_dim", -1),
        )

    def _broadcast_values_across_pp_and_ep(
        self,
        param_infos: Sequence[ParamInfo],
        values: Sequence[object],
    ) -> None:
        rank = dist.get_rank()
        pp = get_parallel_state().pp
        if pp.size > 1:
            pp_ranks = dist.get_process_group_ranks(pp.group)
            handles = []
            for info, value in zip(param_infos, values, strict=True):
                if info.src_rank in pp_ranks:
                    handles.extend(
                        dist.broadcast(component, src=info.src_rank, group=pp.group, async_op=True)
                        for component in self._value_components(value)
                    )
            for handle in handles:
                handle.wait()

        ep = get_parallel_state().ep
        if ep.size > 1:
            ep_ranks = dist.get_process_group_ranks(ep.group)
            handles = []
            for info, value in zip(param_infos, values, strict=True):
                if is_routed_expert_param(info.name):
                    src_rank = info.src_rank if info.src_rank in ep_ranks else rank
                    handles.extend(
                        dist.broadcast(component, src=src_rank, group=ep.group, async_op=True)
                        for component in self._value_components(value)
                    )
            for handle in handles:
                handle.wait()

    def _all_gather_value_components_async(
        self,
        param_infos: Sequence[ParamInfo],
        values: Sequence[object],
    ) -> list[object]:
        """Batch TP all-gather every physical component, then rebuild logical values."""
        tasks_by_value = []
        handles = []

        for info, value in zip(param_infos, values, strict=True):
            component_tasks = []
            for component_idx, component in enumerate(self._value_components(value)):
                if not _needs_tp_gather(info):
                    component_tasks.append((component, None, None, None, None))
                    continue

                parallel = get_parallel_state().etp if is_routed_expert_param(info.name) else get_parallel_state().tp
                if parallel.size <= 1:
                    component_tasks.append((component, None, None, None, None))
                    continue

                partition_stride, partition_dim = self._component_partition(info, value, component_idx)
                partitions = [torch.empty_like(component) for _ in range(parallel.size)]
                handle = dist.all_gather(partitions, component, group=parallel.group, async_op=True)
                handles.append(handle)
                component_tasks.append((component, handle, partitions, partition_dim, partition_stride))
            tasks_by_value.append((value, component_tasks))

        for handle in handles:
            handle.wait()

        gathered_values = []
        for old_value, component_tasks in tasks_by_value:
            gathered_components = []
            for component, handle, partitions, partition_dim, partition_stride in component_tasks:
                if handle is None:
                    gathered_components.append(component)
                else:
                    gathered_components.append(_gather_with_stride(partitions, partition_dim, partition_stride))
            gathered_values.append(self._rebuild_value(old_value, tuple(gathered_components)))
        return gathered_values


def _needs_tp_gather(info: ParamInfo) -> bool:
    attrs = info.attrs
    if "expert_bias" in info.name:
        return False
    if attrs.get("parallel_mode") == "duplicated":
        return False
    unmarked_grouped_expert = (
        is_routed_expert_param(info.name)
        and ("linear_fc1.weight" in info.name or "linear_fc2.weight" in info.name)
        and not attrs.get("tensor_model_parallel", False)
        and get_parallel_state().etp.size > 1
    )
    return bool(attrs.get("tensor_model_parallel", False) or unmarked_grouped_expert)


def _get_megatron_local_param_info_buckets(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    model_name: str,
    *,
    extra_param_info_attrs,
) -> list[list[ParamInfo]]:
    """
    Partition params into buckets ≤ update_weight_buffer_size (with TP replication).

    Model-specific atomic update groups are kept in the same bucket because
    some rollout loaders must see related tensors in the same load_weights call.
    """
    param_infos = _get_megatron_local_param_infos(
        args,
        model,
        extra_param_info_attrs=extra_param_info_attrs,
    )
    param_names = [info.name for info in param_infos]
    atomic_update_groups = get_atomic_update_groups(args, model_name)
    update_units = get_named_update_units(param_names, atomic_update_groups)
    return _pack_update_units(args, param_infos, update_units)


def _get_param_full_size(info: ParamInfo) -> int:
    if is_routed_expert_param(info.name):
        tp_size = get_parallel_state().etp.size
    else:
        tp_size = get_parallel_state().tp.size
    return info.size * tp_size


def _pack_update_units(
    args: Namespace, param_infos: list[ParamInfo], update_units: list[NamedUpdateUnit]
) -> list[list[ParamInfo]]:
    by_name = {info.name: info for info in param_infos}
    param_info_buckets: list[list[ParamInfo]] = [[]]
    buffer_size = 0

    for unit in update_units:
        params = [by_name[name] for name in unit.names]
        unit_size = sum(_get_param_full_size(param) for param in params)
        if buffer_size + unit_size > args.update_weight_buffer_size and param_info_buckets[-1]:
            param_info_buckets.append([])
            buffer_size = 0
        param_info_buckets[-1].extend(params)
        buffer_size += unit_size

    return param_info_buckets


def _get_megatron_local_param_infos(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    *,
    extra_param_info_attrs,
) -> list[ParamInfo]:
    """
    Build global param metadata: collect → exchange PP/EP → resolve duplicates (MTP virtual PP)
    by min src_rank → validate. Returns sorted ParamInfo identical across all ranks.
    """
    pp_size = get_parallel_state().pp.size
    ep_size = get_parallel_state().ep.size

    from ..lora_utils import _is_adapter_param_name

    param_infos = {}
    rank = dist.get_rank()
    for name, param in named_params_and_buffers(args, model):
        if _is_adapter_param_name(name):
            continue
        extra_attrs = extra_param_info_attrs(name, param)
        overlapping_attrs = _BASE_PARAM_INFO_ATTRS.intersection(extra_attrs)
        if overlapping_attrs:
            raise RuntimeError(f"extra ParamInfo attrs override built-in attrs: {sorted(overlapping_attrs)}")
        info = ParamInfo(
            name=name,
            dtype=param.dtype,
            shape=param.shape,
            attrs={
                "tensor_model_parallel": getattr(param, "tensor_model_parallel", False),
                "partition_dim": getattr(param, "partition_dim", -1),
                "partition_stride": getattr(param, "partition_stride", 1),
                "parallel_mode": getattr(param, "parallel_mode", None),
                **extra_attrs,
            },
            size=param.numel() * param.element_size(),
            src_rank=rank,
        )
        previous = param_infos.get(name)
        if previous is not None:
            previous_extra_attrs = _publish_metadata(previous)
            if previous_extra_attrs != extra_attrs:
                raise RuntimeError(
                    f"duplicate parameter {name!r} has incompatible publish metadata: "
                    f"{previous_extra_attrs} != {extra_attrs}"
                )
        param_infos[name] = info

    if pp_size > 1:
        param_infos_list = [None] * pp_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=get_parallel_state().pp.group
        )
        for src_rank, infos in param_infos_list:
            if src_rank == rank:
                continue
            for name, info in infos.items():
                if name in param_infos:
                    assert args.mtp_num_layers is not None
                    old_info = param_infos[name]
                    if _publish_metadata(old_info) != _publish_metadata(info):
                        raise RuntimeError(f"parameter {name!r} has inconsistent publish metadata across PP ranks")
                    if old_info.src_rank > src_rank:
                        param_infos[name] = info
                else:
                    param_infos[name] = info

    if ep_size > 1:
        param_infos_list = [None] * ep_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=get_parallel_state().ep.group
        )
        for src_rank, infos in param_infos_list:
            for name, info in infos.items():
                if name in param_infos:
                    if _publish_metadata(param_infos[name]) != _publish_metadata(info):
                        raise RuntimeError(f"parameter {name!r} has inconsistent publish metadata across EP ranks")
                else:
                    # here we need to set the src_rank to the rank within the expert model parallel group
                    info = dataclasses.replace(info, src_rank=src_rank)
                    param_infos[name] = info

    param_infos = list(param_infos.values())
    param_infos = sorted(param_infos, key=lambda info: info.name)

    # Check all ranks has the same parameter info
    all_param_info_list = [None] * dist.get_world_size()
    dist.all_gather_object(
        obj=param_infos,
        object_list=all_param_info_list,
        group=get_gloo_group(),
    )
    for i, param_info in enumerate(param_infos):
        expected_extra_attrs = _publish_metadata(param_info)
        for infos in all_param_info_list:
            assert infos[i].name == param_info.name, f"Parameter name mismatch: {infos[i].name} != {param_info.name}"
            assert (
                infos[i].shape == param_info.shape
            ), f"Parameter shape mismatch: {infos[i].shape} != {param_info.shape}"
            assert (
                infos[i].dtype == param_info.dtype
            ), f"Parameter dtype mismatch: {infos[i].dtype} != {param_info.dtype}"
            actual_extra_attrs = _publish_metadata(infos[i])
            assert actual_extra_attrs == expected_extra_attrs, (
                f"Parameter publish metadata mismatch for {param_info.name}: "
                f"{actual_extra_attrs} != {expected_extra_attrs}"
            )

    return param_infos
