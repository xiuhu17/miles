import copy
import dataclasses
import itertools
import json
import os

from miles.backends.megatron_utils.update_weight.hf_weight_iterator import (
    MegatronHfWeightIteratorBase,
    _iter_mm_tower_units,
)
from miles.utils import megatron_bridge_utils
from miles.utils.lora import is_lora_weight_name

from ..megatron_to_hf import postprocess_hf_param
from ..megatron_to_hf.processors import quantize_params
from ..misc_utils import strip_param_name_prefix


class HfWeightIteratorBridge(MegatronHfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)

        if (
            self.quantization_config is not None
            and self.quantization_config.get("quant_method") == "compressed-tensors"
        ):
            quantized_basenames = _load_quantized_param_basenames(self.args.hf_checkpoint)
            if quantized_basenames is not None:
                # Quantize exactly the params the checkpoint stores packed; the
                # published ignore list of multimodal checkpoints (e.g.
                # Kimi-K2.5 VL) omits vision_tower/mm_projector, so it cannot
                # be trusted as the sole quantization criterion.
                self.quantization_config = {
                    **self.quantization_config,
                    "_miles_quantized_basenames": quantized_basenames,
                }

    def _iter_hf_param_units(self, weights, *, materialize):
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)
            named_weights = self._bridge.export_hf_weights(
                self.model,
                cpu=False,
                conversion_tasks=conversion_tasks,
                merge_adapter_weights=False,
            )

            # Apply postprocess + quantization (when targeting a quantized rollout,
            # e.g. FP8 sglang): base weights are quantized to match the rollout's
            # storage format so update_weights_from_tensor lands real weight + scale
            # pairs.
            if not materialize:
                # The export's internal TP collectives must still run on every rank.
                for _ in named_weights:
                    pass
                return

            named_weights = self._postprocess_and_quantize(named_weights, "base")
            # One unit per megatron param: quantize emits weight + scales
            # consecutively, so grouping by source name keeps them together.
            for _megatron_name, group in itertools.groupby(named_weights, key=lambda item: item[2]):
                unit = [(h, w) for h, w, _m in group if not is_lora_weight_name(h)]
                if unit:
                    yield unit
        yield from _iter_mm_tower_units(self.args, materialize=materialize)

    def _export_pp_local_lora(self, adapter, weights):
        if adapter is None:
            return self._export_current_adapter(weights)

        from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

        from ..multi_lora_utils import slice_lora_to_rank

        with expose_adapter_slot(self.model, adapter.slot):
            named_tensors = self._export_current_adapter()
        return [(h, slice_lora_to_rank(h, w, adapter.config.rank)) for h, w in named_tensors]

    def _export_current_adapter(self, weights=None) -> list:
        with megatron_bridge_utils.patch_megatron_model(self.model):
            renamed_weights = {strip_param_name_prefix(k): v for k, v in (weights or {}).items()}
            named_weights = _export_adapter_weights_from_local_weights(self._bridge, self.model, renamed_weights)
            named_weights = self._postprocess_and_quantize(named_weights, "lora")
            return [(h, w) for h, w, _m in named_weights if is_lora_weight_name(h)]

    def _postprocess_and_quantize(self, named_weights, weight_type: str):
        for hf_param_name, weight, megatron_param_name in named_weights:
            hf_name = hf_param_name.replace(".base_layer.", ".")
            weight = postprocess_hf_param(
                args=self.args,
                megatron_param_name=megatron_param_name,
                hf_param_name=hf_name,
                param=weight,
            )
            if weight_type == "base" and self.quantization_config is not None:
                # quantize_params expects the megatron name with the `module.module.`
                # prefix that the direct iterator uses; the bridge yields it without.
                qmegatron_name = f"module.module.{megatron_param_name}"
                for q_hf_name, q_weight in quantize_params(
                    self.args, qmegatron_name, [(hf_name, weight)], self.quantization_config
                ):
                    yield q_hf_name, q_weight, megatron_param_name
            else:
                yield hf_name, weight, megatron_param_name


def _load_quantized_param_basenames(hf_checkpoint):
    """Base names of params stored packed (`<base>.weight_packed`) in the checkpoint, or None if unknown."""
    index_path = os.path.join(hf_checkpoint, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return None
    with open(index_path) as f:
        names = json.load(f)["weight_map"]
    return {n.removesuffix(".weight_packed") for n in names if n.endswith(".weight_packed")}


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    return _MapWithLen(
        lambda task: _replace_task_weight(task, new_weight_dict),
        vanilla_conversion_tasks,
    )


def _replace_task_weight(task, weights, *, required=False, uploaded=None):
    if task is None or task.param_weight is None:
        return task

    key = f"vp_stages.{task.vp_stage}.{task.param_name}"
    if key not in weights:
        if required:
            raise KeyError(
                f"LoRA publish source is missing adapter tensor {key!r}; "
                "refusing to fall back to the released live parameter buffer"
            )
        # Buffer-like params (Gemma-4 layer_scalar/scale) aren't in optimizer state.
        return task

    if uploaded is None:
        weight = weights[key].cuda()
    else:
        if key not in uploaded:
            uploaded[key] = weights[key].cuda(non_blocking=True)
        weight = uploaded[key]
    return dataclasses.replace(task, param_weight=weight)


def _export_adapter_weights_from_local_weights(bridge, model, new_weight_dict):
    """Run Bridge's adapter exporter with weights supplied by ``weights_getter``.

    Megatron-Bridge's public adapter exporter always reads ``model`` directly,
    unlike its base exporter which accepts conversion tasks. For colocated
    remat-off training, however, the DDP parameter buffer has already been
    released and ``new_weight_dict`` is the committed pinned snapshot. Replace
    only the adapter tasks' tensor inputs and retain Bridge's complete fused,
    TP, PP, and EP conversion implementation.

    An empty mapping is the existing distributed/multi-LoRA contract and keeps
    using the live-model public API.
    """
    if not new_weight_dict:
        return bridge.export_adapter_weights(model, cpu=False, show_progress=False)

    model_bridge = getattr(bridge, "_model_bridge", None)
    if (
        model_bridge is None
        or not hasattr(model_bridge, "build_adapter_conversion_tasks")
        or not hasattr(model_bridge, "stream_adapter_weights_megatron_to_hf")
    ):
        raise RuntimeError(
            "The installed Megatron-Bridge cannot export LoRA weights from a supplied snapshot. "
            "MILES requires build_adapter_conversion_tasks() and "
            "stream_adapter_weights_megatron_to_hf() for colocated LoRA offload."
        )

    uploaded_weights = {}
    adapter_tasks = {
        base_name: [
            dataclasses.replace(
                task,
                linear_in_task=_replace_task_weight(
                    task.linear_in_task,
                    new_weight_dict,
                    required=True,
                    uploaded=uploaded_weights,
                ),
                linear_out_task=_replace_task_weight(
                    task.linear_out_task,
                    new_weight_dict,
                    required=True,
                    uploaded=uploaded_weights,
                ),
            )
            for task in tasks
        ]
        for base_name, tasks in model_bridge.build_adapter_conversion_tasks(model).items()
    }

    # Avoid mutating the bridge shared by later base/adapter exports. The pinned
    # fork's stream implementation asks self.build_adapter_conversion_tasks()
    # once, then performs all of its normal conversion and collective logic.
    snapshot_bridge = copy.copy(model_bridge)

    def _get_snapshot_tasks(_model):
        return adapter_tasks

    snapshot_bridge.build_adapter_conversion_tasks = _get_snapshot_tasks
    return snapshot_bridge.stream_adapter_weights_megatron_to_hf(model, cpu=False, show_progress=False)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
