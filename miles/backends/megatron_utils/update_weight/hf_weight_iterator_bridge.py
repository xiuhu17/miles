import dataclasses
import re

from miles.backends.megatron_utils.lora_utils import is_lora_weight_name
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils import megatron_bridge_utils
from miles.utils.iter_utils import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..megatron_to_hf.processors import quantize_params
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)

    def get_hf_weight_chunks(self, megatron_local_weights, weight_type: str = "base"):
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            if weight_type == "lora":
                named_weights = self._bridge.export_adapter_weights(
                    self.model,
                    cpu=False,
                    show_progress=False,
                )
            elif weight_type == "base":
                conversion_tasks = self._bridge.get_conversion_tasks(self.model)
                conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)
                named_weights = self._bridge.export_hf_weights(
                    self.model,
                    cpu=False,
                    conversion_tasks=conversion_tasks,
                    merge_adapter_weights=False,
                )

            # Apply postprocess + quantization (when targeting a quantized rollout,
            # e.g. FP8 sglang). Base weights are quantized to match the rollout's
            # storage format so update_weights_from_tensor lands real weight + scale
            # pairs; LoRA adapters are passed through unchanged.
            named_weights = self._postprocess_and_quantize(named_weights, weight_type)

            if weight_type == "base":
                named_weights = ((n, t) for n, t in named_weights if not is_lora_weight_name(n))
            elif weight_type == "lora":
                named_weights = ((n, t) for n, t in named_weights if is_lora_weight_name(n))

            yield from chunk_named_params_by_size(named_weights, chunk_size=self.args.update_weight_buffer_size)

    def _postprocess_and_quantize(self, named_weights, weight_type: str):
        for hf_param_name, weight, megatron_param_name in named_weights:
            hf_name = hf_param_name.replace(".base_layer.", ".")
            hf_name = _maybe_globalize_expert_hf_name(
                self.args,
                hf_name=hf_name,
                megatron_param_name=megatron_param_name,
            )
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
                yield from quantize_params(self.args, qmegatron_name, [(hf_name, weight)], self.quantization_config)
            else:
                yield hf_name, weight


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    def _handle_one(task):
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert (
            weight_dict_key in new_weight_dict
        ), f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"

        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()
        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)


_GROUPED_EXPERT_RE = re.compile(r"(?:^|\.)mlp\.experts\.linear_fc[12]\.weight(?P<expert>\d+)$")
_LOCAL_EXPERT_RE = re.compile(r"(?:^|\.)mlp\.experts\.local_experts\.(?P<expert>\d+)\.linear_fc[12]\.weight$")
_HF_EXPERT_RE = re.compile(r"(?P<prefix>\.mlp\.experts\.)(?P<expert>\d+)(?P<suffix>\.)")


def _maybe_globalize_expert_hf_name(args, hf_name: str, megatron_param_name: str) -> str:
    """Convert Bridge-exported local expert ids to global HF expert ids.

    In bridge mode the actor's weight backup keeps Megatron's local EP names so
    ``AutoBridge.get_conversion_tasks`` can find the current local tensors. Some
    bridge mappings also export those local expert ids in the HF name. SGLang's
    loader, however, treats HF ``mlp.experts.{id}`` as global expert ids and maps
    them back to local ids with its own EP rank. Offset only when the HF id still
    matches the local Megatron id; if a bridge already emitted global ids, leave it
    untouched.
    """
    local_expert = _extract_local_expert_id(megatron_param_name)
    if local_expert is None:
        return hf_name

    try:
        ep = get_parallel_state().ep
        ep_rank = int(ep.rank)
        ep_size = int(ep.size)
    except Exception:
        return hf_name

    if ep_size <= 1:
        return hf_name

    num_experts = getattr(args, "num_experts", None)
    if not num_experts:
        return hf_name

    expert_offset = ep_rank * int(num_experts) // ep_size
    if expert_offset == 0:
        return hf_name

    match = _HF_EXPERT_RE.search(hf_name)
    if match is None:
        return hf_name

    hf_expert = int(match.group("expert"))
    if hf_expert != local_expert:
        return hf_name

    global_expert = hf_expert + expert_offset
    return (
        hf_name[: match.start("expert")]
        + str(global_expert)
        + hf_name[match.end("expert") :]
    )


def _extract_local_expert_id(megatron_param_name: str) -> int | None:
    for pattern in (_GROUPED_EXPERT_RE, _LOCAL_EXPERT_RE):
        match = pattern.search(megatron_param_name)
        if match is not None:
            return int(match.group("expert"))
    return None
