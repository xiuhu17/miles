from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.bridge_lora_helpers import (
    _clear_frozen_high_precision_init_values,
    _freeze_lora_base_persistent_state,
)
from miles.backends.megatron_utils.model_provider import _apply_bridge_fp8_runtime_config
from miles.backends.megatron_utils.update_weight.common import weight_update_format
from miles.utils.arguments import _validate_native_mxfp8_param_gather


def _fp8_args(**overrides):
    values = {
        "fp8": "e4m3",
        "fp8_recipe": "mxfp8",
        "fp8_param_gather": True,
        "fp8_wgrad": True,
        "fp8_output_proj": False,
        "tp_only_amax_red": False,
        "first_last_layers_bf16": False,
        "num_layers_at_start_in_bf16": 1,
        "num_layers_at_end_in_bf16": 1,
    }
    values.update(overrides)
    return Namespace(**values)


def _fp8_provider(**overrides):
    values = {
        "fp8": None,
        "fp8_recipe": "delayed",
        "fp8_param": False,
        "fp8_wgrad": False,
        "fp8_output_proj": False,
        "tp_only_amax_red": True,
        "first_last_layers_bf16": True,
        "num_layers_at_start_in_bf16": 0,
        "num_layers_at_end_in_bf16": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_bridge_fp8_runtime_config_enables_mxfp8_primary_storage():
    provider = _fp8_provider()

    _apply_bridge_fp8_runtime_config(provider, _fp8_args())

    assert provider.fp8 == "e4m3"
    assert provider.fp8_recipe == "mxfp8"
    assert provider.fp8_param is True
    assert provider.fp8_wgrad is True
    assert provider.tp_only_amax_red is False
    assert provider.first_last_layers_bf16 is False
    assert provider.num_layers_at_start_in_bf16 == 1
    assert provider.num_layers_at_end_in_bf16 == 1


def test_bridge_fp8_runtime_config_fails_closed_when_provider_is_too_old():
    provider = SimpleNamespace(fp8=None, fp8_recipe="delayed")

    with pytest.raises(RuntimeError, match="does not expose fp8_param"):
        _apply_bridge_fp8_runtime_config(provider, _fp8_args())


def _native_publish_args(**overrides):
    values = {
        "fp8_param_gather": True,
        "fp8_recipe": "mxfp8",
        "fp4_param_gather": False,
        "fp4_param": False,
        "colocate": True,
        "lora_rank": 0,
        "lora_adapter_path": None,
        "megatron_to_hf_mode": "raw",
        "optimizer_cpu_offload": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_native_mxfp8_capability_accepts_full_ft_raw_and_lora_bridge():
    _validate_native_mxfp8_param_gather(_native_publish_args())
    _validate_native_mxfp8_param_gather(_native_publish_args(lora_rank=8, megatron_to_hf_mode="bridge"))


def test_native_mxfp8_weight_update_format_applies_only_to_full_ft_base():
    assert weight_update_format(_native_publish_args()) == "native_mxfp8"
    assert weight_update_format(_native_publish_args(lora_rank=8)) == "default"
    assert weight_update_format(_native_publish_args(fp8_param_gather=False)) == "default"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"fp8_recipe": "delayed"}, "mxfp8 only"),
        ({"fp4_param_gather": True}, "NVFP4/FP4"),
        ({"colocate": False}, "requires --colocate"),
        ({"megatron_to_hf_mode": "bridge"}, "requires --megatron-to-hf-mode raw"),
        ({"optimizer_cpu_offload": True}, "HDO CPU optimizer"),
    ],
)
def test_native_mxfp8_capability_rejects_unimplemented_paths(overrides, message):
    with pytest.raises(ValueError, match=message):
        _validate_native_mxfp8_param_gather(_native_publish_args(**overrides))


class _FakeParam:
    def __init__(self, *, requires_grad: bool, init_value: torch.Tensor | None):
        self.requires_grad = requires_grad
        self._init_value = init_value
        self.clear_calls = 0

    def get_high_precision_init_val(self):
        return self._init_value

    def clear_high_precision_init_val(self):
        self._init_value = None
        self.clear_calls += 1


class _FakeModel:
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return iter(self._params)

    def modules(self):
        return iter(())


def test_lora_freeze_clears_only_frozen_high_precision_init_backups():
    frozen = _FakeParam(requires_grad=False, init_value=torch.empty(8, dtype=torch.bfloat16))
    trainable = _FakeParam(requires_grad=True, init_value=torch.empty(4, dtype=torch.bfloat16))
    no_backup = _FakeParam(requires_grad=False, init_value=None)

    count, num_bytes = _clear_frozen_high_precision_init_values(
        [_FakeModel([frozen, trainable, no_backup]), _FakeModel([frozen])]
    )

    assert count == 1
    assert num_bytes == 8 * torch.empty((), dtype=torch.bfloat16).element_size()
    assert frozen.clear_calls == 1
    assert frozen.get_high_precision_init_val() is None
    assert trainable.clear_calls == 0
    assert trainable.get_high_precision_init_val() is not None


def test_lora_freeze_also_freezes_mutable_router_base_state():
    router = SimpleNamespace(
        expert_bias=torch.zeros(4, dtype=torch.float32),
        frozen_expert_bias=False,
    )
    model = _FakeModel([])
    model.modules = lambda: iter((model, router))

    count = _freeze_lora_base_persistent_state([model])

    assert count == 1
    assert router.frozen_expert_bias is True
