from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.rematerialize_utils import (
    _build_cast_main_to_params_fn,
    _named_restore_extras,
    _replay_hybrid_device_copy_back,
)


class _FakeDistOpt:
    def __init__(self):
        self.copied = 0

    def _copy_main_params_to_model_params(self):
        self.copied += 1


def test_mcore_cast_calls_every_chained_optimizer():
    dist_opts = [_FakeDistOpt(), _FakeDistOpt()]
    cast = _build_cast_main_to_params_fn(SimpleNamespace(chained_optimizers=dist_opts), precision_aware=False)
    cast()
    assert [opt.copied for opt in dist_opts] == [1, 1]


def test_hdo_replay_covers_both_fractions():
    cpu_view = torch.zeros(4, dtype=torch.bfloat16)
    cpu_master = torch.randn(4)
    gpu_view = torch.zeros(4, dtype=torch.bfloat16)
    gpu_master = torch.randn(4)
    hdo = SimpleNamespace(
        gpu_params_map_cpu_copy={cpu_view: cpu_master},
        param_to_fp32_param={gpu_view: gpu_master},
    )
    _replay_hybrid_device_copy_back(hdo)
    assert torch.equal(cpu_view, cpu_master.to(torch.bfloat16))
    assert torch.equal(gpu_view, gpu_master.to(torch.bfloat16))


def test_hdo_replay_cpu_hook_takes_precedence_on_overlap():
    # A CPU-fraction param is in both maps; the GPU pass must skip it, like the hooks do.
    view = torch.zeros(4, dtype=torch.bfloat16)
    cpu_master = torch.full((4,), 2.0)
    stale = torch.full((4,), 3.0)
    hdo = SimpleNamespace(
        gpu_params_map_cpu_copy={view: cpu_master},
        param_to_fp32_param={view: stale},
    )
    _replay_hybrid_device_copy_back(hdo)
    assert torch.equal(view, cpu_master.to(torch.bfloat16))


def test_builder_rejects_non_hdo_inner_under_precision_aware():
    pytest.importorskip("megatron.core")
    optimizer = SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=object())])
    with pytest.raises(AssertionError):
        _build_cast_main_to_params_fn(optimizer, precision_aware=True)


def test_lora_rematerialize_only_pins_frozen_base_not_bf16_adapter():
    model = torch.nn.Module()
    model.register_parameter(
        "frozen_base",
        torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=False),
    )
    model.register_parameter(
        "adapter",
        torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=True),
    )

    extras = dict(_named_restore_extras([model]))

    assert "vp_stages.0.frozen_base" in extras
    assert "vp_stages.0.adapter" not in extras


def test_adapter_only_rematerialize_leaves_frozen_base_to_tms():
    model = torch.nn.Module()
    model.register_parameter(
        "frozen_base",
        torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=False),
    )
    model.register_parameter(
        "lora_adapter",
        torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=True),
    )

    extras = dict(
        _named_restore_extras(
            [model],
            parameter_filter=lambda name, _tensor: "lora_adapter" in name,
        )
    )

    assert extras == {}
