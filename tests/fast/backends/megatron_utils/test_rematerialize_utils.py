from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.rematerialize_utils import (
    _build_cast_main_to_params_fn,
    _replay_hybrid_device_copy_back,
)


class _FakeDistOpt:
    def __init__(self, *, layer_wise=False):
        self.model_copies = 0
        self.buffer_copies = 0
        self._layer_wise_non_distopt_child = layer_wise

    def _copy_main_params_to_model_params(self):
        self.model_copies += 1

    def _copy_main_params_to_param_buffer(self):
        self.buffer_copies += 1


def test_mcore_cast_calls_every_chained_optimizer():
    dist_opts = [_FakeDistOpt(), _FakeDistOpt()]
    cast = _build_cast_main_to_params_fn(SimpleNamespace(chained_optimizers=dist_opts), precision_aware=False)
    cast()
    assert [opt.model_copies for opt in dist_opts] == [1, 1]
    assert [opt.buffer_copies for opt in dist_opts] == [0, 0]


def test_mxfp8_reuse_stages_distopt_and_direct_copies_layerwise_child():
    distopt = _FakeDistOpt()
    layer_wise = _FakeDistOpt(layer_wise=True)
    cast = _build_cast_main_to_params_fn(
        SimpleNamespace(chained_optimizers=[distopt, layer_wise]),
        precision_aware=False,
        reuse_grad_buf_for_mxfp8_param_ag=True,
    )

    cast()

    assert (distopt.buffer_copies, distopt.model_copies) == (1, 0)
    assert (layer_wise.buffer_copies, layer_wise.model_copies) == (0, 1)


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
