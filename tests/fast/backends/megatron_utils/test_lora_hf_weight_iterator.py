"""Unit tests for the megatron hf-weight-iterator factory: mode routing and
placement resolution against each implementation's forced placement."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


from argparse import Namespace
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement


@dataclass(frozen=True)
class _ConversionTask:
    param_name: str
    vp_stage: int
    param_weight: object


@dataclass(frozen=True)
class _AdapterTask:
    linear_in_task: _ConversionTask
    linear_out_task: _ConversionTask


class TestHfWeightIteratorFactory:
    def _make_args(self, mode="bridge"):
        return Namespace(
            megatron_to_hf_mode=mode,
            hf_checkpoint="/fake/path",
            update_weight_buffer_size=1,
        )

    def _create(self, mode, required_placement=None):
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator import get_hf_weight_iterator

        if required_placement is None:
            required_placement = WeightUpdatePlacement(gather_pp=True)
        return get_hf_weight_iterator(
            self._make_args(mode),
            [MagicMock()],
            required_placement=required_placement,
            model_name="qwen",
            quantization_config=None,
        )

    def test_bridge_mode_creates_bridge_iterator(self):
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge

        with patch.object(HfWeightIteratorBridge, "__init__", return_value=None):
            iterator = self._create("bridge")
            assert isinstance(iterator, HfWeightIteratorBridge)

    def test_raw_mode_creates_direct_iterator(self):
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

        with patch.object(HfWeightIteratorDirect, "__init__", return_value=None):
            iterator = self._create("raw")
            assert isinstance(iterator, HfWeightIteratorDirect)

    def test_invalid_mode_raises(self):
        with pytest.raises(KeyError):
            self._create("invalid_mode")

    def test_forced_placement_resolves_by_implementation(self):
        """Bridge forces a full gather; the direct iterator can keep PP local,
        so resolution joins the requirement with each implementation's floor."""
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

        full = WeightUpdatePlacement(gather_pp=True)
        keep_pp = WeightUpdatePlacement(gather_pp=False)
        assert HfWeightIteratorBridge.forced_placement == full
        assert HfWeightIteratorDirect.forced_placement == keep_pp

        captured = {}

        def _capture_init(self, args, model, *, placement, model_name, quantization_config):
            captured["placement"] = placement

        with patch.object(HfWeightIteratorBridge, "__init__", _capture_init):
            self._create("bridge", required_placement=keep_pp)
        assert captured["placement"] == full

        with patch.object(HfWeightIteratorDirect, "__init__", _capture_init):
            self._create("raw", required_placement=keep_pp)
        assert captured["placement"] == keep_pp


def _adapter_tasks(live_in, live_out):
    return {
        "decoder.layers.0.mlp.linear_fc1": [
            _AdapterTask(
                linear_in_task=_ConversionTask(
                    "decoder.layers.0.mlp.linear_fc1.adapter.linear_in.weight", 0, live_in
                ),
                linear_out_task=_ConversionTask(
                    "decoder.layers.0.mlp.linear_fc1.adapter.linear_out.weight", 0, live_out
                ),
            )
        ]
    }


def test_adapter_export_uses_weights_getter_snapshot_without_mutating_bridge():
    from miles.backends.megatron_utils.update_weight import hf_weight_iterator_bridge as iterator_module

    live_in, live_out = object(), object()
    pinned_in, pinned_out = MagicMock(), MagicMock()
    cuda_in, cuda_out = object(), object()
    pinned_in.cuda.return_value = cuda_in
    pinned_out.cuda.return_value = cuda_out
    tasks = _adapter_tasks(live_in, live_out)

    class _ModelBridge:
        def build_adapter_conversion_tasks(self, _model):
            return tasks

        def stream_adapter_weights_megatron_to_hf(self, model, *, cpu, show_progress):
            assert model == ["model"]
            assert cpu is False
            assert show_progress is False
            task = self.build_adapter_conversion_tasks(model)["decoder.layers.0.mlp.linear_fc1"][0]
            yield "linear_in", task.linear_in_task.param_weight
            yield "linear_out", task.linear_out_task.param_weight

    model_bridge = _ModelBridge()
    bridge = SimpleNamespace(_model_bridge=model_bridge)
    snapshot = {
        "vp_stages.0.decoder.layers.0.mlp.linear_fc1.adapter.linear_in.weight": pinned_in,
        "vp_stages.0.decoder.layers.0.mlp.linear_fc1.adapter.linear_out.weight": pinned_out,
    }

    result = list(iterator_module._export_adapter_weights_from_local_weights(bridge, ["model"], snapshot))

    assert result == [("linear_in", cuda_in), ("linear_out", cuda_out)]
    pinned_in.cuda.assert_called_once_with(non_blocking=True)
    pinned_out.cuda.assert_called_once_with(non_blocking=True)
    original_task = model_bridge.build_adapter_conversion_tasks(["model"])["decoder.layers.0.mlp.linear_fc1"][0]
    assert original_task.linear_in_task.param_weight is live_in
    assert original_task.linear_out_task.param_weight is live_out


def test_adapter_snapshot_export_fails_instead_of_reading_released_live_parameter():
    from miles.backends.megatron_utils.update_weight import hf_weight_iterator_bridge as iterator_module

    model_bridge = SimpleNamespace(
        build_adapter_conversion_tasks=lambda _model: _adapter_tasks(object(), object()),
        stream_adapter_weights_megatron_to_hf=MagicMock(),
    )
    bridge = SimpleNamespace(_model_bridge=model_bridge)

    with pytest.raises(KeyError, match="refusing to fall back to the released live parameter buffer"):
        iterator_module._export_adapter_weights_from_local_weights(
            bridge,
            ["model"],
            {"vp_stages.0.some_other_adapter.weight": object()},
        )


def test_empty_adapter_weight_mapping_preserves_live_export_contract():
    from miles.backends.megatron_utils.update_weight import hf_weight_iterator_bridge as iterator_module

    expected = [("adapter", object())]
    bridge = SimpleNamespace(export_adapter_weights=MagicMock(return_value=iter(expected)))

    result = list(iterator_module._export_adapter_weights_from_local_weights(bridge, ["model"], {}))

    assert result == expected
    bridge.export_adapter_weights.assert_called_once_with(["model"], cpu=False, show_progress=False)


def test_direct_adapter_export_uses_weights_getter_snapshot():
    from miles.backends.megatron_utils.update_weight import hf_weight_iterator_direct as iterator_module
    from miles_plugins.models.inkling import lora as inkling_lora

    adapter_name = "module.module.decoder.layers.0.self_attention.lora_adapter.wq_A"
    live_adapter = object()
    pinned_adapter = MagicMock()
    cuda_adapter = object()
    pinned_adapter.cuda.return_value = cuda_adapter
    iterator = SimpleNamespace(args=Namespace(), model=["model"])

    def _export(model, *, parameter_getter):
        assert model == ["model"]
        return [("adapter", parameter_getter(live_adapter))]

    with (
        patch.object(
            iterator_module,
            "named_params_and_buffers",
            return_value=iter([(adapter_name, live_adapter), ("module.module.base.weight", object())]),
        ),
        patch.object(inkling_lora, "export_inkling_lora_hf_named", side_effect=_export),
    ):
        result = iterator_module.HfWeightIteratorDirect._export_pp_local_lora(
            iterator, None, {adapter_name: pinned_adapter}
        )

    assert result == [("adapter", cuda_adapter)]
    pinned_adapter.cuda.assert_called_once_with(non_blocking=True)


def test_direct_adapter_snapshot_export_fails_before_reading_live_parameter():
    from miles.backends.megatron_utils.update_weight import hf_weight_iterator_direct as iterator_module
    from miles_plugins.models.inkling import lora as inkling_lora

    adapter_name = "module.module.decoder.layers.0.self_attention.lora_adapter.wq_A"
    iterator = SimpleNamespace(args=Namespace(), model=["model"])
    exporter = MagicMock()
    with (
        patch.object(
            iterator_module,
            "named_params_and_buffers",
            return_value=iter([(adapter_name, object())]),
        ),
        patch.object(inkling_lora, "export_inkling_lora_hf_named", exporter),
        pytest.raises(KeyError, match="refusing to fall back to the released live parameter buffer"),
    ):
        iterator_module.HfWeightIteratorDirect._export_pp_local_lora(iterator, None, {"other": object()})

    exporter.assert_not_called()


def test_direct_empty_weight_mapping_preserves_live_export_contract():
    from miles.backends.megatron_utils.update_weight import hf_weight_iterator_direct as iterator_module
    from miles_plugins.models.inkling import lora as inkling_lora

    expected = [("adapter", object())]
    iterator = SimpleNamespace(args=Namespace(), model=["model"])
    with patch.object(inkling_lora, "export_inkling_lora_hf_named", return_value=expected) as exporter:
        result = iterator_module.HfWeightIteratorDirect._export_pp_local_lora(iterator, None, {})

    assert result == expected
    exporter.assert_called_once_with(["model"], parameter_getter=None)


def test_inkling_exporter_applies_parameter_getter_before_layout_conversion():
    from miles_plugins.models.inkling.lora import InklingLoRAAdapter, export_inkling_lora_hf_named

    adapter = InklingLoRAAdapter("dense_mlp", "model.layers.0.mlp.")
    adapter.fc1_A = torch.nn.Parameter(torch.full((1, 2), 1.0))
    adapter.fc1_B = torch.nn.Parameter(torch.full((2, 1), 2.0))
    adapter.fc2_A = torch.nn.Parameter(torch.full((1, 2), 3.0))
    adapter.fc2_B = torch.nn.Parameter(torch.full((2, 1), 4.0))
    adapter.load_meta = {"i_loc": 1}
    model = torch.nn.Module()
    model.adapter = adapter
    replacements = {
        id(adapter.fc1_A): torch.full_like(adapter.fc1_A, 11.0),
        id(adapter.fc1_B): torch.tensor([[12.0], [13.0]]),
        id(adapter.fc2_A): torch.full_like(adapter.fc2_A, 14.0),
        id(adapter.fc2_B): torch.full_like(adapter.fc2_B, 15.0),
    }

    with (
        patch("megatron.core.parallel_state.get_tensor_model_parallel_world_size", return_value=1),
        patch("torch.distributed.get_rank", return_value=1),
    ):
        exported = dict(
            export_inkling_lora_hf_named([model], parameter_getter=lambda param: replacements[id(param)])
        )

    assert torch.equal(
        exported["model.layers.0.mlp.gate_up_proj.lora_A.weight"], replacements[id(adapter.fc1_A)].bfloat16()
    )
    assert torch.equal(
        exported["model.layers.0.mlp.gate_up_proj.lora_B.weight"], replacements[id(adapter.fc1_B)].bfloat16()
    )
    assert torch.equal(
        exported["model.layers.0.mlp.down_proj.lora_A.weight"], replacements[id(adapter.fc2_A)].bfloat16()
    )
    assert torch.equal(
        exported["model.layers.0.mlp.down_proj.lora_B.weight"], replacements[id(adapter.fc2_B)].bfloat16()
    )
