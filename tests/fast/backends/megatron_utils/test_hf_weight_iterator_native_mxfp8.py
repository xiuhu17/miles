from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from miles.backends.megatron_utils.quantized_storage import MXFP8PublishComponents
from miles.backends.megatron_utils.update_weight.hf_weight_iterator_native_mxfp8 import HfWeightIteratorNativeMXFP8


def _args(**overrides):
    values = {
        "extra_high_precision_layers_megatron": False,
        "first_last_layers_bf16": False,
        "hidden_size": 128,
        "indexer_rope_interleave": False,
        "kv_channels": 128,
        "num_attention_heads": 1,
        "num_query_groups": 1,
        "vocab_size": 32000,
    }
    values.update(overrides)
    return Namespace(**values)


def _iterator():
    iterator = object.__new__(HfWeightIteratorNativeMXFP8)
    iterator.args = _args()
    iterator.model_name = "glmmoedsa"
    return iterator


def test_native_expert_conversion_keeps_data_and_scale_in_lockstep():
    name = "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight0"
    data_bytes = torch.arange(4 * 64, dtype=torch.int64).to(torch.uint8).reshape(4, 64)
    scale = torch.arange(4 * 2, dtype=torch.uint8).reshape(4, 2)
    payload = MXFP8PublishComponents(
        data=data_bytes.view(torch.float8_e4m3fn),
        scale_inv=scale,
    )
    iterator = _iterator()

    outputs = iterator._convert_to_hf_named_tensors([payload], [SimpleNamespace(name=name)])

    assert [output_name for output_name, _tensor in outputs] == [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight_scale_inv",
    ]
    assert torch.equal(outputs[0][1].view(torch.uint8), data_bytes[:2])
    assert torch.equal(outputs[1][1], scale[:2])
    assert torch.equal(outputs[2][1].view(torch.uint8), data_bytes[2:])
    assert torch.equal(outputs[3][1], scale[2:])


def test_native_target_never_falls_back_to_bf16_requantization():
    name = "module.module.decoder.layers.0.self_attention.linear_proj.weight"
    iterator = _iterator()

    with pytest.raises(RuntimeError, match="refusing BF16 re-quantization fallback"):
        iterator._convert_to_hf_named_tensors(
            [torch.ones(4, 64, dtype=torch.bfloat16)],
            [SimpleNamespace(name=name)],
        )


def test_high_precision_exception_is_dequantized_only_for_publish():
    name = "module.module.decoder.layers.0.self_attention.weights_proj.weight"
    value = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    iterator = _iterator()

    outputs = iterator._convert_to_hf_named_tensors([value], [SimpleNamespace(name=name)])

    assert len(outputs) == 1
    assert outputs[0][0] == "model.layers.0.self_attn.indexer.weights_proj.weight"
    assert outputs[0][1] is value


def test_interleaved_wk_materializes_through_dequantized_exception():
    iterator = _iterator()
    iterator.args = _args(indexer_rope_interleave=True)
    name = "module.module.decoder.layers.0.self_attention.wk.weight"
    expected = torch.ones(4, 64, dtype=torch.bfloat16)
    adapter = SimpleNamespace(
        schema_digest="digest",
        dequantize=lambda **_kwargs: expected,
    )
    info = SimpleNamespace(
        name=name,
        dtype=torch.bfloat16,
        attrs={"native_publish_storage_schema": "digest"},
    )

    module = "miles.backends.megatron_utils.update_weight.hf_weight_iterator_native_mxfp8"
    with patch(f"{module}.TEQuantizedStorageAdapter.from_tensor", return_value=adapter):
        value = iterator._materialize_local_value(info, object(), torch.device("cpu"))

    assert value is expected


def test_storage_probe_records_adapter_error_instead_of_raising_before_collectives():
    iterator = _iterator()
    module = "miles.backends.megatron_utils.update_weight.hf_weight_iterator_native_mxfp8"
    with (
        patch(f"{module}.native_format_name", return_value="MXFP8"),
        patch(f"{module}.TEQuantizedStorageAdapter.from_tensor", side_effect=RuntimeError("bad schema")),
    ):
        attrs = iterator._extra_param_info_attrs("weight", torch.empty(1))

    assert attrs["native_publish_storage_schema"] is None
    assert attrs["native_publish_storage_format"] == "MXFP8"
    assert attrs["native_publish_storage_error"] == "bad schema"


def test_native_publish_model_converter_allowlist_is_component_safe():
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator_native_mxfp8 import (
        _has_component_safe_converter,
    )

    assert _has_component_safe_converter("GlmMoeDsaConfig")
    assert _has_component_safe_converter("DeepSeekV4Config")
    assert not _has_component_safe_converter("Qwen3Config")


def test_ordinary_values_keep_atomic_sibling_bucket():
    iterator = _iterator()
    iterator.model_name = "deepseekv4"
    prefix = "module.module.decoder.layers.0.self_attention_hyper_connection"
    infos = [SimpleNamespace(name=f"{prefix}.alpha_{suffix}") for suffix in ("pre", "post", "res")]
    values = [torch.tensor(value, dtype=torch.float32) for value in (1, 2, 3)]

    outputs = iterator._convert_to_hf_named_tensors(values, infos)

    assert len(outputs) == 1
    assert outputs[0][0] == "model.layers.0.hc_attn_scale"
    assert torch.equal(outputs[0][1], torch.tensor([1, 2, 3], dtype=torch.float32))


def test_native_payload_uses_shared_component_hooks():
    iterator = _iterator()
    data = torch.empty(4, 64, dtype=torch.float8_e4m3fn)
    scale = torch.empty(4, 2, dtype=torch.uint8)
    payload = MXFP8PublishComponents(data=data, scale_inv=scale)

    assert iterator._value_components(payload) == (data, scale)

    new_data = torch.zeros_like(data)
    new_scale = torch.zeros_like(scale)
    rebuilt = iterator._rebuild_value(payload, (new_data, new_scale))
    assert rebuilt.data is new_data
    assert rebuilt.scale_inv is new_scale


def test_native_remote_allocation_derives_compact_scale_from_logical_shape():
    iterator = _iterator()
    info = SimpleNamespace(
        name="module.module.decoder.layers.0.self_attention.linear_proj.weight",
        shape=torch.Size((4, 64)),
        dtype=torch.bfloat16,
    )

    payload = iterator._allocate_remote_value(info, torch.device("cpu"))

    assert payload.data.shape == (4, 64)
    assert payload.data.dtype == torch.float8_e4m3fn
    assert payload.scale_inv.shape == (4, 2)
    assert payload.scale_inv.dtype == torch.uint8


def test_shared_tp_gather_rebuilds_native_data_and_scale_together():
    iterator = _iterator()
    payload = MXFP8PublishComponents(
        data=torch.zeros(2, 64, dtype=torch.float8_e4m3fn),
        scale_inv=torch.zeros(2, 2, dtype=torch.uint8),
    )
    info = SimpleNamespace(
        name="module.module.decoder.layers.0.self_attention.linear_proj.weight",
        shape=torch.Size((2, 64)),
        attrs={
            "tensor_model_parallel": True,
            "partition_dim": 0,
            "partition_stride": 1,
            "parallel_mode": None,
        },
    )
    parallel = SimpleNamespace(size=2, group=object())
    parallel_state = SimpleNamespace(tp=parallel, etp=parallel)

    def _all_gather(partitions, component, **_kwargs):
        for partition in partitions:
            partition.copy_(component)
        return SimpleNamespace(wait=lambda: None)

    module = "miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct"
    with (
        patch(f"{module}.get_parallel_state", return_value=parallel_state),
        patch(f"{module}.dist.all_gather", side_effect=_all_gather),
    ):
        values = iterator._all_gather_value_components_async(
            [info],
            [payload],
        )

    assert len(values) == 1
    assert values[0].data.shape == (4, 64)
    assert values[0].scale_inv.shape == (4, 2)
