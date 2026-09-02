import pytest
import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from miles.backends.megatron_utils.quantized_storage import TEQuantizedStorageAdapter


class FakeMXFP8Tensor:
    def __init__(self, shape=(3, 64), *, swizzled=False, fp8_dtype="DType.kFloat8E4M3"):
        self._shape = tuple(shape)
        self.dtype = torch.bfloat16
        self.device = torch.device("cpu")
        rows = 1
        for dim in shape[:-1]:
            rows *= dim
        cols = shape[-1]
        self._rowwise_data = torch.arange(rows * cols, dtype=torch.int64).to(torch.uint8).reshape(shape)
        self._rowwise_scale_inv = torch.arange(128 * 4, dtype=torch.int64).to(torch.uint8).reshape(128, 4)
        self._columnwise_data = None
        self._columnwise_scale_inv = None
        self._context = {
            "cls": type(self),
            "is_tensor": True,
            "requires_grad": False,
            "nontensor_kwargs": {
                "fp8_dtype": fp8_dtype,
                "with_gemm_swizzled_scales": swizzled,
                "fake_dtype": self.dtype,
            },
        }

    def size(self):
        return torch.Size(self._shape)

    def __tensor_flatten__(self):
        return ["_rowwise_data", "_rowwise_scale_inv"], self._context


def test_adapter_exports_compact_rowwise_bytes_without_quantizing():
    tensor = FakeMXFP8Tensor()
    adapter = TEQuantizedStorageAdapter.from_tensor(tensor)
    assert adapter is not None

    components = adapter.compact_publish_components(device="cpu")

    assert components.data.dtype == torch.float8_e4m3fn
    assert components.data.shape == (3, 64)
    assert torch.equal(components.data.view(torch.uint8), tensor._rowwise_data)
    assert components.scale_inv.shape == (3, 2)
    assert torch.equal(components.scale_inv, tensor._rowwise_scale_inv[:3, :2])
    assert adapter.publish_nbytes == 3 * 64 + 3 * 2


def test_adapter_restores_logical_leading_dimensions_for_scales():
    tensor = FakeMXFP8Tensor(shape=(2, 3, 64))
    adapter = TEQuantizedStorageAdapter.from_tensor(tensor)
    assert adapter is not None

    components = adapter.compact_publish_components(device="cpu")

    assert components.data.shape == (2, 3, 64)
    assert components.scale_inv.shape == (2, 3, 2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"swizzled": True}, "swizzled"),
        ({"fp8_dtype": "DType.kFloat8E5M2"}, "E4M3 only"),
    ],
)
def test_adapter_fails_closed_on_incompatible_descriptor(overrides, message):
    with pytest.raises(RuntimeError, match=message):
        TEQuantizedStorageAdapter.from_tensor(FakeMXFP8Tensor(**overrides))


def test_adapter_rejects_unexpected_scale_shape():
    tensor = FakeMXFP8Tensor()
    tensor._rowwise_scale_inv = torch.empty(3, 2, dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="unexpected TE MXFP8 rowwise scale metadata"):
        TEQuantizedStorageAdapter.from_tensor(tensor)


class LegacyMXFP8Tensor:
    pass


def test_adapter_requires_te_flatten_protocol_for_mxfp8():
    with pytest.raises(RuntimeError, match="__tensor_flatten__"):
        TEQuantizedStorageAdapter.from_tensor(LegacyMXFP8Tensor())
