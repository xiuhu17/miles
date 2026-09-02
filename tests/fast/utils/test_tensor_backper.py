from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from miles.utils.native_param_storage import native_components
from miles.utils.tensor_backper import (
    MainCastContext,
    TensorBackuper,
    _allocate_pinned_like,
    _copy_tensor_value,
    _hash_tensor_sha256,
    _TensorBackuperMainCast,
)


@pytest.fixture(autouse=True)
def _cpu_fallback(monkeypatch):
    if torch.cuda.is_available():
        yield
        return
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    real_empty_like = torch.empty_like

    def empty_like_no_pin(tensor, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_empty_like(tensor, **kwargs)

    monkeypatch.setattr(torch, "empty_like", empty_like_no_pin)
    yield


class _FakeOptimizer:
    """Like _copy_main_params_to_model_params: writes only this rank's owned shard."""

    def __init__(self, mains, staging, shards):
        self._mains = mains
        self._staging = staging
        self._shards = shards

    def _copy_main_params_to_model_params(self):
        for name, main in self._mains.items():
            self._staging[name][self._shards[name]] = main.to(torch.bfloat16)


class _FakeModelChunk:
    """start_param_sync = param all-gather: staging buffers -> live params."""

    def __init__(self, params, staging):
        self._params = params
        self._staging = staging

    def start_param_sync(self, force_sync=False):
        assert force_sync
        for name, param in self._params.items():
            param.copy_(self._staging[name])


class _FakeQuantized:
    """Minimal flatten-protocol object; dequantization must never be called."""

    def __init__(self, data: torch.Tensor, scale: torch.Tensor):
        self._rowwise_data = data
        self._rowwise_scale_inv = scale

    def __tensor_flatten__(self):
        return ["_rowwise_data", "_rowwise_scale_inv"], {}

    def dequantize(self):
        raise AssertionError("native backup must not dequantize")


class _LegacyMXFP8Tensor:
    """Old image-only API shape; production must fail rather than guess its ABI."""

    def __init__(self):
        self._rowwise_data = torch.ones(4, dtype=torch.uint8)
        self._rowwise_scale_inv = torch.ones(1, dtype=torch.uint8)


class _FakeGroupedTensor:
    """Current TE GroupedTensor has component fields but no flatten protocol."""

    def __init__(
        self,
        shape,
        dtype,
        *,
        num_tensors,
        shapes=None,
        quantizer=None,
        data=None,
        columnwise_data=None,
        scale_inv=None,
        columnwise_scale_inv=None,
        amax=None,
        columnwise_amax=None,
        scale=None,
        first_dims=None,
        last_dims=None,
        tensor_offsets=None,
        offsets=None,
        scale_inv_offsets=None,
        columnwise_scale_inv_offsets=None,
        requires_grad=False,
        stride=None,
        with_gemm_swizzled_scales=False,
        row_scaled_nvfp4=False,
        nvfp4_use_4over6=False,
        nvfp4_e4m3_max=448,
    ):
        del requires_grad
        self.logical_shape = shape
        self.fake_dtype = dtype
        self.num_tensors = num_tensors
        self.tensor_shapes = shapes
        self.quantizer = quantizer
        self.rowwise_data = data
        self.columnwise_data = columnwise_data
        self.scale_inv = scale_inv
        self.columnwise_scale_inv = columnwise_scale_inv
        self.amax = amax
        self.columnwise_amax = columnwise_amax
        self.scale = scale
        self.first_dims = first_dims
        self.last_dims = last_dims
        self.tensor_offsets = tensor_offsets
        self.offsets = offsets
        self.scale_inv_offsets = scale_inv_offsets
        self.columnwise_scale_inv_offsets = columnwise_scale_inv_offsets
        self._stride = stride or (shape[-1], 1)
        self._with_gemm_swizzled_scales = with_gemm_swizzled_scales
        self.row_scaled_nvfp4 = row_scaled_nvfp4
        self.nvfp4_use_4over6 = nvfp4_use_4over6
        self.nvfp4_e4m3_max = nvfp4_e4m3_max

    def stride(self):
        return self._stride

    def dequantize(self):
        raise AssertionError("native grouped backup must not dequantize")


class _Setup:
    def __init__(self, num_params=3, numel=16, num_extras=1, check=False, shards=None):
        generator = torch.Generator().manual_seed(0)
        self.mains = {f"p{i}": torch.randn(numel, generator=generator) for i in range(num_params)}
        self.params = {name: main.to(torch.bfloat16) for name, main in self.mains.items()}
        self.staging = {name: param.clone() for name, param in self.params.items()}
        self.shards = shards or {name: slice(None) for name in self.mains}
        self.extras = {f"extra{i}": torch.randn(4, generator=generator) for i in range(num_extras)}
        owned = {name: main[self.shards[name]] for name, main in self.mains.items()}
        self.optimizer = SimpleNamespace(chained_optimizers=[_FakeOptimizer(owned, self.staging, self.shards)])
        self.model_chunk = _FakeModelChunk(self.params, self.staging)
        ctx = MainCastContext(
            cast_main_to_params=lambda: [
                opt._copy_main_params_to_model_params() for opt in self.optimizer.chained_optimizers
            ],
            model_chunks=[self.model_chunk],
            extras_getter=lambda: iter(self.extras.items()),
            rematerializable_ids={id(t) for t in self.params.values()},
            check=check,
        )
        self.backuper = TensorBackuper.create(
            source_getter=lambda: iter({**self.params, **self.extras}.items()),
            main_cast_ctx=ctx,
        )

    def corrupt_live_tensors(self):
        for param in self.params.values():
            param.fill_(float("nan"))
        for extra in self.extras.values():
            extra.fill_(float("nan"))


def test_create_returns_main_cast_variant():
    assert isinstance(_Setup().backuper, _TensorBackuperMainCast)


def test_round_trip_restores_bit_identical_weights():
    setup = _Setup()
    expected = {name: t.clone() for name, t in {**setup.params, **setup.extras}.items()}
    setup.backuper.backup("actor")
    setup.corrupt_live_tensors()
    setup.backuper.restore("actor")
    for name, tensor in {**setup.params, **setup.extras}.items():
        assert torch.equal(tensor, expected[name]), name


def test_restore_only_covers_owned_shard_and_relies_on_param_sync():
    numel = 16
    shards = {f"p{i}": slice(0, numel // 2) for i in range(3)}
    setup = _Setup(numel=numel, shards=shards)
    expected = {name: param.clone() for name, param in setup.params.items()}
    setup.backuper.backup("actor")
    setup.corrupt_live_tensors()
    # The unowned staging halves stand in for the other DP rank's cast.
    setup.backuper.restore("actor")
    for name, param in setup.params.items():
        assert torch.equal(param, expected[name]), name


def test_chained_optimizer_casts_every_inner_optimizer():
    setup = _Setup()
    names = list(setup.mains)
    inner = [_FakeOptimizer({n: setup.mains[n]}, setup.staging, setup.shards) for n in names]
    setup.optimizer.chained_optimizers = inner  # the ctx closure reads through
    expected = {name: param.clone() for name, param in setup.params.items()}
    setup.backuper.backup("actor")
    setup.corrupt_live_tensors()
    setup.backuper.restore("actor")
    for name, param in setup.params.items():
        assert torch.equal(param, expected[name]), name


def test_get_returns_pinned_backup_for_extras_and_live_tensors_for_params():
    setup = _Setup()
    setup.backuper.backup("actor")
    got = setup.backuper.get("actor")
    for name, param in setup.params.items():
        assert got[name].data_ptr() == param.data_ptr(), name
    for name, extra in setup.extras.items():
        # Extras are paused during update_weights, so get() must hand out the backup.
        assert got[name].data_ptr() != extra.data_ptr(), name
        assert torch.equal(got[name], extra), name


def test_main_cast_matches_pinned_extras_by_tensor_identity_across_namespaces():
    setup = _Setup()
    setup.backuper.backup("actor")
    extra_name, extra = next(iter(setup.extras.items()))
    setup.backuper._source_getter = lambda: iter([*setup.params.items(), (f"global.{extra_name}", extra)])

    got = setup.backuper.get("actor")

    assert got[f"global.{extra_name}"].data_ptr() != extra.data_ptr()
    assert torch.equal(got[f"global.{extra_name}"], extra)
    assert setup.backuper.pinned_backup_names("actor") == {f"global.{extra_name}"}


def test_check_verifies_first_cycles_and_raises_on_corruption():
    setup = _Setup(check=True)
    setup.backuper.backup("actor")
    setup.mains["p0"][0] += 1.0
    with pytest.raises(RuntimeError, match="not bit-identical"):
        setup.backuper.restore("actor")


def test_check_stops_after_check_num_cycles():
    setup = _Setup(check=True)
    for _ in range(setup.backuper._check_num_cycles):
        setup.backuper.backup("actor")
        setup.backuper.restore("actor")
    setup.backuper.backup("actor")
    assert setup.backuper._expected_hashes is None
    setup.mains["p0"][0] += 1.0
    setup.backuper.restore("actor")


def test_no_check_computes_no_hashes():
    setup = _Setup(check=False)
    setup.backuper.backup("actor")
    assert setup.backuper._expected_hashes is None
    setup.mains["p0"][0] += 1.0
    setup.backuper.restore("actor")


def test_check_detects_tensor_set_change():
    setup = _Setup(check=True)
    setup.backuper.backup("actor")
    setup.params["p_new"] = torch.zeros(4, dtype=torch.bfloat16)
    setup.staging["p_new"] = torch.zeros(4, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="changed the tensor set"):
        setup.backuper.restore("actor")


def test_get_rejects_unknown_tensor_in_source():
    setup = _Setup()
    setup.backuper.backup("actor")
    setup.params["stray_buffer"] = torch.zeros(4, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="stray_buffer"):
        setup.backuper.get("actor")


def test_non_actor_tag_keeps_full_pinned_copy():
    setup = _Setup()
    ref_values = {name: t.clone() for name, t in {**setup.params, **setup.extras}.items()}
    setup.backuper.backup("ref")
    assert setup.backuper.backup_tags == ["actor", "ref"]
    setup.corrupt_live_tensors()
    setup.backuper.restore("ref")
    for name, tensor in {**setup.params, **setup.extras}.items():
        assert torch.equal(tensor, ref_values[name]), name
    got = setup.backuper.get("ref")
    for name in ref_values:
        # Pinned host copies, never the live tensors.
        assert got[name].data_ptr() != {**setup.params, **setup.extras}[name].data_ptr(), name


def test_actor_restore_wins_after_ref_switch():
    # A ref switch overwrites the live params; the actor restore must win them back.
    setup = _Setup()
    actor_values = {name: t.clone() for name, t in {**setup.params, **setup.extras}.items()}
    setup.backuper.backup("actor")
    for param in setup.params.values():
        param.fill_(7.0)  # stand-in for loading ref weights into the model
    setup.backuper.backup("ref")
    setup.backuper.restore("ref")
    setup.backuper.restore("actor")
    for name, tensor in {**setup.params, **setup.extras}.items():
        assert torch.equal(tensor, actor_values[name]), name


def test_component_copy_preserves_native_data_and_scale_bytes():
    src = _FakeQuantized(
        torch.tensor([1, 2, 3], dtype=torch.uint8),
        torch.tensor([4, 5], dtype=torch.uint8),
    )
    dst = _FakeQuantized(torch.zeros(3, dtype=torch.uint8), torch.zeros(2, dtype=torch.uint8))

    _copy_tensor_value(dst, src)

    assert torch.equal(dst._rowwise_data, src._rowwise_data)
    assert torch.equal(dst._rowwise_scale_inv, src._rowwise_scale_inv)


def test_legacy_mxfp8_without_flatten_protocol_fails_closed():
    tensor = _LegacyMXFP8Tensor()

    with pytest.raises(RuntimeError, match="__tensor_flatten__"):
        native_components(tensor)
    with pytest.raises(RuntimeError, match="__tensor_flatten__"):
        _allocate_pinned_like(tensor)


def test_component_copy_rejects_schema_change_instead_of_requantizing():
    src = _FakeQuantized(torch.ones(3, dtype=torch.uint8), torch.ones(2, dtype=torch.uint8))
    dst = _FakeQuantized(torch.zeros(3, dtype=torch.uint8), torch.zeros(3, dtype=torch.uint8))

    with pytest.raises(RuntimeError, match="changed metadata"):
        _copy_tensor_value(dst, src)


def test_component_hash_covers_scale_bytes():
    tensor = _FakeQuantized(torch.ones(3, dtype=torch.uint8), torch.ones(2, dtype=torch.uint8))
    before = _hash_tensor_sha256(tensor)
    tensor._rowwise_scale_inv[0] = 2
    assert _hash_tensor_sha256(tensor) != before


def test_grouped_component_backup_never_dequantizes():
    src = _FakeGroupedTensor(
        (2, 4),
        torch.bfloat16,
        num_tensors=2,
        shapes=[(1, 4), (1, 4)],
        data=torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
        scale_inv=torch.tensor([5, 6], dtype=torch.uint8),
        offsets=[0, 4],
        scale_inv_offsets=[0, 2],
    )
    dst = _allocate_pinned_like(src)

    _copy_tensor_value(dst, src)

    assert isinstance(dst, _FakeGroupedTensor)
    assert torch.equal(dst.rowwise_data, src.rowwise_data)
    assert torch.equal(dst.scale_inv, src.scale_inv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_te_mxfp8_and_grouped_backups_copy_native_components():
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex

    available, reason = te.is_mxfp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    source = quantizer(torch.randn(256, 256, dtype=torch.bfloat16, device="cuda"))
    source_backup = _allocate_pinned_like(source)
    _copy_tensor_value(source_backup, source)
    source_target = quantizer(torch.zeros(256, 256, dtype=torch.bfloat16, device="cuda"))
    _copy_tensor_value(source_target, source_backup)

    source_components = dict(native_components(source))
    backup_components = dict(native_components(source_backup))
    target_components = dict(native_components(source_target))
    assert source_components.keys() == backup_components.keys() == target_components.keys()
    for name, expected in source_components.items():
        assert backup_components[name].device.type == "cpu"
        assert backup_components[name].is_pinned()
        assert torch.equal(backup_components[name], expected.cpu()), name
        assert torch.equal(target_components[name], expected), name

    grouped_quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    first_dims = torch.tensor([128, 256], dtype=torch.int64, device="cuda")
    grouped_source = tex.group_quantize(
        torch.randn(384, 256, dtype=torch.bfloat16, device="cuda"),
        grouped_quantizer,
        2,
        first_dims,
        None,
    )
    grouped_backup = _allocate_pinned_like(grouped_source)
    _copy_tensor_value(grouped_backup, grouped_source)
    grouped_target = tex.group_quantize(
        torch.zeros(384, 256, dtype=torch.bfloat16, device="cuda"),
        grouped_quantizer,
        2,
        first_dims,
        None,
    )
    _copy_tensor_value(grouped_target, grouped_backup)

    grouped_source_components = dict(native_components(grouped_source))
    grouped_backup_components = dict(native_components(grouped_backup))
    grouped_target_components = dict(native_components(grouped_target))
    assert grouped_source_components.keys() == grouped_backup_components.keys() == grouped_target_components.keys()
    for name, expected in grouped_source_components.items():
        assert grouped_backup_components[name].device.type == "cpu"
        assert grouped_backup_components[name].is_pinned()
        assert torch.equal(grouped_backup_components[name], expected.cpu()), name
        assert torch.equal(grouped_target_components[name], expected), name


def test_normal_backuper_deduplicates_same_schema_aliases():
    source = torch.arange(8, dtype=torch.float32)
    backuper = TensorBackuper.create(source_getter=lambda: iter((("a", source), ("b", source))))

    backuper.backup("actor")

    backups = backuper.get("actor")
    assert backups["a"] is backups["b"]


def test_normal_backuper_keeps_distinct_views_of_one_ddp_storage_separate():
    storage = torch.arange(12, dtype=torch.float32)
    first = storage[:4]
    second = storage[4:].view(2, 4)
    backuper = TensorBackuper.create(source_getter=lambda: iter((("first", first), ("second", second))))

    backuper.backup("actor")
    first.zero_()
    second.zero_()
    backuper.restore("actor")

    backups = backuper.get("actor")
    assert backups["first"] is not backups["second"]
    assert torch.equal(first, torch.arange(4, dtype=torch.float32))
    assert torch.equal(second, torch.arange(4, 12, dtype=torch.float32).view(2, 4))


def test_normal_backuper_rejects_alias_topology_change_between_cycles():
    shared = torch.arange(8, dtype=torch.float32)
    sources = {"a": shared, "b": shared}
    backuper = TensorBackuper.create(source_getter=lambda: iter(sources.items()))
    backuper.backup("actor")
    sources["b"] = shared.clone()

    with pytest.raises(RuntimeError, match="alias topology"):
        backuper.backup("actor")
