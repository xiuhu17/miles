import hashlib
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from miles.utils.native_param_storage import (
    exact_alias_key,
    iter_components,
    native_components,
    storage_wrapper_components,
    tensor_schema,
)

_SourceGetter = Callable[[], Iterable[tuple[str, torch.Tensor]]]


@dataclass(frozen=True)
class MainCastContext:
    # Writes this rank's owned shard from the master weights, as the train-step end does.
    cast_main_to_params: Callable[[], None]
    model_chunks: list
    extras_getter: _SourceGetter
    rematerializable_ids: set
    check: bool


class TensorBackuper(ABC):
    @staticmethod
    def create(source_getter, main_cast_ctx: "MainCastContext | None" = None):
        if main_cast_ctx is not None:
            return _TensorBackuperMainCast(source_getter=source_getter, ctx=main_cast_ctx)
        return _TensorBackuperNormal(source_getter=source_getter)

    def __init__(self, source_getter: _SourceGetter):
        self._source_getter = source_getter

    @property
    @abstractmethod
    def backup_tags(self):
        raise NotImplementedError

    @abstractmethod
    def get(self, tag: str):
        raise NotImplementedError

    @abstractmethod
    def backup(self, tag: str):
        raise NotImplementedError

    def copy(self, *, src_tag: str, dst_tag: str):
        raise NotImplementedError

    @abstractmethod
    def restore(self, tag: str):
        raise NotImplementedError

    @abstractmethod
    def pinned_backup_names(self, tag: str) -> set[str]:
        """Names whose recovery bytes are physically held by this backuper."""
        raise NotImplementedError


def _allocate_pinned_like(tensor: torch.Tensor) -> torch.Tensor:
    """Allocate a CPU wrapper whose physical components are individually pinned."""
    candidate, components = storage_wrapper_components(tensor)
    if components:
        flatten = getattr(candidate, "__tensor_flatten__", None)
        if callable(flatten):
            _names, context = flatten()
            components = {
                name: torch.empty_like(component, device=torch.device("cpu"), pin_memory=True)
                for name, component in components
            }
            return type(candidate).__tensor_unflatten__(
                components,
                context,
                candidate.size(),
                candidate.stride(),
            )

        if "GroupedTensor" in type(candidate).__name__:
            components = {
                name: torch.empty_like(value, device=torch.device("cpu"), pin_memory=True)
                for name, value in components
            }
            return type(candidate)(
                shape=candidate.logical_shape,
                dtype=candidate.fake_dtype,
                num_tensors=candidate.num_tensors,
                shapes=candidate.tensor_shapes,
                quantizer=candidate.quantizer,
                data=components.get("rowwise_data"),
                columnwise_data=components.get("columnwise_data"),
                scale_inv=components.get("scale_inv"),
                columnwise_scale_inv=components.get("columnwise_scale_inv"),
                amax=components.get("amax"),
                columnwise_amax=components.get("columnwise_amax"),
                scale=components.get("scale"),
                first_dims=components.get("first_dims"),
                last_dims=components.get("last_dims"),
                tensor_offsets=components.get("tensor_offsets"),
                offsets=candidate.offsets,
                scale_inv_offsets=candidate.scale_inv_offsets,
                columnwise_scale_inv_offsets=candidate.columnwise_scale_inv_offsets,
                requires_grad=False,
                stride=tuple(candidate.stride()),
                with_gemm_swizzled_scales=candidate._with_gemm_swizzled_scales,
                row_scaled_nvfp4=candidate.row_scaled_nvfp4,
                nvfp4_use_4over6=candidate.nvfp4_use_4over6,
                nvfp4_e4m3_max=candidate.nvfp4_e4m3_max,
            )

    return torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)


def _copy_tensor_value(dst: torch.Tensor, src: torch.Tensor, *, non_blocking: bool = False) -> None:
    """Copy native component bytes when tensors are storage-wrapper subclasses."""
    dst_components = native_components(dst)
    src_components = native_components(src)
    if dst_components or src_components:
        dst_names = tuple(name for name, _tensor in dst_components)
        src_names = tuple(name for name, _tensor in src_components)
        if dst_names != src_names:
            raise RuntimeError(
                f"quantized backup component schema changed: source={src_names}, destination={dst_names}"
            )
        for (name, dst_component), (_src_name, src_component) in zip(dst_components, src_components, strict=True):
            if dst_component.shape != src_component.shape or dst_component.dtype != src_component.dtype:
                raise RuntimeError(
                    f"quantized backup component {name!r} changed metadata: "
                    f"source={src_component.shape}/{src_component.dtype}, "
                    f"destination={dst_component.shape}/{dst_component.dtype}"
                )
            dst_component.copy_(src_component.detach(), non_blocking=non_blocking)
        return
    dst.copy_(src.detach(), non_blocking=non_blocking)


class _TensorBackuperNormal(TensorBackuper):
    def __init__(self, source_getter):
        super().__init__(source_getter=source_getter)
        self._backups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        self._alias_of: dict[str, str] | None = None
        self._schema_by_name: dict[str, tuple] | None = None

    @property
    def backup_tags(self):
        return list(self._backups)

    def get(self, tag: str):
        assert tag in self._backups, f"tag {tag!r} was never backed up"
        return self._backups[tag]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        source_items = list(self._source_getter())
        source_by_name = dict(source_items)
        if len(source_by_name) != len(source_items):
            raise RuntimeError("TensorBackuper source contains duplicate names")

        names_by_alias: dict[tuple[tuple, ...], list[str]] = defaultdict(list)
        schema_by_name = {}
        for name, param in source_by_name.items():
            alias_key = exact_alias_key(param)
            names_by_alias[alias_key].append(name)
            schema_by_name[name] = tensor_schema(param)

        alias_of = {}
        for names in names_by_alias.values():
            canonical_name = min(names)
            canonical_schema = schema_by_name[canonical_name]
            for name in names:
                if schema_by_name[name] != canonical_schema:
                    raise RuntimeError(
                        "PINNED tensors alias one physical allocation with incompatible schemas: "
                        f"{canonical_name!r} and {name!r}"
                    )
                alias_of[name] = canonical_name

        if self._alias_of is None:
            self._alias_of = alias_of
            self._schema_by_name = schema_by_name
        elif alias_of != self._alias_of or schema_by_name != self._schema_by_name:
            raise RuntimeError("TensorBackuper source alias topology or component schema changed after initialization")

        for canonical_name in sorted(set(alias_of.values())):
            param = source_by_name[canonical_name]
            if canonical_name not in backup_dict:
                backup_dict[canonical_name] = _allocate_pinned_like(param)
            _copy_tensor_value(backup_dict[canonical_name], param, non_blocking=True)
        for name, canonical_name in alias_of.items():
            backup_dict[name] = backup_dict[canonical_name]
        torch.cuda.synchronize()

    @torch.no_grad()
    def copy(self, *, src_tag: str, dst_tag: str):
        for name in self._backups[dst_tag]:
            _copy_tensor_value(self._backups[dst_tag][name], self._backups[src_tag][name])

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            assert name in backup_dict
            _copy_tensor_value(param, backup_dict[name], non_blocking=True)
        torch.cuda.synchronize()

    def pinned_backup_names(self, tag: str) -> set[str]:
        return set(self._backups.get(tag, ()))


class _TensorBackuperMainCast(TensorBackuper):
    """Rebuilds the actor weights instead of keeping a pinned CPU copy of them.

    Restore replays the step end's cast + all-gather, so it is bit-identical. Only
    `extras_getter` tensors keep a pinned backup. Non-actor tags (ref/teacher) have no
    master weights to rebuild from, so they keep full pinned copies via a delegated
    _TensorBackuperNormal.
    """

    _check_num_cycles = 2

    def __init__(self, source_getter, ctx: MainCastContext):
        super().__init__(source_getter=source_getter)
        self._ctx = ctx
        self._others = _TensorBackuperNormal(source_getter=source_getter)
        self._extras = _TensorBackuperNormal(source_getter=ctx.extras_getter)
        self._backup_count = 0
        self._expected_hashes: dict[str, str] | None = None

    @property
    def backup_tags(self):
        return ["actor", *self._others.backup_tags]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        if tag != "actor":
            return self._others.backup(tag)
        self._extras.backup("actor")
        self._backup_count += 1
        if self._ctx.check and self._backup_count <= self._check_num_cycles:
            self._expected_hashes = self._compute_hashes()
        else:
            self._expected_hashes = None

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        if tag != "actor":
            return self._others.restore(tag)
        self._ctx.cast_main_to_params()
        for model_chunk in self._ctx.model_chunks:
            model_chunk.start_param_sync(force_sync=True)
        self._extras.restore("actor")
        if self._expected_hashes is not None:
            self._verify_hashes()

    def get(self, tag: str):
        if tag != "actor":
            return self._others.get(tag)
        # Extras are paused during update_weights. Read them from the pinned backup.
        out = {}
        extras_by_tensor_id = self._actor_extra_backups_by_tensor_id()
        for name, tensor in self._source_getter():
            # extras_getter uses Megatron-local vp_stages.* names while the
            # rollout source may use globally converted HF names. Match the
            # stable live tensor object, not those two unrelated namespaces.
            backup = extras_by_tensor_id.get(id(tensor))
            if backup is None:
                assert (
                    id(tensor) in self._ctx.rematerializable_ids
                ), f"{name} is neither in the DDP param buffers nor in the extras backup"
                backup = tensor.detach()
            out[name] = backup
        return out

    def pinned_backup_names(self, tag: str) -> set[str]:
        if tag == "actor":
            extras_by_tensor_id = self._actor_extra_backups_by_tensor_id()
            return {name for name, tensor in self._source_getter() if id(tensor) in extras_by_tensor_id}
        return self._others.pinned_backup_names(tag)

    def _actor_extra_backups_by_tensor_id(self) -> dict[int, torch.Tensor]:
        backups = self._extras.get("actor")
        return {id(tensor): backups[name] for name, tensor in self._ctx.extras_getter()}

    def _compute_hashes(self) -> dict[str, str]:
        return {name: _hash_tensor_sha256(tensor) for name, tensor in self._source_getter()}

    def _verify_hashes(self) -> None:
        actual = self._compute_hashes()
        expected = self._expected_hashes
        assert expected is not None
        assert actual.keys() == expected.keys(), (
            f"main-cast restore changed the tensor set: "
            f"missing={sorted(expected.keys() - actual.keys())[:5]} "
            f"extra={sorted(actual.keys() - expected.keys())[:5]}"
        )
        mismatches = [name for name in expected if actual[name] != expected[name]]
        if mismatches:
            raise RuntimeError(
                f"main-cast weight restore is not bit-identical to the weights at "
                f"backup time for {len(mismatches)}/{len(expected)} tensors "
                f"(cycle {self._backup_count}): {mismatches[:20]}"
            )


def _hash_tensor_sha256(x: torch.Tensor) -> str:
    """Real (cryptographic) hash: a mismatch here has to mean a bug."""
    digest = hashlib.sha256()
    for name, component in iter_components(x):
        data = component.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(data.shape)).encode("utf-8"))
        digest.update(str(data.dtype).encode("utf-8"))
        digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
