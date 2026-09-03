"""Shared trainable-parameter memory lifecycle for Megatron actors."""

from argparse import Namespace
from collections.abc import Callable
from enum import Enum

from miles.utils.lora import is_lora_enabled, lora_rollout_enabled
from miles.utils.multi_lora import is_multi_lora_enabled

_TaggedMemoryOperation = Callable[..., None]


class TrainableParameterMode(Enum):
    """The four supported publish/restore ownership combinations."""

    PINNED = "pinned"
    REMATERIALIZE = "rematerialize"
    LEGACY_DEFAULT_ONLY = "legacy_default_only"
    LEGACY_ALL = "legacy_all"


def _uses_single_lora_adapter_lifecycle(args: Namespace, role: str) -> bool:
    return (
        role == "actor"
        and getattr(args, "colocate", False)
        and getattr(args, "offload_train", False)
        and lora_rollout_enabled(args)
        and not is_multi_lora_enabled(args)
        and not getattr(args, "debug_train_only", False)
        and getattr(args, "use_distributed_optimizer", False)
    )


class TrainableParameterLifecycle:
    """Coordinate trainer offload around an opaque weight publisher."""

    def __init__(self, mode: TrainableParameterMode) -> None:
        self.mode = mode
        self._restore_pending = False

    @classmethod
    def from_args(cls, args: Namespace, role: str) -> "TrainableParameterLifecycle":
        manages_trainable_parameters = role == "actor" and (
            not is_lora_enabled(args) or _uses_single_lora_adapter_lifecycle(args, role)
        )
        if manages_trainable_parameters:
            mode = (
                TrainableParameterMode.REMATERIALIZE
                if getattr(args, "rematerialize_param_from_master_weight", False)
                else TrainableParameterMode.PINNED
            )
        else:
            # Frozen-base and unsupported LoRA allocations retain their existing behavior.
            mode = (
                TrainableParameterMode.LEGACY_DEFAULT_ONLY
                if lora_rollout_enabled(args)
                else TrainableParameterMode.LEGACY_ALL
            )
        return cls(mode)

    @property
    def manages_trainable_parameters(self) -> bool:
        return self.mode in (TrainableParameterMode.PINNED, TrainableParameterMode.REMATERIALIZE)

    def offload_after_train(self, *, pause: _TaggedMemoryOperation) -> None:
        if self.mode is TrainableParameterMode.REMATERIALIZE:
            pause(tag="grad_buffer")
            pause(tag="default")
        elif self.mode is TrainableParameterMode.LEGACY_DEFAULT_ONLY:
            pause(tag="default")
        else:
            pause(tag=None)

        if self.mode is TrainableParameterMode.PINNED:
            self._restore_pending = True

    def finish_publish_after_ack(self, *, pause: _TaggedMemoryOperation) -> None:
        if self.mode is not TrainableParameterMode.REMATERIALIZE:
            return
        pause(tag="param_buffer")
        self._restore_pending = True

    def onload_before_train(self, *, resume: _TaggedMemoryOperation) -> None:
        tag = "default" if self.mode is TrainableParameterMode.LEGACY_DEFAULT_ONLY else None
        resume(tag=tag)

    def restore_before_train(self, restore: Callable[[], None]) -> None:
        if self._restore_pending:
            restore()
            self._restore_pending = False

    def mark_trainable_parameters_restored(self) -> None:
        self._restore_pending = False
