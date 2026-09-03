from argparse import Namespace

import pytest

from miles.backends.megatron_utils.trainable_param_lifecycle import TrainableParameterLifecycle, TrainableParameterMode


def _args(**overrides) -> Namespace:
    values = dict(
        rematerialize_param_from_master_weight=False,
        colocate=True,
        offload_train=True,
        lora_rank=0,
        lora_adapter_path=None,
        lora_train_only=False,
        multi_lora=False,
        debug_train_only=False,
        use_distributed_optimizer=True,
    )
    return Namespace(**(values | overrides))


@pytest.mark.parametrize(
    ("overrides", "expected_mode", "expected_events"),
    [
        (
            {},
            TrainableParameterMode.PINNED,
            [("pause", None), ("resume", None), ("restore", "actor")],
        ),
        (
            {"rematerialize_param_from_master_weight": True},
            TrainableParameterMode.REMATERIALIZE,
            [
                ("pause", "grad_buffer"),
                ("pause", "default"),
                ("pause", "param_buffer"),
                ("resume", None),
                ("restore", "actor"),
            ],
        ),
        (
            {"lora_rank": 8},
            TrainableParameterMode.PINNED,
            [
                ("pause", None),
                ("resume", None),
                ("restore", "actor"),
            ],
        ),
        (
            {"lora_rank": 8, "rematerialize_param_from_master_weight": True},
            TrainableParameterMode.REMATERIALIZE,
            [
                ("pause", "grad_buffer"),
                ("pause", "default"),
                ("pause", "param_buffer"),
                ("resume", None),
                ("restore", "actor"),
            ],
        ),
    ],
)
def test_full_ft_and_lora_execute_one_lifecycle(overrides, expected_mode, expected_events):
    events = []
    lifecycle = TrainableParameterLifecycle.from_args(_args(**overrides), "actor")

    lifecycle.offload_after_train(pause=lambda *, tag: events.append(("pause", tag)))
    lifecycle.finish_publish_after_ack(pause=lambda *, tag: events.append(("pause", tag)))
    lifecycle.onload_before_train(resume=lambda *, tag: events.append(("resume", tag)))
    lifecycle.restore_before_train(lambda: events.append(("restore", "actor")))

    assert lifecycle.mode is expected_mode
    assert lifecycle.manages_trainable_parameters
    assert events == expected_events


@pytest.mark.parametrize(
    ("overrides", "role", "expected_mode", "tag"),
    [
        ({"lora_rank": 8, "multi_lora": True}, "actor", TrainableParameterMode.LEGACY_DEFAULT_ONLY, "default"),
        ({"lora_rank": 8, "lora_train_only": True}, "actor", TrainableParameterMode.LEGACY_ALL, None),
        ({"lora_rank": 8, "offload_train": False}, "actor", TrainableParameterMode.LEGACY_DEFAULT_ONLY, "default"),
        (
            {"lora_rank": 8, "use_distributed_optimizer": False},
            "actor",
            TrainableParameterMode.LEGACY_DEFAULT_ONLY,
            "default",
        ),
        ({"lora_rank": 8}, "critic", TrainableParameterMode.LEGACY_DEFAULT_ONLY, "default"),
    ],
)
def test_out_of_scope_lora_paths_preserve_legacy_behavior(overrides, role, expected_mode, tag):
    events = []
    lifecycle = TrainableParameterLifecycle.from_args(_args(**overrides), role)

    lifecycle.offload_after_train(pause=lambda *, tag: events.append(("pause", tag)))
    lifecycle.finish_publish_after_ack(pause=lambda *, tag: events.append(("pause", tag)))
    lifecycle.onload_before_train(resume=lambda *, tag: events.append(("resume", tag)))
    lifecycle.restore_before_train(lambda: events.append(("restore", "actor")))

    assert lifecycle.mode is expected_mode
    assert not lifecycle.manages_trainable_parameters
    assert events == [("pause", tag), ("resume", tag)]


def test_actor_model_switch_satisfies_pending_restore_once():
    restores = []
    lifecycle = TrainableParameterLifecycle.from_args(_args(), "actor")
    lifecycle.offload_after_train(pause=lambda *, tag: None)

    lifecycle.mark_trainable_parameters_restored()
    lifecycle.restore_before_train(lambda: restores.append("actor"))

    assert restores == []
