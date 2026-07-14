import json
import logging
from pathlib import Path

import torch

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def trajectory_rows(samples: list[Sample]) -> list[dict]:
    """One row per sample that recorded a raw conversation
    (``metadata["messages"]``, attached by the session / multi_turn paths)."""
    rows = []
    for sample in samples:
        messages = sample.metadata.get("messages") if sample.metadata else None
        if messages is None:
            continue
        rows.append(
            dict(
                sample_index=sample.index,
                group_index=sample.group_index,
                status=sample.status.value,
                reward=sample.reward if isinstance(sample.reward, (int, float)) else None,
                prompt=sample.prompt,
                messages=messages,
            )
        )
    return rows


def save_debug_trajectory_data(args, samples: list[Sample], rollout_id, evaluation: bool):
    if (path_template := args.save_debug_trajectory_data) is None:
        return
    rows = trajectory_rows(samples)
    if not rows:
        return  # no conversations: no file (the dashboard keys off its presence)
    path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
    logger.info(f"Save trajectory dump to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    for sample in samples:
        if sample.metadata:
            sample.metadata.pop("messages", None)  # the sidecar is their home; keep the .pt lean


def load_debug_rollout_data(args, rollout_id: int):
    data = torch.load(
        args.load_debug_rollout_data.format(rollout_id=rollout_id),
        weights_only=False,
    )["samples"]
    data = [Sample.from_dict(sample) for sample in data]
    if (ratio := args.load_debug_rollout_data_subsample) is not None:
        original_num_rows = len(data)
        rough_subsample_num_rows = int(original_num_rows * ratio)
        data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
        logger.info(
            f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
        )
    return data


def save_debug_rollout_data(args, data, rollout_id, evaluation: bool):
    # TODO to be refactored (originally Buffer._set_data)
    if (path_template := args.save_debug_rollout_data) is not None:
        path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
        logger.info(f"Save debug rollout data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        samples = [sample for info in data.values() for sample in info["samples"]] if evaluation else list(data)
        save_debug_trajectory_data(args, samples, rollout_id, evaluation)

        # TODO may improve the format
        dump_data = dict(samples=[sample.to_dict() for sample in samples])
        torch.save(dict(rollout_id=rollout_id, **dump_data), path)
