"""Safe trajectory persistence for the Problem 1 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_trajectories(path: Path, trajectories: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "ground_truth_params": trajectories["ground_truth"]["params"],
        "faulty_model_params": trajectories["faulty_model"]["params"],
        "injected_error": trajectories["injected_error"],
    }
    np.savez_compressed(
        path,
        times=trajectories["times"],
        ground_truth_joint_states=trajectories["ground_truth"]["joint_states"],
        ground_truth_ee_positions=trajectories["ground_truth"]["ee_positions"],
        faulty_model_joint_states=trajectories["faulty_model"]["joint_states"],
        faulty_model_ee_positions=trajectories["faulty_model"]["ee_positions"],
        metadata=json.dumps(metadata, sort_keys=True),
    )


def load_trajectories(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        return {
            "times": data["times"],
            "ground_truth": {
                "joint_states": data["ground_truth_joint_states"],
                "ee_positions": data["ground_truth_ee_positions"],
                "params": metadata["ground_truth_params"],
            },
            "faulty_model": {
                "joint_states": data["faulty_model_joint_states"],
                "ee_positions": data["faulty_model_ee_positions"],
                "params": metadata["faulty_model_params"],
            },
            "injected_error": metadata["injected_error"],
        }
