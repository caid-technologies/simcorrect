from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from Problem1_ForearmLength.trajectory_io import load_trajectories, save_trajectories


class Problem1TrajectoryIoTests(unittest.TestCase):
    def test_round_trip_uses_npz_without_pickle(self) -> None:
        trajectories = {
            "times": np.array([0.0, 0.1]),
            "ground_truth": {
                "joint_states": np.array([[1.0, 2.0], [1.1, 2.1]]),
                "ee_positions": np.array([[0.0, 0.1, 0.2], [0.2, 0.3, 0.4]]),
                "params": {"link1_length": 0.30, "link2_length": 0.25},
            },
            "faulty_model": {
                "joint_states": np.array([[0.9, 1.9], [1.0, 2.0]]),
                "ee_positions": np.array([[0.0, 0.0, 0.1], [0.1, 0.2, 0.3]]),
                "params": {"link1_length": 0.30, "link2_length": 0.22},
            },
            "injected_error": {
                "parameter": "link2_length",
                "true_value": 0.25,
                "faulty_value": 0.22,
                "error_magnitude": 0.03,
            },
        }

        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "trajectories.npz"
            save_trajectories(path, trajectories)
            loaded = load_trajectories(path)

        np.testing.assert_allclose(loaded["times"], trajectories["times"])
        np.testing.assert_allclose(
            loaded["ground_truth"]["joint_states"],
            trajectories["ground_truth"]["joint_states"],
        )
        np.testing.assert_allclose(
            loaded["faulty_model"]["ee_positions"],
            trajectories["faulty_model"]["ee_positions"],
        )
        self.assertEqual(loaded["injected_error"], trajectories["injected_error"])


if __name__ == "__main__":
    unittest.main()
