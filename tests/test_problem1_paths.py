from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from Problem1_ForearmLength.paths import (
    OUTPUT_DIR_ENV,
    correction_plot_path,
    divergence_plot_path,
    identification_result_path,
    output_dir,
    trajectories_path,
    video_path as problem1_video_path,
)
from Problem2_WristOffset.paths import video_path as problem2_video_path
from Problem3_JointFriction.paths import problem_dir as problem3_dir
from Problem3_JointFriction.paths import output_dir as problem3_output_dir
from Problem3_JointFriction.paths import video_path as problem3_video_path
from Problem4_JointZeroOffset.paths import output_dir as problem4_output_dir
from Problem4_JointZeroOffset.paths import problem_dir as problem4_dir
from Problem4_JointZeroOffset.paths import video_path as problem4_video_path
from Problem5_ToolMassMismatch.paths import corrected_grip_xml_path


class Problem1PathTests(unittest.TestCase):
    def test_problem1_outputs_are_rooted_in_configured_output_dir(self):
        root = Path(__file__).resolve().parent

        with patch.dict(os.environ, {OUTPUT_DIR_ENV: str(root)}):
            self.assertEqual(output_dir(), root)
            self.assertEqual(trajectories_path(), root / "trajectories.npz")
            self.assertEqual(identification_result_path(), root / "identification_result.json")
            self.assertEqual(divergence_plot_path(), root / "divergence_detection.png")
            self.assertEqual(correction_plot_path(), root / "correction_validation.png")

    def test_problem5_outputs_share_configured_output_dir(self):
        root = Path(__file__).resolve().parent

        with patch.dict(os.environ, {OUTPUT_DIR_ENV: str(root)}):
            self.assertEqual(corrected_grip_xml_path(), root / "grip_corrected.xml")

    def test_problem_video_outputs_share_configured_output_dir(self):
        root = Path(__file__).resolve().parent

        with patch.dict(os.environ, {OUTPUT_DIR_ENV: str(root)}):
            self.assertEqual(problem1_video_path("problem1.mp4"), root / "problem1.mp4")
            self.assertEqual(problem2_video_path("problem2.mp4"), root / "problem2.mp4")
            self.assertEqual(problem3_video_path("problem3.mp4"), root / "problem3.mp4")
            self.assertEqual(problem4_video_path("problem4.mp4"), root / "problem4.mp4")

    def test_problem4_defaults_to_problem_output_directory(self):
        with patch.dict(os.environ, {OUTPUT_DIR_ENV: ""}):
            self.assertEqual(problem4_output_dir(), problem4_dir() / "output")

    def test_problem3_snapshot_directory_uses_problem_output_directory(self):
        with patch.dict(os.environ, {OUTPUT_DIR_ENV: ""}):
            self.assertEqual(problem3_output_dir(), problem3_dir() / "output")


if __name__ == "__main__":
    unittest.main()
