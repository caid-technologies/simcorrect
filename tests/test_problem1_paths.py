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
)


class Problem1PathTests(unittest.TestCase):
    def test_problem1_outputs_are_rooted_in_configured_output_dir(self):
        root = Path(__file__).resolve().parent

        with patch.dict(os.environ, {OUTPUT_DIR_ENV: str(root)}):
            self.assertEqual(output_dir(), root)
            self.assertEqual(trajectories_path(), root / "trajectories.npy")
            self.assertEqual(identification_result_path(), root / "identification_result.json")
            self.assertEqual(divergence_plot_path(), root / "divergence_detection.png")
            self.assertEqual(correction_plot_path(), root / "correction_validation.png")


if __name__ == "__main__":
    unittest.main()
