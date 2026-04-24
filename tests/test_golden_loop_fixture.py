from __future__ import annotations

from pathlib import Path
import unittest

from caid_contract import (
    apply_parameter_patch,
    apply_patch_to_simulation_params,
    load_artifact,
    make_patch_from_identification,
)


FIXTURE = Path(__file__).parent / "fixtures" / "opencad_forearm_artifact.json"


class GoldenLoopFixtureTests(unittest.TestCase):
    def test_opencad_artifact_fixture_produces_simcorrect_patch(self):
        artifact = load_artifact(FIXTURE)
        identification = {
            "identified_parameter": "link2_length",
            "current_value": 0.25,
            "proposed_value": 0.30,
            "method": "sensitivity_analysis",
        }

        patch = make_patch_from_identification(artifact, identification)
        corrected_artifact = apply_parameter_patch(artifact, patch)
        corrected_params = apply_patch_to_simulation_params(
            artifact,
            patch,
            {"link1_length": 0.30, "link2_length": 0.25},
        )

        self.assertEqual(patch["artifact_id"], "simcorrect-problem1-forearm")
        self.assertEqual(patch["parameter_patches"][0]["name"], "forearm_length")
        self.assertEqual(patch["parameter_patches"][0]["old_value"], 0.25)
        self.assertEqual(patch["parameter_patches"][0]["value"], 0.30)
        self.assertEqual(corrected_artifact["parameters"]["forearm_length"]["value"], 0.30)
        self.assertEqual(corrected_params["link2_length"], 0.30)


if __name__ == "__main__":
    unittest.main()
