from __future__ import annotations

import unittest

from Problem1_ForearmLength.caid_loop import correct_params_from_artifact


def forearm_artifact():
    return {
        "schema_version": 1,
        "artifact_id": "forearm-demo",
        "producer": {"name": "opencad", "version": "0.1.1"},
        "feature_tree": {"root_id": "root", "nodes": {}},
        "parameters": {
            "forearm_length": {
                "name": "forearm_length",
                "value": 0.25,
                "unit": "m",
                "role": "geometry",
                "feature_id": "feat-0001",
            }
        },
        "simulation_tags": [
            {"name": "forearm_length", "kind": "parameter", "target": "link2_length"},
        ],
    }


class Problem1CaidLoopTests(unittest.TestCase):
    def test_link2_identification_updates_forearm_artifact_and_sim_params(self):
        result = correct_params_from_artifact(
            forearm_artifact(),
            {
                "identified_parameter": "link2_length",
                "proposed_value": 0.30,
                "method": "sensitivity_analysis",
            },
            {"link1_length": 0.30, "link2_length": 0.25},
        )

        self.assertEqual(result.patch["parameter_patches"][0]["name"], "forearm_length")
        self.assertEqual(result.corrected_artifact["parameters"]["forearm_length"]["value"], 0.30)
        self.assertEqual(result.corrected_params["link2_length"], 0.30)


if __name__ == "__main__":
    unittest.main()
