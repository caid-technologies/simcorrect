from __future__ import annotations

import json
from pathlib import Path
import unittest

from caid_contract import (
    ARTIFACT_REQUIRED_KEYS,
    ContractError,
    PATCH_REQUIRED_KEYS,
    apply_parameter_patch,
    apply_patch_to_simulation_params,
    get_parameter,
    load_artifact,
    make_parameter_patch,
    make_patch_from_identification,
    resolve_parameter_name,
    simulation_target_for_parameter,
)


def sample_artifact():
    return {
        "schema_version": 1,
        "artifact_id": "forearm-demo",
        "producer": {"name": "opencad", "version": "0.1.1"},
        "created_at": "2026-04-24T00:00:00Z",
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
            {"name": "right_forearm", "kind": "body", "target": "r_forearm"},
            {"name": "forearm_length", "kind": "parameter", "target": "link2_length"},
        ],
    }


class CaidContractTests(unittest.TestCase):
    def test_committed_json_schemas_expose_required_contract_keys(self):
        repo_root = Path(__file__).resolve().parents[1]
        artifact_schema = json.loads(
            (repo_root / "docs" / "schemas" / "caid-design-artifact-v1.schema.json").read_text(encoding="utf-8")
        )
        patch_schema = json.loads(
            (repo_root / "docs" / "schemas" / "caid-design-patch-v1.schema.json").read_text(encoding="utf-8")
        )

        self.assertEqual(set(artifact_schema["required"]), ARTIFACT_REQUIRED_KEYS)
        self.assertEqual(set(patch_schema["required"]), PATCH_REQUIRED_KEYS)
        self.assertEqual(artifact_schema["properties"]["schema_version"]["const"], 1)
        self.assertEqual(patch_schema["properties"]["schema_version"]["const"], 1)

    def test_load_and_read_parameter(self):
        artifact = load_artifact(sample_artifact())

        parameter = get_parameter(artifact, "forearm_length")

        self.assertEqual(parameter["value"], 0.25)
        self.assertEqual(parameter["unit"], "m")

    def test_make_and_apply_parameter_patch(self):
        artifact = load_artifact(sample_artifact())
        patch = make_parameter_patch(
            artifact,
            "forearm_length",
            0.30,
            reason="Vertical overshoot isolated to forearm length.",
        )

        updated = apply_parameter_patch(artifact, patch)

        self.assertEqual(patch["artifact_id"], "forearm-demo")
        self.assertEqual(patch["parameter_patches"][0]["old_value"], 0.25)
        self.assertEqual(updated["parameters"]["forearm_length"]["value"], 0.30)
        self.assertEqual(artifact["parameters"]["forearm_length"]["value"], 0.25)

    def test_patch_rejects_wrong_artifact(self):
        artifact = load_artifact(sample_artifact())
        patch = make_parameter_patch(artifact, "forearm_length", 0.30)
        patch["artifact_id"] = "other-design"

        with self.assertRaises(ContractError):
            apply_parameter_patch(artifact, patch)

    def test_patch_rejects_stale_old_value(self):
        artifact = load_artifact(sample_artifact())
        patch = make_parameter_patch(artifact, "forearm_length", 0.30)
        patch["parameter_patches"][0]["old_value"] = 0.20

        with self.assertRaisesRegex(ContractError, "expected old value"):
            apply_parameter_patch(artifact, patch)

    def test_patch_rejects_empty_parameter_patches(self):
        artifact = load_artifact(sample_artifact())

        with self.assertRaises(ContractError):
            apply_parameter_patch(
                artifact,
                {
                    "schema_version": 1,
                    "artifact_id": "forearm-demo",
                    "source": "simcorrect",
                    "parameter_patches": [],
                },
            )

    def test_artifact_rejects_missing_required_contract_key(self):
        artifact = sample_artifact()
        del artifact["producer"]

        with self.assertRaisesRegex(ContractError, "missing required key"):
            load_artifact(artifact)

    def test_artifact_rejects_parameter_name_mismatch(self):
        artifact = sample_artifact()
        artifact["parameters"]["forearm_length"]["name"] = "link2_length"

        with self.assertRaisesRegex(ContractError, "does not match"):
            load_artifact(artifact)

    def test_artifact_rejects_malformed_simulation_tag(self):
        artifact = sample_artifact()
        artifact["simulation_tags"][1]["kind"] = "unknown"

        with self.assertRaisesRegex(ContractError, "unsupported kind"):
            load_artifact(artifact)

    def test_patch_rejects_missing_source(self):
        artifact = load_artifact(sample_artifact())
        patch = make_parameter_patch(artifact, "forearm_length", 0.30)
        del patch["source"]

        with self.assertRaisesRegex(ContractError, "missing required key"):
            apply_parameter_patch(artifact, patch)

    def test_patch_rejects_unsupported_value_type(self):
        artifact = load_artifact(sample_artifact())
        patch = make_parameter_patch(artifact, "forearm_length", 0.30)
        patch["parameter_patches"][0]["value"] = {"meters": 0.30}

        with self.assertRaisesRegex(ContractError, "unsupported value type"):
            apply_parameter_patch(artifact, patch)

    def test_identification_requires_target_and_value(self):
        artifact = load_artifact(sample_artifact())

        with self.assertRaises(ContractError):
            make_patch_from_identification(artifact, {"identified_parameter": "link2_length"})

    def test_identification_patch_resolves_sim_parameter_target(self):
        artifact = load_artifact(sample_artifact())
        identification = {
            "identified_parameter": "link2_length",
            "proposed_value": 0.30,
            "method": "sensitivity_analysis",
        }

        patch = make_patch_from_identification(artifact, identification)

        self.assertEqual(resolve_parameter_name(artifact, "link2_length"), "forearm_length")
        self.assertEqual(simulation_target_for_parameter(artifact, "forearm_length"), "link2_length")
        self.assertEqual(patch["parameter_patches"][0]["name"], "forearm_length")
        self.assertEqual(patch["parameter_patches"][0]["value"], 0.30)

    def test_patch_updates_simulation_params_via_tag(self):
        artifact = load_artifact(sample_artifact())
        patch = make_parameter_patch(artifact, "forearm_length", 0.30)

        updated = apply_patch_to_simulation_params(
            artifact,
            patch,
            {"link1_length": 0.30, "link2_length": 0.25},
        )

        self.assertEqual(updated["link2_length"], 0.30)

    def test_simulation_params_reject_wrong_artifact_patch(self):
        artifact = load_artifact(sample_artifact())
        patch = make_parameter_patch(artifact, "forearm_length", 0.30)
        patch["artifact_id"] = "other-design"

        with self.assertRaises(ContractError):
            apply_patch_to_simulation_params(
                artifact,
                patch,
                {"link1_length": 0.30, "link2_length": 0.25},
            )

    def test_simulation_params_reject_stale_patch(self):
        artifact = load_artifact(sample_artifact())
        patch = make_parameter_patch(artifact, "forearm_length", 0.30)
        patch["parameter_patches"][0]["old_value"] = 0.20

        with self.assertRaisesRegex(ContractError, "expected old value"):
            apply_patch_to_simulation_params(
                artifact,
                patch,
                {"link1_length": 0.30, "link2_length": 0.25},
            )


if __name__ == "__main__":
    unittest.main()
