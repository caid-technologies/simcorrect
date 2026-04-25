from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ARTIFACT_REQUIRED_KEYS = frozenset(
    {"schema_version", "artifact_id", "producer", "created_at", "feature_tree", "parameters", "simulation_tags"}
)
PATCH_REQUIRED_KEYS = frozenset({"schema_version", "artifact_id", "source", "parameter_patches"})
SIMULATION_TAG_KINDS = frozenset({"body", "joint", "geom", "site", "parameter"})
PARAMETER_VALUE_TYPES = (bool, int, float, str)

__all__ = [
    "SCHEMA_VERSION",
    "ARTIFACT_REQUIRED_KEYS",
    "PATCH_REQUIRED_KEYS",
    "ContractError",
    "apply_parameter_patch",
    "apply_patch_to_simulation_params",
    "get_parameter",
    "load_artifact",
    "make_parameter_patch",
    "make_patch_from_identification",
    "resolve_parameter_name",
    "simulation_target_for_parameter",
    "write_json",
]


class ContractError(ValueError):
    """Raised when a CAID artifact or patch violates the integration contract."""


def load_artifact(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    artifact = _load_json(source)
    _require_artifact(artifact)
    return artifact


def get_parameter(artifact: dict[str, Any], name: str) -> dict[str, Any]:
    _require_artifact(artifact)
    try:
        parameter = artifact["parameters"][name]
    except KeyError as exc:
        raise ContractError(f"Unknown design parameter '{name}'.") from exc
    if not isinstance(parameter, dict) or "value" not in parameter:
        raise ContractError(f"Design parameter '{name}' is malformed.")
    return parameter


def resolve_parameter_name(artifact: dict[str, Any], name_or_target: str) -> str:
    _require_artifact(artifact)
    if name_or_target in artifact["parameters"]:
        return name_or_target

    for tag in artifact.get("simulation_tags", []):
        if tag.get("kind") == "parameter" and tag.get("target") == name_or_target:
            name = tag.get("name")
            if name in artifact["parameters"]:
                return name

    raise ContractError(f"Unknown design parameter or simulation target '{name_or_target}'.")


def simulation_target_for_parameter(artifact: dict[str, Any], name: str) -> str:
    _require_artifact(artifact)
    get_parameter(artifact, name)
    for tag in artifact.get("simulation_tags", []):
        if tag.get("kind") == "parameter" and tag.get("name") == name:
            target = tag.get("target")
            if isinstance(target, str) and target:
                return target
            raise ContractError(f"Simulation tag for '{name}' is missing a target.")
    return name


def make_parameter_patch(
    artifact: dict[str, Any],
    name: str,
    value: bool | int | float | str,
    *,
    reason: str | None = None,
    source: str = "simcorrect",
) -> dict[str, Any]:
    parameter = get_parameter(artifact, name)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": artifact["artifact_id"],
        "source": source,
        "parameter_patches": [
            {
                "name": name,
                "old_value": parameter["value"],
                "value": value,
                "reason": reason,
            }
        ],
    }


def make_patch_from_identification(
    artifact: dict[str, Any],
    identification: dict[str, Any],
    *,
    source: str = "simcorrect",
) -> dict[str, Any]:
    missing = [key for key in ("identified_parameter", "proposed_value") if key not in identification]
    if missing:
        raise ContractError(f"Identification result missing required key(s): {', '.join(missing)}.")
    name = resolve_parameter_name(artifact, identification["identified_parameter"])
    return make_parameter_patch(
        artifact,
        name,
        identification["proposed_value"],
        reason=f"{identification.get('method', 'parameter_identification')} identified {identification['identified_parameter']}.",
        source=source,
    )


def apply_parameter_patch(artifact: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    _require_artifact(artifact)
    _require_patch(patch)
    _require_patch_targets_artifact(artifact, patch)

    updated = deepcopy(artifact)
    for item in patch["parameter_patches"]:
        parameter = get_parameter(updated, item["name"])
        _require_current_value(parameter, item)
        parameter["value"] = item["value"]

    return updated


def apply_patch_to_simulation_params(
    artifact: dict[str, Any],
    patch: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    _require_artifact(artifact)
    _require_patch(patch)
    _require_patch_targets_artifact(artifact, patch)
    updated = dict(params)
    for item in patch["parameter_patches"]:
        parameter = get_parameter(artifact, item["name"])
        _require_current_value(parameter, item)
        target = simulation_target_for_parameter(artifact, item["name"])
        if target not in updated:
            raise ContractError(f"Simulation parameter '{target}' is not present in current params.")
        updated[target] = item["value"]
    return updated


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        return deepcopy(source)
    if not isinstance(source, (str, Path)):
        raise ContractError("CAID JSON source must be a path or object.")
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError("CAID JSON payload must be an object.")
    return payload


def _require_version(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"Unsupported CAID schema_version: {payload.get('schema_version')!r}.")


def _require_artifact(payload: dict[str, Any]) -> None:
    _require_object(payload, "CAID artifact")
    _require_keys(payload, ARTIFACT_REQUIRED_KEYS, "CAID artifact")
    _require_version(payload)
    if not isinstance(payload.get("artifact_id"), str) or not payload["artifact_id"]:
        raise ContractError("CAID artifact must contain a non-empty artifact_id.")
    _require_producer(payload["producer"])
    _require_feature_tree(payload["feature_tree"])
    if not isinstance(payload.get("parameters"), dict):
        raise ContractError("CAID artifact must contain a parameters object.")
    for name, parameter in payload["parameters"].items():
        _require_parameter(name, parameter)
    if not isinstance(payload["simulation_tags"], list):
        raise ContractError("CAID artifact simulation_tags must be a list.")
    for tag in payload["simulation_tags"]:
        _require_simulation_tag(tag)


def _require_patch(payload: dict[str, Any]) -> None:
    _require_object(payload, "CAID patch")
    _require_keys(payload, PATCH_REQUIRED_KEYS, "CAID patch")
    _require_version(payload)
    if not isinstance(payload.get("artifact_id"), str) or not payload["artifact_id"]:
        raise ContractError("CAID patch must contain a non-empty artifact_id.")
    if not isinstance(payload.get("source"), str) or not payload["source"]:
        raise ContractError("CAID patch must contain a non-empty source.")
    patches = payload.get("parameter_patches")
    if not isinstance(patches, list) or not patches:
        raise ContractError("CAID patch must contain at least one parameter patch.")
    for item in patches:
        _require_patch_item(item)


def _require_patch_item(item: Any) -> None:
    if not isinstance(item, dict):
        raise ContractError("Each parameter patch must be an object.")
    if not isinstance(item.get("name"), str) or not item["name"]:
        raise ContractError("Each parameter patch must contain a non-empty name.")
    if "value" not in item:
        raise ContractError(f"Parameter patch '{item.get('name')}' is missing value.")
    if not isinstance(item["value"], PARAMETER_VALUE_TYPES):
        raise ContractError(f"Parameter patch '{item['name']}' has unsupported value type.")
    if "old_value" in item and item["old_value"] is not None and not isinstance(item["old_value"], PARAMETER_VALUE_TYPES):
        raise ContractError(f"Parameter patch '{item['name']}' has unsupported old_value type.")
    if "reason" in item and item["reason"] is not None and not isinstance(item["reason"], str):
        raise ContractError(f"Parameter patch '{item['name']}' reason must be a string when present.")


def _require_patch_targets_artifact(artifact: dict[str, Any], patch: dict[str, Any]) -> None:
    if patch.get("artifact_id") != artifact.get("artifact_id"):
        raise ContractError("Patch artifact_id does not match artifact.")


def _require_current_value(parameter: dict[str, Any], item: dict[str, Any]) -> None:
    if item.get("old_value") is not None and item["old_value"] != parameter["value"]:
        raise ContractError(
            f"Patch for '{item['name']}' expected old value {item['old_value']!r}, "
            f"but artifact has {parameter['value']!r}."
        )


def _require_object(payload: Any, label: str) -> None:
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be an object.")


def _require_keys(payload: dict[str, Any], required: frozenset[str], label: str) -> None:
    missing = sorted(required - payload.keys())
    if missing:
        raise ContractError(f"{label} missing required key(s): {', '.join(missing)}.")


def _require_producer(producer: Any) -> None:
    if not isinstance(producer, dict):
        raise ContractError("CAID artifact producer must be an object.")
    if not isinstance(producer.get("name"), str) or not producer["name"]:
        raise ContractError("CAID artifact producer must contain a non-empty name.")
    if not isinstance(producer.get("version"), str) or not producer["version"]:
        raise ContractError("CAID artifact producer must contain a non-empty version.")


def _require_feature_tree(feature_tree: Any) -> None:
    if not isinstance(feature_tree, dict):
        raise ContractError("CAID artifact feature_tree must be an object.")
    if not isinstance(feature_tree.get("root_id"), str) or not feature_tree["root_id"]:
        raise ContractError("CAID artifact feature_tree must contain a non-empty root_id.")
    if not isinstance(feature_tree.get("nodes"), dict):
        raise ContractError("CAID artifact feature_tree must contain a nodes object.")


def _require_parameter(name: Any, parameter: Any) -> None:
    if not isinstance(name, str) or not name:
        raise ContractError("CAID artifact parameter keys must be non-empty strings.")
    if not isinstance(parameter, dict):
        raise ContractError(f"Design parameter '{name}' must be an object.")
    if parameter.get("name") != name:
        raise ContractError(f"Design parameter key '{name}' does not match parameter name '{parameter.get('name')}'.")
    if "value" not in parameter:
        raise ContractError(f"Design parameter '{name}' is missing value.")
    if not isinstance(parameter["value"], PARAMETER_VALUE_TYPES):
        raise ContractError(f"Design parameter '{name}' has unsupported value type.")
    for optional in ("unit", "role", "feature_id"):
        if optional in parameter and parameter[optional] is not None and not isinstance(parameter[optional], str):
            raise ContractError(f"Design parameter '{name}' field '{optional}' must be a string when present.")


def _require_simulation_tag(tag: Any) -> None:
    if not isinstance(tag, dict):
        raise ContractError("CAID artifact simulation tags must be objects.")
    if not isinstance(tag.get("name"), str) or not tag["name"]:
        raise ContractError("CAID artifact simulation tag must contain a non-empty name.")
    if tag.get("kind") not in SIMULATION_TAG_KINDS:
        raise ContractError(f"CAID artifact simulation tag has unsupported kind: {tag.get('kind')!r}.")
    if not isinstance(tag.get("target"), str) or not tag["target"]:
        raise ContractError("CAID artifact simulation tag must contain a non-empty target.")
    if "metadata" in tag and not isinstance(tag["metadata"], dict):
        raise ContractError("CAID artifact simulation tag metadata must be an object when present.")
