from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

__all__ = [
    "SCHEMA_VERSION",
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
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError("CAID JSON payload must be an object.")
    return payload


def _require_version(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"Unsupported CAID schema_version: {payload.get('schema_version')!r}.")


def _require_artifact(payload: dict[str, Any]) -> None:
    _require_version(payload)
    if not isinstance(payload.get("artifact_id"), str) or not payload["artifact_id"]:
        raise ContractError("CAID artifact must contain a non-empty artifact_id.")
    if not isinstance(payload.get("parameters"), dict):
        raise ContractError("CAID artifact must contain a parameters object.")
    if "simulation_tags" in payload and not isinstance(payload["simulation_tags"], list):
        raise ContractError("CAID artifact simulation_tags must be a list when present.")


def _require_patch(payload: dict[str, Any]) -> None:
    _require_version(payload)
    if not isinstance(payload.get("artifact_id"), str) or not payload["artifact_id"]:
        raise ContractError("CAID patch must contain a non-empty artifact_id.")
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


def _require_patch_targets_artifact(artifact: dict[str, Any], patch: dict[str, Any]) -> None:
    if patch.get("artifact_id") != artifact.get("artifact_id"):
        raise ContractError("Patch artifact_id does not match artifact.")


def _require_current_value(parameter: dict[str, Any], item: dict[str, Any]) -> None:
    if item.get("old_value") is not None and item["old_value"] != parameter["value"]:
        raise ContractError(
            f"Patch for '{item['name']}' expected old value {item['old_value']!r}, "
            f"but artifact has {parameter['value']!r}."
        )
