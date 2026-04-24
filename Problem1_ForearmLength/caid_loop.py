from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from caid_contract import (
    apply_parameter_patch,
    apply_patch_to_simulation_params,
    load_artifact,
    make_patch_from_identification,
)


@dataclass(frozen=True)
class CaidCorrection:
    corrected_params: dict[str, Any]
    patch: dict[str, Any]
    corrected_artifact: dict[str, Any]


def correct_params_from_artifact(
    artifact_source: str | Path | dict[str, Any],
    identification_result: dict[str, Any],
    current_params: dict[str, Any],
) -> CaidCorrection:
    artifact = load_artifact(artifact_source)
    patch = make_patch_from_identification(artifact, identification_result)
    corrected_artifact = apply_parameter_patch(artifact, patch)
    corrected_params = apply_patch_to_simulation_params(artifact, patch, current_params)
    return CaidCorrection(
        corrected_params=corrected_params,
        patch=patch,
        corrected_artifact=corrected_artifact,
    )
