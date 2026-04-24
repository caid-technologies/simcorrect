"""Compatibility facade for legacy SimCorrect OpenCAD imports.

New MJCF correction code should import from `mjcf_correction` directly. The
CAID artifact helpers remain re-exported here so older scripts keep working.
"""

from caid_contract import (
    ContractError,
    apply_parameter_patch,
    apply_patch_to_simulation_params,
    get_parameter,
    load_artifact,
    make_parameter_patch,
    make_patch_from_identification,
    resolve_parameter_name,
    simulation_target_for_parameter,
    write_json,
)
from mjcf_correction import CorrectionRecord, Part

__all__ = [
    "ContractError",
    "CorrectionRecord",
    "Part",
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
