from __future__ import annotations

from pathlib import Path

from simcorrect_paths import OUTPUT_DIR_ENV, problem_paths

_PATHS = problem_paths(__file__)
output_dir = _PATHS.output_dir
output_path = _PATHS.output_path
video_path = _PATHS.video_path


def corrected_grip_xml_path() -> Path:
    return output_path("grip_corrected.xml")


def smoke_test_xml_path() -> Path:
    return output_path("opencad_test.xml")
