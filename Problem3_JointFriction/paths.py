from __future__ import annotations

from pathlib import Path

from simcorrect_paths import OUTPUT_DIR_ENV, problem_paths

_PATHS = problem_paths(__file__)
output_dir = _PATHS.output_dir
output_path = _PATHS.output_path
video_path = _PATHS.video_path


def problem_dir() -> Path:
    return _PATHS.problem_dir
