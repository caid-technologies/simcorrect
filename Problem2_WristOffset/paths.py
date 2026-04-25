from __future__ import annotations

from pathlib import Path
import sys

_PROBLEM_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PROBLEM_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simcorrect_paths import OUTPUT_DIR_ENV, problem_paths

_PATHS = problem_paths(__file__)
output_dir = _PATHS.output_dir
output_path = _PATHS.output_path
video_path = _PATHS.video_path


def problem_dir() -> Path:
    return _PATHS.problem_dir
