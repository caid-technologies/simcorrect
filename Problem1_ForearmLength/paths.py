from __future__ import annotations

from pathlib import Path
import sys

_PROBLEM_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PROBLEM_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simcorrect_paths import OUTPUT_DIR_ENV, output_dir as _output_dir, output_path as _output_path

_VIDEO_OUTPUT_DIR = _PROBLEM_DIR / "output"


def output_dir() -> Path:
    return _output_dir()


def output_path(filename: str) -> Path:
    return _output_path(filename)


def video_path(filename: str) -> Path:
    return _output_path(filename, _VIDEO_OUTPUT_DIR)


def trajectories_path() -> Path:
    return output_path("trajectories.npy")


def identification_result_path() -> Path:
    return output_path("identification_result.json")


def divergence_plot_path() -> Path:
    return output_path("divergence_detection.png")


def correction_plot_path() -> Path:
    return output_path("correction_validation.png")
