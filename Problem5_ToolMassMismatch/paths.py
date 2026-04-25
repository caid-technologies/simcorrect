from __future__ import annotations

from pathlib import Path
import sys

_PROBLEM_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PROBLEM_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simcorrect_paths import OUTPUT_DIR_ENV, output_dir as _output_dir, output_path as _output_path

_DEFAULT_OUTPUT_DIR = _PROBLEM_DIR / "output"


def output_dir() -> Path:
    return _output_dir(_DEFAULT_OUTPUT_DIR)


def output_path(filename: str | Path) -> Path:
    return _output_path(filename, _DEFAULT_OUTPUT_DIR)


def corrected_grip_xml_path() -> Path:
    return output_path("grip_corrected.xml")


def smoke_test_xml_path() -> Path:
    return output_path("opencad_test.xml")


def video_path(filename: str) -> Path:
    return output_path(filename)
