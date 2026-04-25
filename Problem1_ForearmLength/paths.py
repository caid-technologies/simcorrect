from __future__ import annotations

import os
import tempfile
from pathlib import Path

OUTPUT_DIR_ENV = "SIMCORRECT_OUTPUT_DIR"


def output_dir() -> Path:
    return Path(os.environ.get(OUTPUT_DIR_ENV, tempfile.gettempdir()))


def output_path(filename: str) -> Path:
    path = output_dir() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def trajectories_path() -> Path:
    return output_path("trajectories.npy")


def identification_result_path() -> Path:
    return output_path("identification_result.json")


def divergence_plot_path() -> Path:
    return output_path("divergence_detection.png")


def correction_plot_path() -> Path:
    return output_path("correction_validation.png")
