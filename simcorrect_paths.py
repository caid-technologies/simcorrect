from __future__ import annotations

import os
import tempfile
from pathlib import Path

OUTPUT_DIR_ENV = "SIMCORRECT_OUTPUT_DIR"


def output_dir(default: str | Path | None = None) -> Path:
    configured = os.environ.get(OUTPUT_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    if default is not None:
        return Path(default).expanduser()
    return Path(tempfile.gettempdir())


def output_path(filename: str | Path, default_dir: str | Path | None = None) -> Path:
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = output_dir(default_dir) / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
