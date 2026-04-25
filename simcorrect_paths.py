from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ProblemPaths:
    problem_dir: Path

    @property
    def default_output_dir(self) -> Path:
        return self.problem_dir / "output"

    def output_dir(self) -> Path:
        return output_dir(self.default_output_dir)

    def output_path(self, filename: str | Path) -> Path:
        return output_path(filename, self.default_output_dir)

    def video_path(self, filename: str) -> Path:
        return self.output_path(filename)


def problem_paths(file: str | Path) -> ProblemPaths:
    return ProblemPaths(Path(file).resolve().parent)
