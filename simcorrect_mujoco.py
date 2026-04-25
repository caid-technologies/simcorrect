from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any


def load_model_from_xml(xml: str | bytes) -> Any:
    """Load a MuJoCo model from XML text and remove the temporary file."""
    binary = isinstance(xml, bytes)
    mode = "wb" if binary else "w"
    path: str | None = None
    kwargs = {} if binary else {"encoding": "utf-8"}
    try:
        with tempfile.NamedTemporaryFile(
            mode=mode,
            suffix=".xml",
            delete=False,
            **kwargs,
        ) as file:
            file.write(xml)
            path = file.name

        import mujoco

        return mujoco.MjModel.from_xml_path(path)
    finally:
        if path is not None:
            Path(path).unlink(missing_ok=True)
