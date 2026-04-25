from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from simcorrect_mujoco import load_model_from_xml


class _FakeMjModel:
    last_path: str | None = None
    last_xml: str | None = None

    @classmethod
    def from_xml_path(cls, path: str) -> object:
        cls.last_path = path
        cls.last_xml = Path(path).read_text(encoding="utf-8")
        return {"loaded_from": path}


class MuJoCoHelperTests(unittest.TestCase):
    def setUp(self):
        _FakeMjModel.last_path = None
        _FakeMjModel.last_xml = None

    def test_load_model_from_xml_unlinks_temporary_file(self):
        with patch.dict(sys.modules, {"mujoco": SimpleNamespace(MjModel=_FakeMjModel)}):
            model = load_model_from_xml("<mujoco model='test'/>")

        self.assertEqual(model, {"loaded_from": _FakeMjModel.last_path})
        self.assertEqual(_FakeMjModel.last_xml, "<mujoco model='test'/>")
        self.assertFalse(Path(_FakeMjModel.last_path).exists())

    def test_load_model_from_xml_accepts_bytes(self):
        with patch.dict(sys.modules, {"mujoco": SimpleNamespace(MjModel=_FakeMjModel)}):
            load_model_from_xml(b"<mujoco model='bytes'/>")

        self.assertEqual(_FakeMjModel.last_xml, "<mujoco model='bytes'/>")


if __name__ == "__main__":
    unittest.main()
