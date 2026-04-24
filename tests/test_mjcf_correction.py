from __future__ import annotations

import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from mjcf_correction import Part
from opencad import Part as CompatPart


class MjcfCorrectionTests(unittest.TestCase):
    def test_explicit_module_writes_correction_record(self):
        with patch.object(ET.ElementTree, "write") as write:
            part = Part("grip").set_mass(0.160).export("mjcf_correction_test.xml")

        write.assert_called_once_with(
            "mjcf_correction_test.xml",
            encoding="unicode",
            xml_declaration=True,
        )

        self.assertEqual(part.corrections[0].field, "inertial.mass")
        self.assertEqual(part.corrections[0].new_value, 0.160)

    def test_legacy_opencad_import_still_exports_part(self):
        self.assertIs(CompatPart, Part)


if __name__ == "__main__":
    unittest.main()
