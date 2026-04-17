"""
OpenCAD — Autonomous model correction API for SimCorrect.
The central abstraction between fault identification and deployment.
SimCorrect identifies what is wrong and by how much.
OpenCAD applies that correction to the simulation model.

Usage:
    from opencad import Part
    Part('grip').set_mass(0.160).export('grip_corrected.xml')
    Part('joint1').set_ref(0.0000).export('joint1_corrected.xml')
"""
import xml.etree.ElementTree as ET
import os, time


class CorrectionRecord:
    def __init__(self, part_name, field, old_value, new_value, elapsed_s):
        self.part_name = part_name
        self.field     = field
        self.old_value = old_value
        self.new_value = new_value
        self.elapsed_s = elapsed_s
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    def __repr__(self):
        return (f"CorrectionRecord(part={self.part_name!r}, "
                f"field={self.field!r}, "
                f"{self.old_value} -> {self.new_value}, "
                f"{self.elapsed_s*1000:.1f}ms)")


class Part:
    """
    Fluent interface for correcting a named body or joint in a MuJoCo MJCF model.
    """
    def __init__(self, name: str, xml_source: str = None):
        self.name         = name
        self.xml_source   = xml_source
        self._mass        = None
        self._ref         = None
        self._corrections = []
        self._tree        = None
        self._root        = None
        self._t0          = time.perf_counter()
        if xml_source and os.path.exists(xml_source):
            self._tree = ET.parse(xml_source)
            self._root = self._tree.getroot()

    def set_mass(self, mass_kg: float) -> "Part":
        self._mass = float(mass_kg)
        return self

    def set_ref(self, ref_rad: float) -> "Part":
        self._ref = float(ref_rad)
        return self

    def export(self, output_path: str) -> "Part":
        elapsed = time.perf_counter() - self._t0
        if self._root is not None:
            self._apply_to_tree(elapsed)
            self._tree.write(output_path, encoding="unicode", xml_declaration=False)
        else:
            self._write_record_xml(output_path, elapsed)
        return self

    def _apply_to_tree(self, elapsed):
        if self._mass is not None:
            for body in self._root.iter("body"):
                if body.get("name") == self.name:
                    inertial = body.find("inertial")
                    if inertial is None:
                        inertial = ET.SubElement(body, "inertial")
                    old = inertial.get("mass", "unknown")
                    inertial.set("mass", f"{self._mass:.6f}")
                    self._corrections.append(
                        CorrectionRecord(self.name, "inertial.mass",
                                         old, self._mass, elapsed))
                    break
        if self._ref is not None:
            for joint in self._root.iter("joint"):
                if joint.get("name") == self.name:
                    old = joint.get("ref", "0.0")
                    joint.set("ref", f"{self._ref:.6f}")
                    self._corrections.append(
                        CorrectionRecord(self.name, "joint.ref",
                                         old, self._ref, elapsed))
                    break

    def _write_record_xml(self, output_path, elapsed):
        root = ET.Element("opencad_correction")
        root.set("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        root.set("elapsed_s", f"{elapsed:.4f}")
        part_el = ET.SubElement(root, "part")
        part_el.set("name", self.name)
        if self._mass is not None:
            c = ET.SubElement(part_el, "correction")
            c.set("field", "inertial.mass")
            c.set("value", f"{self._mass:.6f}")
            c.set("unit", "kg")
            self._corrections.append(
                CorrectionRecord(self.name, "inertial.mass",
                                 "unknown", self._mass, elapsed))
        if self._ref is not None:
            c = ET.SubElement(part_el, "correction")
            c.set("field", "joint.ref")
            c.set("value", f"{self._ref:.6f}")
            c.set("unit", "rad")
            self._corrections.append(
                CorrectionRecord(self.name, "joint.ref",
                                 "unknown", self._ref, elapsed))
        ET.ElementTree(root).write(output_path, encoding="unicode",
                                   xml_declaration=True)

    def report(self) -> str:
        lines = [f"OpenCAD correction — part: {self.name!r}"]
        if not self._corrections:
            lines.append("  No corrections recorded.")
        for r in self._corrections:
            lines.append(f"  {r.field}: {r.old_value} -> {r.new_value}"
                         f"  ({r.elapsed_s*1000:.1f}ms)")
        return "\n".join(lines)

    @property
    def corrections(self):
        return list(self._corrections)

    def __repr__(self):
        return f"Part({self.name!r}, mass={self._mass}, ref={self._ref})"
