from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
import xml.etree.ElementTree as ET


__all__ = ["CorrectionRecord", "Part"]


@dataclass(frozen=True)
class CorrectionRecord:
    part_name: str
    field: str
    old_value: object
    new_value: object
    elapsed_s: float
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    @classmethod
    def create(
        cls,
        part_name: str,
        field: str,
        old_value: object,
        new_value: object,
        elapsed_s: float,
    ) -> "CorrectionRecord":
        return cls(
            part_name=part_name,
            field=field,
            old_value=old_value,
            new_value=new_value,
            elapsed_s=elapsed_s,
        )

    def __repr__(self) -> str:
        return (
            f"CorrectionRecord(part={self.part_name!r}, "
            f"field={self.field!r}, "
            f"{self.old_value} -> {self.new_value}, "
            f"{self.elapsed_s * 1000:.1f}ms)"
        )


class Part:
    """Fluent helper for MJCF body mass and joint reference corrections."""

    def __init__(self, name: str, xml_source: str | None = None):
        self.name = name
        self.xml_source = xml_source
        self._mass: float | None = None
        self._ref: float | None = None
        self._corrections: list[CorrectionRecord] = []
        self._tree: ET.ElementTree | None = None
        self._root: ET.Element | None = None
        self._t0 = time.perf_counter()
        if xml_source and Path(xml_source).exists():
            self._tree = ET.parse(xml_source)
            self._root = self._tree.getroot()

    def set_mass(self, mass_kg: float) -> "Part":
        self._mass = float(mass_kg)
        return self

    def set_ref(self, ref_rad: float) -> "Part":
        self._ref = float(ref_rad)
        return self

    def export(self, output_path: str | Path) -> "Part":
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = str(path)
        elapsed = time.perf_counter() - self._t0
        if self._tree is not None and self._root is not None:
            self._apply_to_tree(elapsed)
            self._tree.write(target, encoding="unicode", xml_declaration=False)
        else:
            self._write_record_xml(target, elapsed)
        return self

    @property
    def corrections(self) -> list[CorrectionRecord]:
        return list(self._corrections)

    def report(self) -> str:
        lines = [f"OpenCAD correction - part: {self.name!r}"]
        if not self._corrections:
            lines.append("  No corrections recorded.")
        for record in self._corrections:
            lines.append(
                f"  {record.field}: {record.old_value} -> {record.new_value}"
                f"  ({record.elapsed_s * 1000:.1f}ms)"
            )
        return "\n".join(lines)

    def _apply_to_tree(self, elapsed: float) -> None:
        if self._root is None:
            return
        if self._mass is not None:
            self._apply_mass(elapsed)
        if self._ref is not None:
            self._apply_ref(elapsed)

    def _apply_mass(self, elapsed: float) -> None:
        if self._root is None or self._mass is None:
            return
        for body in self._root.iter("body"):
            if body.get("name") != self.name:
                continue
            inertial = body.find("inertial")
            if inertial is None:
                inertial = ET.SubElement(body, "inertial")
            old = inertial.get("mass", "unknown")
            inertial.set("mass", f"{self._mass:.6f}")
            self._corrections.append(
                CorrectionRecord.create(self.name, "inertial.mass", old, self._mass, elapsed)
            )
            return

    def _apply_ref(self, elapsed: float) -> None:
        if self._root is None or self._ref is None:
            return
        for joint in self._root.iter("joint"):
            if joint.get("name") != self.name:
                continue
            old = joint.get("ref", "0.0")
            joint.set("ref", f"{self._ref:.6f}")
            self._corrections.append(
                CorrectionRecord.create(self.name, "joint.ref", old, self._ref, elapsed)
            )
            return

    def _write_record_xml(self, output_path: str, elapsed: float) -> None:
        root = ET.Element("opencad_correction")
        root.set("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        root.set("elapsed_s", f"{elapsed:.4f}")
        part_el = ET.SubElement(root, "part")
        part_el.set("name", self.name)
        if self._mass is not None:
            correction = ET.SubElement(part_el, "correction")
            correction.set("field", "inertial.mass")
            correction.set("value", f"{self._mass:.6f}")
            correction.set("unit", "kg")
            self._corrections.append(
                CorrectionRecord.create(self.name, "inertial.mass", "unknown", self._mass, elapsed)
            )
        if self._ref is not None:
            correction = ET.SubElement(part_el, "correction")
            correction.set("field", "joint.ref")
            correction.set("value", f"{self._ref:.6f}")
            correction.set("unit", "rad")
            self._corrections.append(
                CorrectionRecord.create(self.name, "joint.ref", "unknown", self._ref, elapsed)
            )
        ET.ElementTree(root).write(output_path, encoding="unicode", xml_declaration=True)

    def __repr__(self) -> str:
        return f"Part({self.name!r}, mass={self._mass}, ref={self._ref})"
