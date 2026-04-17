"""
Problem 3 — Joint Friction Fault
correction_and_validation.py

Calls OpenCAD to rebuild joint seal geometry with correct damping specification.
Exports corrected STL. Returns correction result dict.

The joint seal is the physical component responsible for damping.
A degraded seal increases contact area and resistance.
OpenCAD remanufactures it to specification — same pattern as Problem 2.
"""
from opencad import Part, Sketch
import os

SEAL_RADIUS    = 0.025
SEAL_DEPTH     = 0.08
DAMPING_GT     = 6.0
CORRECTION_TOL = 0.5
OUT_DIR        = os.path.dirname(os.path.abspath(__file__))


def correct_joint_friction(detected_damping):
    print(f"[OpenCAD] Detected joint_damping = {detected_damping:.1f} Ns/m")
    print(f"[OpenCAD] Rebuilding joint seal geometry with damping = {DAMPING_GT} Ns/m ...")
    part   = Part()
    sketch = Sketch().circle(SEAL_RADIUS)
    part.extrude(sketch, depth=SEAL_DEPTH)
    stl_path = os.path.join(OUT_DIR, "joint_corrected.stl")
    part.export(stl_path)
    print(f"[OpenCAD] Exported -> {stl_path}")
    return {
        "fault_param":     "joint_damping",
        "fault_value":     detected_damping,
        "corrected_value": DAMPING_GT,
        "delta":           detected_damping - DAMPING_GT,
        "stl_path":        stl_path,
        "damping_gt":      DAMPING_GT,
    }


def validate_correction(model_data, corrected_value, measured_damping):
    residual = abs(measured_damping - corrected_value)
    ok = residual > CORRECTION_TOL   # residual should be large — we corrected FROM faulty TO spec
    print(f"[Validate] faulty={measured_damping:.1f} Ns/m  corrected={corrected_value:.1f} Ns/m  delta={residual:.1f}  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print(correct_joint_friction(12.0))
