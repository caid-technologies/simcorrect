"""
Parameter identifier -- Problem 5: Tool Mass Mismatch.

Estimates actual tool mass from gravitational sag at two arm extensions.
The 2:1 linear scaling of sag with reach is the unique mathematical
signature of a pure mass error.

Physics:
    gravity_torque = delta_mass x g x reach
    joint_lag      = gravity_torque / kp
    delta_mass     = sag_mm x kp / (1000 x g x reach)

OpenCAD applies the correction to the MJCF model.
"""
import numpy as np

from paths import corrected_grip_xml_path
from opencad import Part

G          = 9.81
KP_J4      = 400.0
REACH_FULL = 0.75
REACH_HALF = 0.375


def identify(sag_full_mm, sag_half_mm, model_mass,
             reach_full=REACH_FULL, reach_half=REACH_HALF,
             export_path=None):
    if export_path is None:
        export_path = corrected_grip_xml_path()
    print("=" * 55)
    print("SimCorrect -- Parameter Identifier")
    print("=" * 55)
    print(f"  Sag at {reach_full:.3f}m:  {sag_full_mm:.1f} mm")
    print(f"  Sag at {reach_half:.3f}m:  {sag_half_mm:.1f} mm")

    sag_ratio = sag_full_mm / (sag_half_mm + 1e-9)
    print(f"  Sag ratio:        {sag_ratio:.2f}  (2.0 = pure mass error)")
    if abs(sag_ratio - 2.0) < 0.35:
        print("  Scaling CONFIRMED: mass mismatch signature")
    else:
        print(f"  Warning: ratio {sag_ratio:.2f} deviates -- mixed fault possible")

    dm_full     = (sag_full_mm / 1000.0) * KP_J4 / (G * reach_full)
    dm_half     = (sag_half_mm / 1000.0) * KP_J4 / (G * reach_half)
    delta_mass  = (dm_full + dm_half) / 2.0
    actual_mass = model_mass + delta_mass
    extra_torque = delta_mass * G * reach_full

    print(f"\n  Delta mass (full):  {dm_full*1000:.1f} g")
    print(f"  Delta mass (half):  {dm_half*1000:.1f} g")
    print(f"  ESTIMATED delta:    +{delta_mass*1000:.1f} g")
    print(f"  Modelled mass:      {model_mass:.3f} kg")
    print(f"  ACTUAL mass est.:   {actual_mass:.3f} kg")
    print(f"  Extra torque:       {extra_torque:.3f} Nm (was uncompensated)")

    print(f"\n  Applying OpenCAD correction...")
    part = Part("grip").set_mass(actual_mass)
    part.export(str(export_path))
    print(f"  {part.report()}")
    print(f"  Corrected XML written to: {export_path}")

    return actual_mass, delta_mass, part.corrections


if __name__ == "__main__":
    from render_demo import SAG_MM, MASS_MODEL
    identify(sag_full_mm=SAG_MM, sag_half_mm=SAG_MM*0.5, model_mass=MASS_MODEL)
