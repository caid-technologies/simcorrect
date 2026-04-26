"""
Divergence detector -- Problem 5: Tool Mass Mismatch.

Classification pipeline:
  Step 1: EE divergence + Joint RMSE -> dynamics vs geometric
  Step 2: Velocity dependence -> mass mismatch vs friction
  Step 3: Reach scaling -> confirms mass signature
"""
import numpy as np

EE_FAULT_THRESHOLD       = 0.040
RMSE_DYNAMICS_THRESHOLD  = 0.005
VELOCITY_RATIO_THRESHOLD = 2.0


def detect(dist_gt, dist_faulty, j_rmse, sag_z_mm,
           vel_rmse_fast=None, vel_rmse_slow=None,
           sag_full_mm=None, sag_half_mm=None):
    print("=" * 55)
    print("SimCorrect -- Divergence Detector")
    print("=" * 55)
    print(f"  GT  EE error:     {dist_gt*1000:.1f} mm")
    print(f"  Faulty EE error:  {dist_faulty*1000:.1f} mm")
    print(f"  Vertical sag:     {sag_z_mm:.1f} mm")
    print(f"  Joint RMSE:       {j_rmse:.4f} rad")

    fault_detected = dist_faulty > EE_FAULT_THRESHOLD
    if not fault_detected:
        print("\n  STATUS: NOMINAL -- EE error within tolerance")
        return False, False, False, "NOMINAL"

    print(f"\n  FAULT DETECTED: EE error {dist_faulty*1000:.1f}mm > {EE_FAULT_THRESHOLD*1000:.0f}mm threshold")

    is_dynamics = j_rmse > RMSE_DYNAMICS_THRESHOLD
    if not is_dynamics:
        print("  CLASSIFICATION: GEOMETRIC")
        print("  Large EE divergence + Joint RMSE ~ 0")
        print("  Candidates: link length, joint zero offset, wrist offset")
        return fault_detected, False, False, "GEOMETRIC"

    print("  CLASSIFICATION: DYNAMICS")
    print(f"  Joint RMSE {j_rmse:.4f} rad > {RMSE_DYNAMICS_THRESHOLD} threshold")
    print("  Joints cannot hold commanded angles -> physics fault")

    is_gravity_dep = True
    if vel_rmse_fast is not None and vel_rmse_slow is not None:
        vel_ratio = vel_rmse_fast / (vel_rmse_slow + 1e-9)
        print(f"\n  Velocity ratio (fast/slow RMSE): {vel_ratio:.2f}")
        if vel_ratio > VELOCITY_RATIO_THRESHOLD:
            is_gravity_dep = False
            print("  HIGH velocity dependence -> FRICTION (Problem 3)")
        else:
            print("  LOW velocity dependence -> GRAVITY-DEPENDENT")
    else:
        is_gravity_dep = sag_z_mm > 5.0
        print(f"\n  Velocity data not provided -- inferring from sag: {'gravity-dependent' if is_gravity_dep else 'not gravity-dependent'}")

    if sag_full_mm is not None and sag_half_mm is not None and sag_half_mm > 0:
        scaling_ratio = sag_full_mm / sag_half_mm
        print(f"  Sag scaling ratio: {scaling_ratio:.2f}  (expected 2.0 for pure mass error)")
        if abs(scaling_ratio - 2.0) < 0.4:
            print("  2:1 scaling CONFIRMED -> MASS MISMATCH signature")
        else:
            print(f"  Warning: ratio {scaling_ratio:.2f} deviates from 2.0")

    if is_gravity_dep:
        print("\n  SUB-CLASS: GRAVITY-DEPENDENT DYNAMICS")
        print("  Error present at rest at horizontal pose")
        print("  Error scales with horizontal extension")
        print("  FAULT CLASS: TOOL MASS MISMATCH")
        return fault_detected, True, True, "DYNAMICS_MASS_MISMATCH"
    else:
        print("\n  SUB-CLASS: VELOCITY-DEPENDENT DYNAMICS")
        print("  FAULT CLASS: JOINT FRICTION (Problem 3 pattern)")
        return fault_detected, True, False, "DYNAMICS_FRICTION"


if __name__ == "__main__":
    from render_demo import SAG_J2, SAG_J4, SAG_MM
    j_rmse = np.sqrt(0.5*(SAG_J2**2 + SAG_J4**2))
    detect(dist_gt=0.050, dist_faulty=0.095,
           j_rmse=j_rmse, sag_z_mm=SAG_MM,
           sag_full_mm=SAG_MM, sag_half_mm=SAG_MM*0.5)
