"""Detect EE divergence WITH non-zero joint RMSE -> dynamics fault classification."""
import numpy as np

RMSE_DYNAMICS_THRESHOLD = 0.005
EE_FAULT_THRESHOLD      = 0.040

def detect(dist_l, dist_r, j_rmse, sag_z,
           vel_rmse_fast=None, vel_rmse_slow=None,
           threshold_ee=EE_FAULT_THRESHOLD):
    print("=== Divergence Detector ===")
    print(f"GT  EE->can:      {dist_l*1000:.1f}mm")
    print(f"Faulty EE->can:   {dist_r*1000:.1f}mm")
    print(f"Vertical sag:     {sag_z:.1f}mm")
    print(f"Joint RMSE:       {j_rmse:.4f} rad")
    fault_detected = dist_r > threshold_ee
    is_dynamics    = fault_detected and j_rmse > RMSE_DYNAMICS_THRESHOLD
    if fault_detected:
        print(f"FAULT DETECTED: EE miss {dist_r*1000:.1f}mm > threshold {threshold_ee*1000:.0f}mm")
    else:
        print("No fault detected (EE miss within tolerance)")
        return False, False, False
    if is_dynamics:
        print("CLASSIFICATION: DYNAMICS (large Cartesian miss + joint RMSE > 0)")
        is_gravity_dep = True
        if vel_rmse_fast is not None and vel_rmse_slow is not None:
            vel_ratio = vel_rmse_fast / (vel_rmse_slow + 1e-9)
            is_gravity_dep = vel_ratio < 2.0
            print(f"Velocity ratio (fast/slow RMSE): {vel_ratio:.2f}")
        else:
            print("Velocity dependency: not measured (assuming gravity-dependent)")
        if is_gravity_dep:
            print("SUB-CLASS: GRAVITY-DEPENDENT DYNAMICS")
            print("Candidate: tool/link mass mismatch")
        else:
            print("SUB-CLASS: VELOCITY-DEPENDENT DYNAMICS")
            print("Candidate: joint friction excess (Problem 3 pattern)")
        return fault_detected, is_dynamics, is_gravity_dep
    else:
        print("CLASSIFICATION: GEOMETRIC (large Cartesian miss + zero joint RMSE)")
        return fault_detected, False, False

if __name__ == "__main__":
    detect(0.012, 0.078, 0.0082, 19.4)
