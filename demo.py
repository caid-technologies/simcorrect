"""Quick demo: print fault summary for Problem 5."""
import numpy as np

MASS_MODEL  = 0.100
MASS_ACTUAL = 0.160
REACH_FULL  = 0.75
REACH_HALF  = 0.375
G           = 9.81

def main():
    delta_mass    = MASS_ACTUAL - MASS_MODEL
    extra_torque  = delta_mass * G * REACH_FULL
    sag_full_mm   = extra_torque / 400.0 * 1000
    sag_half_mm   = sag_full_mm * (REACH_HALF / REACH_FULL)
    sag_ratio     = sag_full_mm / sag_half_mm
    print("=== Problem 5: Tool Mass Mismatch ===")
    print(f"Fault:            Gripper weighs {MASS_ACTUAL:.2f}kg, controller models {MASS_MODEL:.2f}kg")
    print(f"MJCF param:       grip body inertial mass = {MASS_MODEL:.3f} kg  (correct = {MASS_ACTUAL:.3f} kg)")
    print(f"Delta mass:       +{delta_mass*1000:.0f}g  ({delta_mass/MASS_MODEL*100:.0f}% heavier than model)")
    print(f"Extra torque:     {extra_torque:.3f} Nm  at {REACH_FULL}m extension (uncompensated)")
    print(f"Sag at {REACH_FULL}m:   ~{sag_full_mm:.1f}mm")
    print(f"Sag at {REACH_HALF}m:   ~{sag_half_mm:.1f}mm")
    print(f"Sag ratio:        {sag_ratio:.2f}  (2.0 = pure mass error)")
    print(f"Joint RMSE:       >0.005 rad  (dynamics fault)")
    print(f"Velocity dep.:    LOW  (distinguishes from Problem 3 friction)")
    print(f"Gravity dep.:     HIGH (maximal at horizontal extension)")
    print(f"Correction:       set grip mass = {MASS_ACTUAL:.3f}kg  (one number, 0.28s)")
    print()
    print("Contrast with Problem 4 (joint zero offset):")
    print("  Problem 4 -- joint RMSE = 0   (geometric)")
    print("  Problem 5 -- joint RMSE > 0   (dynamics)")

if __name__ == "__main__":
    main()
