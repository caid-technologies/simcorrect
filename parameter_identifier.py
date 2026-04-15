"""Identify tool mass from gravitational sag at two arm extensions."""
import numpy as np

G          = 9.81
KP_J4      = 400.0
REACH_FULL = 0.75
REACH_HALF = 0.375

def identify(sag_full_mm, sag_half_mm, model_mass,
             reach_full=REACH_FULL, reach_half=REACH_HALF):
    print("=== Parameter Identifier ===")
    print(f"Sag at {reach_full:.3f}m reach:  {sag_full_mm:.1f}mm")
    print(f"Sag at {reach_half:.3f}m reach:  {sag_half_mm:.1f}mm")
    sag_ratio = sag_full_mm / (sag_half_mm + 1e-9)
    print(f"Sag ratio:                {sag_ratio:.2f}  (expected ~2.0 for pure mass error)")
    sag_full_m = sag_full_mm / 1000.0
    sag_half_m = sag_half_mm / 1000.0
    dm_from_full = sag_full_m * KP_J4 / (G * reach_full)
    dm_from_half = sag_half_m * KP_J4 / (G * reach_half)
    delta_mass   = (dm_from_full + dm_from_half) / 2.0
    actual_mass  = model_mass + delta_mass
    extra_torque = delta_mass * G * reach_full
    print(f"delta_mass (from full):   {dm_from_full*1000:.1f}g")
    print(f"delta_mass (from half):   {dm_from_half*1000:.1f}g")
    print(f"IDENTIFIED delta_mass:    +{delta_mass*1000:.1f}g")
    print(f"Modelled tool mass:        {model_mass:.3f} kg")
    print(f"ESTIMATED actual mass:     {actual_mass:.3f} kg")
    print(f"Extra torque at {reach_full}m:   {extra_torque:.3f} Nm (was uncompensated)")
    print(f"MJCF field to correct:     grip body inertial mass  {model_mass:.3f} -> {actual_mass:.3f}")
    return actual_mass, delta_mass

if __name__ == "__main__":
    identify(sag_full_mm=19.4, sag_half_mm=9.7, model_mass=0.100)
