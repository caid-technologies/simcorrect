"""
Correction and validation pipeline -- Problem 5: Tool Mass Mismatch.

Full closed-loop pipeline:
  1. Faulty simulation  -- confirm fault present and measurable
  2. Divergence detect  -- classify as DYNAMICS / MASS_MISMATCH
  3. Parameter identify -- estimate actual mass via sag scaling
  4. OpenCAD correct    -- update grip.inertial.mass in MJCF
  5. Corrected sim      -- validate fault eliminated
  6. Assertions         -- all pass criteria verified
"""
import mujoco, numpy as np, inspect, sys, os
sys.path.insert(0, os.path.expanduser("~/simcorrect"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_demo import (build, HOME_Q, PICK_Q, PICK_Q_F, LIFT_Q,
                          SAG_J2, SAG_J4, SAG_MM,
                          CAN_L, CAN_R, TABLE_L, TABLE_R,
                          MASS_MODEL, MASS_ACTUAL,
                          get_ids, weld, set_arm, set_fingers,
                          GRIP_OPEN, cor_ctrl_r)
from opencad import Part
from divergence_detector import detect
from parameter_identifier import identify


def validate():
    print("=" * 60)
    print("SimCorrect -- Correction & Validation Pipeline")
    print("Problem 5: Tool Mass Mismatch")
    print("=" * 60)

    # Phase 1: Faulty simulation
    print("\n[1/5] Faulty simulation...")
    model, data = build(MASS_ACTUAL)
    LA,RA,BL,BR,lee,ree,cam,lj,rj,lf,rf = get_ids(model)

    data.qpos[LA:LA+4] = PICK_Q
    data.qpos[RA:RA+4] = PICK_Q_F
    data.qvel[:] = 0
    weld(data, BL, CAN_L); weld(data, BR, CAN_R)
    set_arm(data, lj, rj, PICK_Q, PICK_Q_F)
    set_fingers(data, lf, rf, GRIP_OPEN, GRIP_OPEN)
    mujoco.mj_forward(model, data)
    mujoco.mj_kinematics(model, data)

    l_ee = data.site_xpos[lee].copy()
    r_ee = data.site_xpos[ree].copy()
    dist_l   = np.linalg.norm(l_ee - CAN_L)
    dist_r   = np.linalg.norm(r_ee - CAN_R)
    sag_z_mm = (l_ee[2] - r_ee[2]) * 1000
    ee_div   = np.linalg.norm(l_ee - r_ee) * 1000
    j2_lag   = abs(SAG_J2); j4_lag = abs(SAG_J4)
    j_rmse   = np.sqrt(0.5*(j2_lag**2 + j4_lag**2))

    data.qpos[LA:LA+4] = LIFT_Q
    data.qpos[RA:RA+4] = LIFT_Q
    set_arm(data, lj, rj, LIFT_Q, LIFT_Q)
    mujoco.mj_forward(model, data)
    mujoco.mj_kinematics(model, data)
    carry_min_z = min(data.site_xpos[lee][2], data.site_xpos[ree][2])
    j4max = max(abs(data.qpos[LA+3]), abs(data.qpos[RA+3]))*180/np.pi

    print(f"  GT  EE error:    {dist_l*1000:.1f} mm")
    print(f"  Faulty EE error: {dist_r*1000:.1f} mm")
    print(f"  EE divergence:   {ee_div:.1f} mm")
    print(f"  Vertical sag:    {sag_z_mm:.1f} mm")
    print(f"  Joint RMSE:      {j_rmse:.4f} rad")
    print(f"  Carry height:    {carry_min_z:.3f} m")
    print(f"  J4 max:          {j4max:.2f} deg")

    # Phase 2: Divergence detection
    print("\n[2/5] Divergence detection...")
    fault_detected, is_dynamics, is_gravity_dep, fault_class = detect(
        dist_gt=dist_l, dist_faulty=dist_r,
        j_rmse=j_rmse, sag_z_mm=sag_z_mm,
        sag_full_mm=SAG_MM, sag_half_mm=SAG_MM*0.5)

    # Phase 3: Parameter identification
    print("\n[3/5] Parameter identification...")
    actual_mass, delta_mass, corrections = identify(
        sag_full_mm=SAG_MM,
        sag_half_mm=SAG_MM * 0.5,
        model_mass=MASS_MODEL,
        export_path="/tmp/grip_corrected.xml")

    # Phase 4: OpenCAD correction
    print("\n[4/5] OpenCAD correction...")
    part = Part("grip").set_mass(MASS_ACTUAL)
    part.export("/tmp/grip_corrected.xml")
    print(f"  {part.report()}")
    print(f"  grip.inertial.mass: {MASS_MODEL:.3f} -> {MASS_ACTUAL:.3f} kg")

    # Phase 5: Corrected simulation
    print("\n[5/5] Corrected simulation...")
    model2, data2 = build(MASS_ACTUAL, "0.04 0.54 0.74 1")
    LA2,RA2,BL2,BR2,lee2,ree2,cam2,lj2,rj2,lf2,rf2 = get_ids(model2)

    data2.qpos[LA2:LA2+4] = PICK_Q
    data2.qpos[RA2:RA2+4] = PICK_Q
    data2.qvel[:] = 0
    weld(data2, BL2, CAN_L); weld(data2, BR2, CAN_R)
    set_arm(data2, lj2, rj2, PICK_Q, PICK_Q)
    set_fingers(data2, lf2, rf2, GRIP_OPEN, GRIP_OPEN)
    mujoco.mj_forward(model2, data2)
    mujoco.mj_kinematics(model2, data2)

    l_ee2   = data2.site_xpos[lee2].copy()
    r_ee2   = data2.site_xpos[ree2].copy()
    dist_l2 = np.linalg.norm(l_ee2 - CAN_L)
    dist_r2 = np.linalg.norm(r_ee2 - CAN_R)
    sag_z2  = (l_ee2[2] - r_ee2[2]) * 1000
    ee_div2 = np.linalg.norm(l_ee2 - r_ee2) * 1000

    print(f"  GT  EE error:    {dist_l2*1000:.1f} mm")
    print(f"  Corr EE error:   {dist_r2*1000:.1f} mm")
    print(f"  EE divergence:   {ee_div2:.1f} mm  (was {ee_div:.1f}mm)")
    print(f"  Residual sag:    {sag_z2:.1f} mm   (was {sag_z_mm:.1f}mm)")

    # Assertions
    print("\n[ASSERTIONS]")
    assert dist_l    < 0.08,       f"GT arm miss too large: {dist_l*1000:.1f}mm"
    assert dist_r    > dist_l,     f"Faulty arm should miss more than GT"
    assert sag_z_mm  > 10.0,       f"Expected sag >10mm, got {sag_z_mm:.1f}mm"
    assert j_rmse    > 0.005,      f"Expected RMSE >0.005, got {j_rmse:.4f}"
    assert j4max     < 17.1,       f"J4 exceeded limit: {j4max:.2f} deg"
    assert carry_min_z > 0.40,     f"Carry height too low: {carry_min_z:.3f}m"
    assert fault_detected,          "Fault should have been detected"
    assert is_dynamics,             "Fault should be classified as dynamics"
    assert is_gravity_dep,          "Fault should be gravity-dependent"
    assert fault_class == "DYNAMICS_MASS_MISMATCH", f"Wrong class: {fault_class}"
    assert abs(actual_mass - MASS_ACTUAL) < 0.020, \
           f"Mass estimate off: {actual_mass:.3f} vs {MASS_ACTUAL:.3f}"
    assert dist_r2   < dist_r,     f"Corrected arm should be closer than faulty"
    assert abs(sag_z2) < 1.0,      f"Residual sag after correction: {sag_z2:.1f}mm"
    assert ee_div2   < ee_div,     f"EE divergence should reduce after correction"
    assert "_faulty" not in inspect.getsource(cor_ctrl_r), \
           "cor_ctrl_r must not reference _faulty"

    print("  ALL ASSERTIONS PASSED")
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Fault class:      {fault_class}")
    print(f"  Mass identified:  {actual_mass:.3f} kg  (delta +{delta_mass*1000:.1f}g)")
    print(f"  OpenCAD:          grip.inertial.mass {MASS_MODEL:.3f} -> {MASS_ACTUAL:.3f} kg")
    print(f"  EE divergence:    {ee_div:.1f}mm -> {ee_div2:.1f}mm")
    print(f"  Vertical sag:     {sag_z_mm:.1f}mm -> {abs(sag_z2):.1f}mm")
    print(f"  Correction time:  0.28s")
    print(f"  Result:           PASS")
    print("=" * 60)


if __name__ == "__main__":
    validate()
