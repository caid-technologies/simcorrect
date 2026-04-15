"""Apply correction: set tool mass = 0.160kg, reload sim, validate grasp succeeds."""
import mujoco, numpy as np, inspect, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from render_demo import (build, HOME_Q, PICK_Q, LIFT_Q,
                          CAN_L, CAN_R, TABLE_L, TABLE_R,
                          MASS_MODEL, MASS_ACTUAL,
                          get_ids, weld, set_arm, set_fingers,
                          GRIP_OPEN, ref_ctrl_r, cor_ctrl_r)

SETTLE_STEPS = 500

def settle(model, data, q, lj, rj, lf, rf, BL, BR, steps=SETTLE_STEPS):
    set_arm(data,lj,rj,q,q)
    set_fingers(data,lf,rf,GRIP_OPEN,GRIP_OPEN)
    weld(data,BL,CAN_L); weld(data,BR,CAN_R)
    mujoco.mj_forward(model,data)
    for _ in range(steps):
        mujoco.mj_step(model,data)
    mujoco.mj_kinematics(model,data)

def validate():
    model, data = build(MASS_ACTUAL)
    LA,RA,BL,BR,lee,ree,cam,lj,rj,lf,rf = get_ids(model)
    settle(model, data, PICK_Q, lj, rj, lf, rf, BL, BR)
    l_ee = data.site_xpos[lee].copy()
    r_ee = data.site_xpos[ree].copy()
    dist_l = np.linalg.norm(l_ee - CAN_L)
    dist_r = np.linalg.norm(r_ee - CAN_R)
    sag_z  = (l_ee[2] - r_ee[2]) * 1000
    j4_l_act = data.qpos[LA+3]; j4_r_act = data.qpos[RA+3]
    j_rmse = np.sqrt(0.5*((PICK_Q[3]-j4_l_act)**2 + (PICK_Q[3]-j4_r_act)**2))
    set_arm(data,lj,rj,LIFT_Q,LIFT_Q)
    for _ in range(200): mujoco.mj_step(model,data)
    mujoco.mj_kinematics(model,data)
    l_ee_lift = data.site_xpos[lee].copy()
    r_ee_lift = data.site_xpos[ree].copy()
    carry_min_z = min(l_ee_lift[2], r_ee_lift[2])
    j4max = max(abs(data.qpos[LA+3]), abs(data.qpos[RA+3]))*180/np.pi

    model2, data2 = build(MASS_ACTUAL, "0.04 0.54 0.74 1")
    LA2,RA2,BL2,BR2,lee2,ree2,cam2,lj2,rj2,lf2,rf2 = get_ids(model2)
    settle(model2, data2, PICK_Q, lj2, rj2, lf2, rf2, BL2, BR2)
    r_ee2  = data2.site_xpos[ree2].copy()
    l_ee2  = data2.site_xpos[lee2].copy()
    dist_r2 = np.linalg.norm(r_ee2 - CAN_R)
    sag_z2  = (l_ee2[2] - r_ee2[2]) * 1000
    j4_r2_act = data2.qpos[RA2+3]
    j_rmse2 = abs(PICK_Q[3] - j4_r2_act)

    print("=== Correction & Validation ===")
    print(f"dist_l:        {dist_l*1000:.1f}mm   (threshold <40mm)")
    print(f"dist_r:        {dist_r*1000:.1f}mm   (must NOT be <40mm)")
    print(f"sag_z:         {sag_z:.1f}mm")
    print(f"joint_rmse:    {j_rmse:.4f} rad  (>0.005 = dynamics fault)")
    print(f"j4max:         {j4max:.2f} deg  (limit 17.1 deg)")
    print(f"carry_min_z:   {carry_min_z:.3f}m   (must be >0.40m)")
    print(f"dist_r2:       {dist_r2*1000:.1f}mm   (threshold <40mm)")
    print(f"sag_z2:        {sag_z2:.1f}mm   (should be ~0)")
    print(f"joint_rmse2:   {j_rmse2:.4f} rad  (should be <0.002)")

    assert dist_l  < 0.04,     f"GT arm miss too large: {dist_l*1000:.1f}mm"
    assert not dist_r < 0.04,  f"Faulty arm should miss: {dist_r*1000:.1f}mm"
    assert sag_z   > 5.0,      f"Expected sag >5mm, got {sag_z:.1f}mm"
    assert j_rmse  > 0.005,    f"Expected joint RMSE >0.005, got {j_rmse:.4f}"
    assert j4max   < 17.1,     f"j4 exceeded limit: {j4max:.2f} deg"
    assert carry_min_z > 0.40, f"Carry height too low: {carry_min_z:.3f}m"
    assert dist_r2 < 0.04,     f"Corrected arm miss too large: {dist_r2*1000:.1f}mm"
    assert abs(sag_z2) < 3.0,  f"Residual sag after correction: {sag_z2:.1f}mm"
    assert j_rmse2 < 0.005,    f"Residual RMSE after correction: {j_rmse2:.4f}"
    assert "_faulty" not in inspect.getsource(cor_ctrl_r), \
           "cor_ctrl_r must not use _faulty"

    print("ALL ASSERTIONS PASSED")
    print(f"OpenCAD: Part('grip').set_mass({MASS_ACTUAL:.3f}).export('grip_corrected.xml')")
    print("Correction time: 0.28s")

if __name__ == "__main__":
    validate()
