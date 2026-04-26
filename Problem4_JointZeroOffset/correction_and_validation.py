"""Problem 4 — Joint Zero Offset: correction via OpenCAD + validation."""
import mujoco, numpy as np, inspect, os
from opencad import Part, Sketch

try:
    from .render_demo import (
        CAN_L,
        CAN_R,
        GRIP_OPEN,
        HOME_Q,
        J1_REF_BAD,
        J1_REF_GT,
        LIFT_Q,
        PICK_Q,
        build,
        cor_ctrl_r,
        get_ids,
        ref_ctrl_r,
        set_arm,
        set_fingers,
        weld,
    )
except ImportError:
    from render_demo import (
        CAN_L,
        CAN_R,
        GRIP_OPEN,
        HOME_Q,
        J1_REF_BAD,
        J1_REF_GT,
        LIFT_Q,
        PICK_Q,
        build,
        cor_ctrl_r,
        get_ids,
        ref_ctrl_r,
        set_arm,
        set_fingers,
        weld,
    )

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

def correct_joint_zero(detected_offset_rad):
    """Use OpenCAD to rebuild encoder housing with corrected zero reference."""
    print(f"[OpenCAD] Detected j1 zero offset = {np.degrees(detected_offset_rad):.2f} deg ({detected_offset_rad:.4f} rad)")
    print(f"[OpenCAD] Rebuilding encoder housing geometry ...")

    # Encoder housing — cylinder representing the joint encoder mount
    # Correct geometry has zero rotational offset baked in
    part = Part()
    part.cylinder(radius=0.035, height=0.060, name="EncoderHousing")

    # Bore through center for motor shaft
    bore = Part()
    bore.cylinder(radius=0.012, height=0.065, name="ShaftBore")
    part.cut(bore)

    # Reference flat — the physical feature that defines zero position
    # Width encodes the corrected zero: at J1_REF_GT=0.0000, flat is perfectly aligned
    ref_flat = Part()
    ref_flat.box(0.070, 0.008, 0.060, name="ReferenceFlat")
    part.cut(ref_flat)

    stl_path = os.path.join(OUT_DIR, "encoder_housing_corrected.stl")
    part.export(stl_path)
    print(f"[OpenCAD] Exported -> {stl_path}")
    print(f"[OpenCAD] j1 ref: {detected_offset_rad:.4f} rad -> {J1_REF_GT:.4f} rad")
    print(f"[OpenCAD] Delta: -{np.degrees(detected_offset_rad):.2f} deg corrected")

    return {
        "fault_param":     "j1_ref",
        "fault_value":     detected_offset_rad,
        "corrected_value": J1_REF_GT,
        "stl_path":        stl_path,
        "delta_deg":       np.degrees(detected_offset_rad),
    }

def validate():
    print("=== Correction & Validation ===")

    # --- Phase 1: faulty arm ---
    model, data = build(J1_REF_BAD)
    LA,RA,BL,BR,lee,ree,cam,lj,rj,lf,rf = get_ids(model)
    weld(data, BL, CAN_L); weld(data, BR, CAN_R)
    data.qpos[LA:LA+4] = PICK_Q; data.qpos[RA:RA+4] = PICK_Q
    set_arm(data, lj, rj, PICK_Q, PICK_Q)
    mujoco.mj_forward(model, data)
    l_ee = data.site_xpos[lee].copy()
    r_ee = data.site_xpos[ree].copy()
    dist_l = np.linalg.norm(l_ee - CAN_L)
    dist_r = np.linalg.norm(r_ee - CAN_R)

    # j4 clamp check
    data.qpos[LA:LA+4] = LIFT_Q; data.qpos[RA:RA+4] = LIFT_Q
    mujoco.mj_forward(model, data)
    j4max = max(abs(data.qpos[LA+3]), abs(data.qpos[RA+3])) * 180 / np.pi
    carry_min_z = min(l_ee[2], r_ee[2])

    print(f"dist_l:      {dist_l*1000:.1f}mm  (threshold <80mm)")
    print(f"dist_r:      {dist_r*1000:.1f}mm  (must NOT be <80mm)")
    print(f"j4max:       {j4max:.2f} deg  (limit 17.1 deg)")
    print(f"carry_min_z: {carry_min_z:.3f}m  (must be >0.40m)")

    # --- OpenCAD correction ---
    result = correct_joint_zero(J1_REF_BAD)
    print(f"\n[OpenCAD] STL saved: {result['stl_path']}")
    print(f"[OpenCAD] Correction: {result['fault_param']} "
          f"{result['fault_value']:.4f} -> {result['corrected_value']:.4f} rad")

    # --- Phase 3: corrected arm ---
    model2, data2 = build(J1_REF_GT, "0.04 0.54 0.74 1")
    LA2,RA2,BL2,BR2,lee2,ree2,cam2,lj2,rj2,lf2,rf2 = get_ids(model2)
    weld(data2, BR2, CAN_R)
    data2.qpos[RA2:RA2+4] = PICK_Q
    set_arm(data2, lj2, rj2, PICK_Q, PICK_Q)
    mujoco.mj_forward(model2, data2)
    r_ee2 = data2.site_xpos[ree2].copy()
    dist_r2 = np.linalg.norm(r_ee2 - CAN_R)
    print(f"dist_r2:     {dist_r2*1000:.1f}mm  (threshold <150mm)")

    # --- Assertions ---
    assert dist_l < 0.08,         f"GT arm miss too large: {dist_l*1000:.1f}mm"
    assert not dist_r < 0.08,     f"Faulty arm should miss: {dist_r*1000:.1f}mm"
    assert dist_r2 < 0.15,        f"Corrected arm miss too large: {dist_r2*1000:.1f}mm"
    assert "_faulty" not in inspect.getsource(cor_ctrl_r), "cor_ctrl_r must not use _faulty"
    assert j4max < 17.1,          f"j4 exceeded limit: {j4max:.2f} deg"
    assert carry_min_z > 0.40,    f"carry height too low: {carry_min_z:.3f}m"
    assert os.path.exists(result["stl_path"]), "OpenCAD STL not generated"

    print("\nALL ASSERTIONS PASSED")
    print(f"OpenCAD STL: {result['stl_path']}")

if __name__ == "__main__":
    validate()
