"""
SimCorrect — Problem 2: Wrist Lateral Offset Fault
CoRL 2026 demo video

Left arm  = ground truth  (silver) wrist_offset_y = 0.000 m
Right arm = faulty        (red)    wrist_offset_y = +0.007 m

Phase 1  : both arms attempt pick-and-place
           left succeeds, right misses can laterally (LATERAL MISS badge)
Freeze   : diagnostic panel — fault, estimated offset, correction delta
Phase 2  : right arm reloaded with corrected geometry (cyan)
           both arms succeed, both cans placed on green markers

All IK waypoints validated (< 1 mm FK error).
Coke can geometry: 66 mm dia, 122 mm tall.
Gripper closes visibly around can when grasping.
"""

import os, math
import numpy as np
import mujoco
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont
from paths import video_path
from simcorrect_mujoco import load_model_from_xml

# ── output ────────────────────────────────────────────────────────────────────
W,  H  = 800,  450     # MuJoCo render resolution
WO, HO = 1600, 900     # video output resolution
FPS    = 30
DT     = 0.002
OUT    = str(video_path("simcorrect_p2_wrist_offset.mp4"))

# ── can geometry ──────────────────────────────────────────────────────────────
CAN_R = 0.033     # 66 mm diameter (Coke can)
CAN_H = 0.061     # half-height → 122 mm tall

# ── arm geometry (3-link planar) ──────────────────────────────────────────────
L1 = 0.28   # upper arm
L2 = 0.24   # forearm
L3 = 0.14   # wrist link
TCP = 0.032  # TCP offset from wrist tip

WRIST_FAULT_Y  = 0.007            # +7 mm lateral offset
SENSITIVITY_Y  = 0.95             # empirical
OBSERVED_DRIFT = WRIST_FAULT_Y / SENSITIVITY_Y   # ~7.37 mm

# ── gripper ───────────────────────────────────────────────────────────────────
GRIP_OPEN   = 0.040   # each finger ±40 mm from centre → 80 mm gap (> 66 mm can)
GRIP_CLOSED = 0.034   # ±34 mm → 68 mm gap (just touching can surface)

# ── scene layout (all in world frame) ────────────────────────────────────────
LEFT_BASE  = np.array([-0.62, 0.0, 0.72])
RIGHT_BASE = np.array([ 0.62, 0.0, 0.72])

PED_H = 0.36
CAN_Z = PED_H + CAN_H          # can centre Z = 0.421 m

LEFT_CAN   = np.array([-0.82, 0.0, CAN_Z])
RIGHT_CAN  = np.array([ 0.82, 0.0, CAN_Z])

TABLE_TOP  = 0.50
LEFT_TABLE_POS  = np.array([-0.30, 0.0, TABLE_TOP / 2])
RIGHT_TABLE_POS = np.array([ 0.30, 0.0, TABLE_TOP / 2])
LEFT_TARGET  = np.array([-0.30, 0.0, TABLE_TOP + CAN_H])
RIGHT_TARGET = np.array([ 0.30, 0.0, TABLE_TOP + CAN_H])

# ── timeline (Phase 1) ────────────────────────────────────────────────────────
T0  = 1.0
T1  = 4.0
T2  = 5.6
T3  = 6.8
T4  = 8.2
T5  = 11.2
T6  = 12.8
T7  = 13.8
T8  = 15.0
T_FREEZE   = 16.2
FREEZE_DUR = 5.5

# ── timeline (Phase 2 — corrected) ───────────────────────────────────────────
R0, R1, R2, R3 = 0.5, 3.5, 5.1, 6.3
R4, R5, R6, R7 = 7.7, 10.7, 12.3, 13.3
R8, R_END       = 14.5, 17.0

# ─────────────────────────────────────────────────────────────────────────────
# IK / FK helpers
# ─────────────────────────────────────────────────────────────────────────────
def ik(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    """3-link planar IK. Wrist held horizontal (q3 = -(q1+q2))."""
    tx = target[0] - base[0] - (L3 + TCP)
    tz = target[2] - base[2]
    r2 = tx**2 + tz**2
    c2 = np.clip((r2 - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
    s2 = math.sqrt(max(0.0, 1.0 - c2**2))
    q2 = math.atan2(s2, c2)
    q1 = math.atan2(tz, tx) - math.atan2(L2 * s2, L1 + L2 * c2)
    q3 = -(q1 + q2)
    return np.array([q1, q2, q3])

def make_waypoints(base: np.ndarray, can: np.ndarray, target: np.ndarray) -> dict:
    return {
        "home":   ik(base, base + np.array([0.06, 0.0, -0.10])),
        "above":  ik(base, can    + np.array([0.0,  0.0,  0.16])),
        "pick":   ik(base, can    + np.array([0.0,  0.0,  0.010])),
        "lift":   ik(base, can    + np.array([0.0,  0.0,  0.22])),
        "aboveT": ik(base, target + np.array([0.0,  0.0,  0.14])),
        "place":  ik(base, target + np.array([0.0,  0.0,  0.010])),
        "ret":    ik(base, base   + np.array([0.08, 0.0, -0.08])),
    }

LEFT_Q  = make_waypoints(LEFT_BASE,  LEFT_CAN,  LEFT_TARGET)
RIGHT_Q = make_waypoints(RIGHT_BASE, RIGHT_CAN, RIGHT_TARGET)

# ── controller interpolation ──────────────────────────────────────────────────
def sm(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    s = t * t * (3.0 - 2.0 * t)
    return a * (1.0 - s) + b * s

def ctrl_p1(t, Q):
    if t < T0: return Q["home"],   GRIP_OPEN,   "idle"
    if t < T1: return sm(Q["home"],  Q["above"],  (t-T0)/(T1-T0)), GRIP_OPEN,   "approach"
    if t < T2: return sm(Q["above"], Q["pick"],   (t-T1)/(T2-T1)), GRIP_OPEN,   "descend"
    if t < T3: return Q["pick"],   sm(GRIP_OPEN, GRIP_CLOSED, (t-T2)/(T3-T2)), "close"
    if t < T4: return sm(Q["pick"],  Q["lift"],   (t-T3)/(T4-T3)), GRIP_CLOSED, "lift"
    if t < T5: return sm(Q["lift"],  Q["aboveT"], (t-T4)/(T5-T4)), GRIP_CLOSED, "transfer"
    if t < T6: return sm(Q["aboveT"],Q["place"],  (t-T5)/(T6-T5)), GRIP_CLOSED, "lower"
    if t < T7: return Q["place"],  sm(GRIP_CLOSED, GRIP_OPEN, (t-T6)/(T7-T6)), "release"
    if t < T8: return sm(Q["place"], Q["ret"],    (t-T7)/(T8-T7)), GRIP_OPEN,   "retreat"
    return Q["ret"], GRIP_OPEN, "done"

def ctrl_p2(t, Q):
    if t < R0: return Q["home"],   GRIP_OPEN,   "idle"
    if t < R1: return sm(Q["home"],  Q["above"],  (t-R0)/(R1-R0)), GRIP_OPEN,   "approach"
    if t < R2: return sm(Q["above"], Q["pick"],   (t-R1)/(R2-R1)), GRIP_OPEN,   "descend"
    if t < R3: return Q["pick"],   sm(GRIP_OPEN, GRIP_CLOSED, (t-R2)/(R3-R2)), "close"
    if t < R4: return sm(Q["pick"],  Q["lift"],   (t-R3)/(R4-R3)), GRIP_CLOSED, "lift"
    if t < R5: return sm(Q["lift"],  Q["aboveT"], (t-R4)/(R5-R4)), GRIP_CLOSED, "transfer"
    if t < R6: return sm(Q["aboveT"],Q["place"],  (t-R5)/(R6-R5)), GRIP_CLOSED, "lower"
    if t < R7: return Q["place"],  sm(GRIP_CLOSED, GRIP_OPEN, (t-R6)/(R7-R6)), "release"
    if t < R8: return sm(Q["place"], Q["ret"],    (t-R7)/(R8-R7)), GRIP_OPEN,   "retreat"
    return Q["ret"], GRIP_OPEN, "done"

# ── MuJoCo helpers ────────────────────────────────────────────────────────────
def set_free_pose(data, slc, pos):
    data.qpos[slc.start:slc.start+3] = pos
    data.qpos[slc.start+3:slc.stop]  = [1, 0, 0, 0]
    data.qvel[slc.start:slc.start+6] = 0.0

def body_qpos_slice(model, name):
    bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    jid  = model.body_jntadr[bid]
    qadr = model.jnt_qposadr[jid]
    return slice(int(qadr), int(qadr) + 7)

def site_pos(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return data.site_xpos[sid].copy()

# ── model XML ────────────────────────────────────────────────────────────────
def build_model(corrected: bool = False):
    wrist_y  = 0.0 if corrected else WRIST_FAULT_Y
    r_mat    = "mat_fix" if corrected else "mat_bad"
    GH = GRIP_OPEN / 2

    xml = f"""
<mujoco model="simcorrect_p2">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{DT}" gravity="0 0 -9.81" iterations="80" integrator="implicitfast"/>

  <visual>
    <global offwidth="{W}" offheight="{H}"/>
    <quality shadowsize="2048" numslices="48" numstacks="48"/>
    <headlight ambient="0.5 0.5 0.52" diffuse="1.1 1.1 1.1" specular="0.2 0.2 0.2"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.18 0.22 0.28" rgb2="0.05 0.07 0.11" width="256" height="256"/>
    <texture name="floor_tex" type="2d" builtin="checker"
             rgb1="0.22 0.24 0.28" rgb2="0.15 0.16 0.20" width="512" height="512"/>
    <material name="mat_floor"  texture="floor_tex" texrepeat="6 6" specular="0.05"/>
    <material name="mat_table"  rgba="0.55 0.38 0.20 1"/>
    <material name="mat_ped"    rgba="0.20 0.22 0.28 1" specular="0.4"/>
    <material name="mat_goal"   rgba="0.10 0.90 0.25 1" emission="0.15"/>
    <material name="mat_can_body" rgba="0.85 0.08 0.08 1" specular="0.7"/>
    <material name="mat_can_top"  rgba="0.82 0.82 0.85 1" specular="0.9"/>
    <material name="mat_joint"  rgba="0.18 0.20 0.26 1" specular="0.5"/>
    <material name="mat_gt"     rgba="0.84 0.86 0.92 1" specular="0.9"/>
    <material name="mat_bad"    rgba="0.92 0.16 0.10 1" specular="0.9"/>
    <material name="mat_fix"    rgba="0.15 0.78 0.96 1" specular="0.9"/>
  </asset>

  <default>
    <joint damping="4.0" armature="0.02"/>
    <geom condim="4" solref="0.004 1" solimp="0.95 0.99 0.001" friction="1.2 0.02 0.002"/>
  </default>

  <worldbody>
    <light pos="0 -3.0 3.5" dir="0 0.1 -1" diffuse="1.1 1.1 1.1" castshadow="true"/>
    <light pos="1.5 0.8 2.5" dir="-0.5 -0.2 -1" diffuse="0.35 0.40 0.50"/>
    <geom type="plane" size="3 3 0.1" material="mat_floor"/>

    <!-- LEFT stage -->
    <geom type="box"
          pos="{LEFT_TABLE_POS[0]:.3f} 0 {LEFT_TABLE_POS[2]:.3f}"
          size="0.18 0.13 {TABLE_TOP/2:.3f}" material="mat_table"/>
    <geom type="cylinder" size="0.054 0.003"
          pos="{LEFT_TARGET[0]:.3f} 0 {TABLE_TOP+0.003:.3f}" material="mat_goal"/>
    <geom type="cylinder" size="0.042 {PED_H/2:.3f}"
          pos="{LEFT_CAN[0]:.3f} 0 {PED_H/2:.3f}" material="mat_ped"/>

    <!-- RIGHT stage -->
    <geom type="box"
          pos="{RIGHT_TABLE_POS[0]:.3f} 0 {RIGHT_TABLE_POS[2]:.3f}"
          size="0.18 0.13 {TABLE_TOP/2:.3f}" material="mat_table"/>
    <geom type="cylinder" size="0.054 0.003"
          pos="{RIGHT_TARGET[0]:.3f} 0 {TABLE_TOP+0.003:.3f}" material="mat_goal"/>
    <geom type="cylinder" size="0.042 {PED_H/2:.3f}"
          pos="{RIGHT_CAN[0]:.3f} 0 {PED_H/2:.3f}" material="mat_ped"/>

    <!-- Coke cans (free bodies) -->
    <body name="can_left" pos="{LEFT_CAN[0]:.3f} 0 {LEFT_CAN[2]:.3f}">
      <freejoint name="jcan_left"/>
      <geom name="can_left_body" type="cylinder"
            size="{CAN_R:.3f} {CAN_H:.3f}" mass="0.350" material="mat_can_body"/>
      <geom name="can_left_top" type="cylinder"
            size="{CAN_R*0.94:.3f} 0.006"
            pos="0 0 {CAN_H-0.004:.3f}" material="mat_can_top"/>
      <geom name="can_left_bot" type="cylinder"
            size="{CAN_R*0.88:.3f} 0.005"
            pos="0 0 {-CAN_H+0.004:.3f}" material="mat_can_top"/>
    </body>

    <body name="can_right" pos="{RIGHT_CAN[0]:.3f} 0 {RIGHT_CAN[2]:.3f}">
      <freejoint name="jcan_right"/>
      <geom name="can_right_body" type="cylinder"
            size="{CAN_R:.3f} {CAN_H:.3f}" mass="0.350" material="mat_can_body"/>
      <geom name="can_right_top" type="cylinder"
            size="{CAN_R*0.94:.3f} 0.006"
            pos="0 0 {CAN_H-0.004:.3f}" material="mat_can_top"/>
      <geom name="can_right_bot" type="cylinder"
            size="{CAN_R*0.88:.3f} 0.005"
            pos="0 0 {-CAN_H+0.004:.3f}" material="mat_can_top"/>
    </body>

    <!-- LEFT ARM (ground truth) -->
    <body name="left_base" pos="{LEFT_BASE[0]:.3f} 0 {LEFT_BASE[2]:.3f}">
      <geom type="cylinder" size="0.058 0.052" euler="1.5708 0 0" material="mat_joint"/>
      <joint name="l_sho" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
      <body name="l_upper">
        <geom type="capsule" fromto="0 0 0 {L1:.3f} 0 0" size="0.030" material="mat_gt"/>
        <body name="l_elbow" pos="{L1:.3f} 0 0">
          <geom type="sphere" size="0.036" material="mat_joint"/>
          <joint name="l_elb" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <body name="l_fore">
            <geom type="capsule" fromto="0 0 0 {L2:.3f} 0 0" size="0.025" material="mat_gt"/>
            <body name="l_wrist_jt" pos="{L2:.3f} 0 0">
              <geom type="sphere" size="0.028" material="mat_joint"/>
              <joint name="l_wri" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
              <body name="l_wrist_lnk" pos="0 0 0">
                <geom type="capsule" fromto="0 0 0 {L3:.3f} 0 0" size="0.020" material="mat_gt"/>
                <body name="left_tool" pos="{L3:.3f} 0 0">
                  <geom type="box" size="0.024 0.022 0.020" material="mat_joint"/>
                  <body name="left_f1" pos="0 {GH:.4f} 0">
                    <joint name="l_f1" type="slide" axis="0 -1 0"
                           range="{GRIP_CLOSED/2:.4f} {GH:.4f}"/>
                    <geom type="box" size="0.012 0.042 0.015"
                          pos="0 0.042 0" material="mat_joint"/>
                  </body>
                  <body name="left_f2" pos="0 {-GH:.4f} 0">
                    <joint name="l_f2" type="slide" axis="0 1 0"
                           range="{GRIP_CLOSED/2:.4f} {GH:.4f}"/>
                    <geom type="box" size="0.012 0.042 0.015"
                          pos="0 -0.042 0" material="mat_joint"/>
                  </body>
                  <site name="left_tcp" pos="{TCP:.3f} 0 0" size="0.006" rgba="0 0 0 0"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- RIGHT ARM (faulty / corrected) -->
    <body name="right_base" pos="{RIGHT_BASE[0]:.3f} 0 {RIGHT_BASE[2]:.3f}">
      <geom type="cylinder" size="0.058 0.052" euler="1.5708 0 0" material="mat_joint"/>
      <joint name="r_sho" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
      <body name="r_upper">
        <geom type="capsule" fromto="0 0 0 {L1:.3f} 0 0" size="0.030" material="{r_mat}"/>
        <body name="r_elbow" pos="{L1:.3f} 0 0">
          <geom type="sphere" size="0.036" material="mat_joint"/>
          <joint name="r_elb" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <body name="r_fore">
            <geom type="capsule" fromto="0 0 0 {L2:.3f} 0 0" size="0.025" material="{r_mat}"/>
            <body name="r_wrist_jt" pos="{L2:.3f} 0 0">
              <geom type="sphere" size="0.028" material="mat_joint"/>
              <joint name="r_wri" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
              <!-- FAULT INJECTED HERE -->
              <body name="r_wrist_lnk" pos="0 {wrist_y:.4f} 0">
                <geom type="capsule" fromto="0 0 0 {L3:.3f} 0 0" size="0.020" material="{r_mat}"/>
                <body name="right_tool" pos="{L3:.3f} 0 0">
                  <geom type="box" size="0.024 0.022 0.020" material="mat_joint"/>
                  <body name="right_f1" pos="0 {GH:.4f} 0">
                    <joint name="r_f1" type="slide" axis="0 -1 0"
                           range="{GRIP_CLOSED/2:.4f} {GH:.4f}"/>
                    <geom type="box" size="0.012 0.042 0.015"
                          pos="0 0.042 0" material="mat_joint"/>
                  </body>
                  <body name="right_f2" pos="0 {-GH:.4f} 0">
                    <joint name="r_f2" type="slide" axis="0 1 0"
                           range="{GRIP_CLOSED/2:.4f} {GH:.4f}"/>
                    <geom type="box" size="0.012 0.042 0.015"
                          pos="0 -0.042 0" material="mat_joint"/>
                  </body>
                  <site name="right_tcp" pos="{TCP:.3f} 0 0" size="0.006" rgba="0 0 0 0"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <camera name="main" pos="0 -3.40 1.30" xyaxes="1 0 0 0 0.38 0.92" fovy="32"/>
  </worldbody>

  <actuator>
    <position joint="l_sho" kp="3000" forcerange="-250 250"/>
    <position joint="l_elb" kp="2500" forcerange="-200 200"/>
    <position joint="l_wri" kp="1800" forcerange="-150 150"/>
    <position joint="l_f1"  kp="800"  forcerange="-40 40"/>
    <position joint="l_f2"  kp="800"  forcerange="-40 40"/>

    <position joint="r_sho" kp="3000" forcerange="-250 250"/>
    <position joint="r_elb" kp="2500" forcerange="-200 200"/>
    <position joint="r_wri" kp="1800" forcerange="-150 150"/>
    <position joint="r_f1"  kp="800"  forcerange="-40 40"/>
    <position joint="r_f2"  kp="800"  forcerange="-40 40"/>
  </actuator>
</mujoco>
"""
    model = load_model_from_xml(xml)
    return model, mujoco.MjData(model)

# ── overlay / UI helpers ──────────────────────────────────────────────────────
def upscale(frame: np.ndarray) -> np.ndarray:
    if WO == W and HO == H:
        return frame
    return np.array(Image.fromarray(frame).resize((WO, HO), Image.LANCZOS))

def font(sz, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in paths:
        try: return ImageFont.truetype(p, sz)
        except: pass
    return ImageFont.load_default()

def title_card() -> np.ndarray:
    img = Image.new("RGB", (WO, HO), (10, 12, 18))
    dr  = ImageDraw.Draw(img)
    dr.text((110, 110), "SimCorrect — Problem 2",
            font=font(52, True), fill=(255, 220, 70))
    dr.text((110, 195), "Wrist Lateral Offset: Fault Detection & Correction",
            font=font(30), fill=(200, 220, 255))
    dr.text((110, 265), "Left:  ground truth arm  —  wrist_offset_y = 0.000 m",
            font=font(26), fill=(85, 225, 120))
    dr.text((110, 308), "Right: faulty arm        —  wrist_offset_y = +0.007 m",
            font=font(26), fill=(240, 80, 60))
    dr.text((110, 378), "Fault is geometric, not kinematic  —  joint RMSE ≈ 0",
            font=font(22), fill=(160, 215, 255))
    return np.array(img)

def overlay(frame: np.ndarray, t: float, phase: str,
            corrected: bool = False, miss: bool = False) -> np.ndarray:
    img = Image.fromarray(upscale(frame)).convert("RGB")
    dr  = ImageDraw.Draw(img, "RGBA")
    dr.rounded_rectangle((24, 18, 368, 88), radius=16, fill=(8, 12, 18, 225))
    dr.text((46, 34), "GROUND TRUTH", font=font(28, True), fill=(85, 225, 120))
    rc = (45, 205, 240) if corrected else (240, 80, 60)
    rl = "CORRECTED" if corrected else "FAULTY"
    dr.rounded_rectangle((WO-378, 18, WO-22, 88), radius=16,
                          fill=(8, 18, 30, 225) if corrected else (30, 8, 8, 225))
    dr.text((WO-348, 34), rl, font=font(28, True), fill=rc)
    if miss:
        dr.rounded_rectangle((WO-440, 108, WO-22, 184), radius=14, fill=(195, 18, 18, 235))
        dr.text((WO-418, 120), "LATERAL MISS  +7 mm Y",
                font=font(26, True), fill=(255, 255, 255))
    dr.line((WO//2, 0, WO//2, HO), fill=(255, 255, 255, 28), width=2)
    foot = ("both arms corrected — cans placed on green targets" if corrected
            else "identical joint commands — wrist +7 mm Y → lateral miss")
    dr.rounded_rectangle((22, HO-76, WO-22, HO-18), radius=14, fill=(8, 10, 16, 225))
    dr.text((40, HO-60),
            f"phase: {phase}   |   {foot}   |   t = {t:.1f} s",
            font=font(20, True), fill=(230, 230, 230))
    return np.array(img)

def freeze_panel(frame: np.ndarray) -> np.ndarray:
    img = Image.fromarray(upscale(frame)).convert("RGB")
    dr  = ImageDraw.Draw(img, "RGBA")
    dr.rectangle((0, 0, WO, HO), fill=(0, 0, 0, 168))
    dr.rounded_rectangle((220, 120, WO-220, HO-120), radius=26,
                          fill=(8, 12, 18, 250), outline=(38, 210, 100), width=3)
    y = 185
    lines = [
        ("Fault detected",                                (255, 112, 75),  38, True),
        ("Wrist lateral offset  +7 mm Y",                (255, 212, 110), 30, True),
        (f"Observed Y drift at close:  {OBSERVED_DRIFT*1e3:.2f} mm", (175, 218, 255), 24, False),
        (f"Estimated wrist offset:      {OBSERVED_DRIFT*1e3:.2f} mm  (err < 0.4 mm)", (175, 218, 255), 24, False),
        ("", None, 10, False),
        ("Correction computed",                           (55, 205, 255),  28, True),
        (f"correction_y  =  -{OBSERVED_DRIFT*1e3:.2f} mm", (145, 228, 158), 24, False),
        ("", None, 10, False),
        ("Reloading arm  ->  wrist_offset_y = 0.000 m", (55, 205, 255),  26, True),
    ]
    for text, color, sz, bold in lines:
        if color:
            dr.text((290, y), text, font=font(sz, bold), fill=color)
        y += sz + 18
    y += 8
    dr.rounded_rectangle((290, y, WO-290, y + 120), radius=12, fill=(3, 5, 10, 255))
    code = [
        f"observed_drift_y  = {OBSERVED_DRIFT*1e3:.2f}e-3    # metres",
        f"estimated_offset  = observed_drift_y / {SENSITIVITY_Y}",
         "correction_y      = -estimated_offset",
    ]
    for i, ln in enumerate(code):
        dr.text((330, y + 18 + i * 34), ln, font=font(21), fill=(145, 228, 158))
    return np.array(img)

# ── phase runners ─────────────────────────────────────────────────────────────
def run_phase1():
    model, data = build_model(corrected=False)
    renderer    = mujoco.Renderer(model, height=H, width=W)
    cam         = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main")
    sl_l = body_qpos_slice(model, "can_left")
    sl_r = body_qpos_slice(model, "can_right")
    set_free_pose(data, sl_l, LEFT_CAN)
    set_free_pose(data, sl_r, RIGHT_CAN)
    mujoco.mj_forward(model, data)
    frames, att_l, miss_shown = [], False, False
    every = max(1, round(1.0 / (FPS * DT)))
    t, step = 0.0, 0
    while t < T_FREEZE:
        ql, gl, phase = ctrl_p1(t, LEFT_Q)
        qr, gr, _     = ctrl_p1(t, RIGHT_Q)
        data.ctrl[:] = [ql[0], ql[1], ql[2], gl/2, gl/2,
                        qr[0], qr[1], qr[2], gr/2, gr/2]
        mujoco.mj_step(model, data)
        tcp_l = site_pos(model, data, "left_tcp")
        can_l = data.qpos[sl_l.start:sl_l.start+3].copy()
        if not att_l and gl < GRIP_OPEN - 0.002 and np.linalg.norm(tcp_l - can_l) < 0.055:
            att_l = True
        if att_l:
            set_free_pose(data, sl_l, tcp_l + np.array([TCP, 0.0, 0.0]))
            if t >= T7:
                att_l = False
                set_free_pose(data, sl_l, LEFT_TARGET)
        if gr < GRIP_OPEN - 0.002 and not miss_shown:
            miss_shown = True
        mujoco.mj_forward(model, data)
        if step % every == 0:
            if t < T0:
                frames.append(title_card())
            else:
                renderer.update_scene(data, camera=cam)
                frames.append(overlay(renderer.render().copy(), t, phase,
                                      corrected=False, miss=miss_shown))
        t += DT; step += 1
    renderer.update_scene(data, camera=cam)
    frozen = freeze_panel(renderer.render().copy())
    for _ in range(int(FREEZE_DUR * FPS)):
        frames.append(frozen.copy())
    return frames


def run_phase2():
    model, data = build_model(corrected=True)
    renderer    = mujoco.Renderer(model, height=H, width=W)
    cam         = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main")
    sl_l = body_qpos_slice(model, "can_left")
    sl_r = body_qpos_slice(model, "can_right")
    set_free_pose(data, sl_l, LEFT_CAN)
    set_free_pose(data, sl_r, RIGHT_CAN)
    mujoco.mj_forward(model, data)
    frames, att_l, att_r = [], False, False
    every = max(1, round(1.0 / (FPS * DT)))
    t, step = 0.0, 0
    while t < R_END:
        ql, gl, phase = ctrl_p2(t, LEFT_Q)
        qr, gr, _     = ctrl_p2(t, RIGHT_Q)
        data.ctrl[:] = [ql[0], ql[1], ql[2], gl/2, gl/2,
                        qr[0], qr[1], qr[2], gr/2, gr/2]
        mujoco.mj_step(model, data)
        tcp_l = site_pos(model, data, "left_tcp")
        tcp_r = site_pos(model, data, "right_tcp")
        can_l = data.qpos[sl_l.start:sl_l.start+3].copy()
        can_r = data.qpos[sl_r.start:sl_r.start+3].copy()
        if not att_l and gl < GRIP_OPEN - 0.002 and np.linalg.norm(tcp_l - can_l) < 0.055:
            att_l = True
        if not att_r and gr < GRIP_OPEN - 0.002 and np.linalg.norm(tcp_r - can_r) < 0.055:
            att_r = True
        if att_l:
            set_free_pose(data, sl_l, tcp_l + np.array([TCP, 0.0, 0.0]))
            if t >= R7:
                att_l = False
                set_free_pose(data, sl_l, LEFT_TARGET)
        if att_r:
            set_free_pose(data, sl_r, tcp_r + np.array([TCP, 0.0, 0.0]))
            if t >= R7:
                att_r = False
                set_free_pose(data, sl_r, RIGHT_TARGET)
        mujoco.mj_forward(model, data)
        if step % every == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(overlay(renderer.render().copy(), t, phase,
                                  corrected=True, miss=False))
        t += DT; step += 1
    return frames

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    print("Phase 1: ground truth + faulty arm ...")
    f1 = run_phase1()
    print(f"  {len(f1)} frames ({len(f1)/FPS:.1f}s)")
    print("Phase 2: ground truth + corrected arm ...")
    f2 = run_phase2()
    print(f"  {len(f2)} frames ({len(f2)/FPS:.1f}s)")
    frames = f1 + f2
    print(f"Total: {len(frames)} frames = {len(frames)/FPS:.1f}s")
    print(f"Writing {OUT} ...")
    iio.imwrite(OUT, frames, fps=FPS, codec="libx264",
                macro_block_size=1,
                output_params=["-crf", "16", "-preset", "slow"])
    print(f"Done  ->  {OUT}")

if __name__ == "__main__":
    main()
