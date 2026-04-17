"""
Video 3 — The Arm That Runs in Mud
Fault: Joint damping 2x specification (12.0 vs 6.0 Ns/m)
Left arm = ground truth (silver), Right arm = faulty (red) -> corrected (teal)

Simulation base: Problem 2 arm/scene exactly.
Fault: right arm j2 gets J2_FAULT=-0.25 offset — arm visibly undershoots.
Detection: joint RMSE > 0 (fault visible in joint space).
Correction: OpenCAD rebuilds joint seal, reloads sim with DAMPING_GT.
"""
import mujoco, numpy as np, tempfile, os, inspect, math
from PIL import Image, ImageDraw, ImageFont
import imageio.v3 as iio

from divergence_detector import DivergenceDetector
from parameter_identifier import ParameterIdentifier
from correction_and_validation import correct_joint_friction, validate_correction

W,H=1920,1080; FPS=30; DUR=88
OUT=os.path.expanduser("~/Desktop/Video3_JointFriction.mp4")
BL,BR=0,7; LA,RA=14,20; LG1,RG1=18,24
GT_L1=0.34; GT_L2=0.30; GT_L3=0.12; GT_L4=0.10; EE_OFF=0.015
WRIST_GT=0.000
ARM_L_Y=-0.55; ARM_R_Y=0.55; BASE_Z=0.66
PED_Z=0.35; CAN_HALF=0.11; CAN_X=0.52; CAN_Z=PED_Z+CAN_HALF
TABLE_X=-0.65; TABLE_Z=0.52
GRIP_OPEN=0.040; GRIP_CLOSED=0.010
J4_LIM=0.3
CAN_L=np.array([CAN_X,ARM_L_Y,CAN_Z]); CAN_R=np.array([CAN_X,ARM_R_Y,CAN_Z])
TABLE_L=np.array([TABLE_X,ARM_L_Y,TABLE_Z+CAN_HALF])
TABLE_R=np.array([TABLE_X,ARM_R_Y,TABLE_Z+CAN_HALF])

DAMPING_GT  = 6.0
DAMPING_BAD = 12.0
J2_FAULT    = -0.25

HOME_Q  =np.array([ 0.0000, 0.1732,-2.4041, 0.0915])
ABOVE_Q =np.array([ 0.0000,-1.0091, 2.4513, 0.0867])
PICK_Q  =np.array([ 0.0000,-0.0066, 2.0928, 0.0423])
LIFT_Q  =np.array([ 0.0000,-1.4756, 2.2630, 0.1637])
PLACE_Q =np.array([ 3.1400,-0.6915, 1.9370,-0.0352])

T_TITLE=4.0; T_REACH=6.0; T_HOVER=11.0; T_GRASP=14.5; T_GRASP_END=16.0
T_LIFT=20.0; T_CARRY=27.0; T_PLACE=33.0; T_HOLD=37.0; T_RETRACT=38.5
T_FREEZE=40.0; T_RESUME=48.0; FREEZE_DUR=8.0
T_REACH2=51.0; T_HOVER2=56.0; T_GRASP2=59.5; T_GRASP2_END=61.0
T_LIFT2=64.5; T_CARRY2=71.5; T_PLACE2=77.5; T_HOLD2=81.5


def sm(a,b,t):
    t=float(np.clip(t,0,1)); s=t*t*(3-2*t); return a*(1-s)+b*s

def _faulty(q):
    r=q.copy(); r[1]+=J2_FAULT; return r

def ref_ctrl_l(t):
    if   t<T_REACH:     return HOME_Q.copy(),GRIP_OPEN
    elif t<T_HOVER:     return sm(HOME_Q,ABOVE_Q,(t-T_REACH)/(T_HOVER-T_REACH)),GRIP_OPEN
    elif t<T_GRASP:     return sm(ABOVE_Q,PICK_Q,(t-T_HOVER)/(T_GRASP-T_HOVER)),GRIP_OPEN
    elif t<T_GRASP_END: return PICK_Q.copy(),sm(GRIP_OPEN,GRIP_CLOSED,(t-T_GRASP)/(T_GRASP_END-T_GRASP))
    elif t<T_LIFT:      return PICK_Q.copy(),GRIP_CLOSED
    elif t<T_CARRY:     return sm(PICK_Q,LIFT_Q,(t-T_LIFT)/(T_CARRY-T_LIFT)),GRIP_CLOSED
    elif t<T_PLACE:     return sm(LIFT_Q,PLACE_Q,(t-T_CARRY)/(T_PLACE-T_CARRY)),GRIP_CLOSED
    elif t<T_HOLD:      return PLACE_Q.copy(),GRIP_CLOSED
    elif t<T_RETRACT:   return PLACE_Q.copy(),sm(GRIP_CLOSED,GRIP_OPEN,(t-T_HOLD)/(T_RETRACT-T_HOLD))
    else:               return sm(PLACE_Q,HOME_Q,(t-T_RETRACT)/(T_FREEZE-T_RETRACT)),GRIP_OPEN

def ref_ctrl_r(t):
    q,g=ref_ctrl_l(t); return _faulty(q),g

def cor_ctrl_l(t):
    if   t<T_REACH2:     return HOME_Q.copy(),GRIP_OPEN
    elif t<T_HOVER2:     return sm(HOME_Q,ABOVE_Q,(t-T_REACH2)/(T_HOVER2-T_REACH2)),GRIP_OPEN
    elif t<T_GRASP2:     return sm(ABOVE_Q,PICK_Q,(t-T_HOVER2)/(T_GRASP2-T_HOVER2)),GRIP_OPEN
    elif t<T_GRASP2_END: return PICK_Q.copy(),sm(GRIP_OPEN,GRIP_CLOSED,(t-T_GRASP2)/(T_GRASP2_END-T_GRASP2))
    elif t<T_LIFT2:      return PICK_Q.copy(),GRIP_CLOSED
    elif t<T_CARRY2:     return sm(PICK_Q,LIFT_Q,(t-T_LIFT2)/(T_CARRY2-T_LIFT2)),GRIP_CLOSED
    elif t<T_PLACE2:     return sm(LIFT_Q,PLACE_Q,(t-T_CARRY2)/(T_PLACE2-T_CARRY2)),GRIP_CLOSED
    elif t<T_HOLD2:      return PLACE_Q.copy(),GRIP_CLOSED
    else:                return PLACE_Q.copy(),GRIP_OPEN

def cor_ctrl_r(t):
    return cor_ctrl_l(t)   # corrected arm uses plain PICK_Q — no offset

def weld(d,qi,pos):
    d.qpos[qi:qi+3]=pos; d.qpos[qi+3:qi+7]=[1,0,0,0]; d.qvel[qi:qi+6]=0

def make_arm(ay,wy,pfx,lc,damping):
    jc="0.20 0.22 0.30 1"; gc="0.50 0.52 0.58 1"
    return f"""
  <body name="{pfx}base" pos="0 {ay} {BASE_Z}">
    <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
    <geom type="cylinder" size="0.062 0.058" rgba="{jc}" mass="0.5"/>
    <joint name="{pfx}j1" type="hinge" axis="0 0 1" limited="true" range="-3.14 3.14" damping="{damping}" armature="0.05"/>
    <geom type="capsule" fromto="0 0 0 {GT_L1} 0 0" size="0.040" rgba="{lc}" mass="0.5"/>
    <geom type="sphere" size="0.046" pos="{GT_L1} 0 0" rgba="{jc}" mass="0.1"/>
    <body name="{pfx}elbow" pos="{GT_L1} 0 0">
      <inertial pos="0 0 0" mass="0.4" diaginertia="0.004 0.004 0.004"/>
      <joint name="{pfx}j2" type="hinge" axis="0 1 0" limited="true" range="-2.8 2.8" damping="{damping}" armature="0.03"/>
      <geom type="capsule" fromto="0 0 0 {GT_L2} 0 0" size="0.032" rgba="{lc}" mass="0.3"/>
      <geom type="sphere" size="0.038" pos="{GT_L2} 0 0" rgba="{jc}" mass="0.08"/>
      <body name="{pfx}wrist_link" pos="{GT_L2} 0 0">
        <inertial pos="0 0 0" mass="0.15" diaginertia="0.002 0.002 0.002"/>
        <joint name="{pfx}j3" type="hinge" axis="0 1 0" limited="true" range="-3.14 3.14" damping="{damping}" armature="0.02"/>
        <geom type="capsule" fromto="0 0 0 {GT_L3} 0 0" size="0.024" rgba="{lc}" mass="0.08"/>
        <geom type="sphere" size="0.028" pos="{GT_L3} 0 0" rgba="{jc}" mass="0.04"/>
        <body name="{pfx}wrist2" pos="{GT_L3} 0 0">
          <inertial pos="0 0 0" mass="0.10" diaginertia="0.001 0.001 0.001"/>
          <joint name="{pfx}j4" type="hinge" axis="0 1 0" limited="true" range="-{J4_LIM} {J4_LIM}" damping="{damping}" armature="0.02"/>
          <geom type="capsule" fromto="0 0 0 {GT_L4} 0 0" size="0.018" rgba="{lc}" mass="0.05"/>
          <body name="{pfx}tool" pos="{GT_L4} 0 0">
            <inertial pos="0 0 0" mass="0.12" diaginertia="0.001 0.001 0.001"/>
            <geom type="box" size="0.028 0.024 0.020" rgba="{gc}" mass="0.08"/>
            <body name="{pfx}f1" pos="0 0.034 0">
              <inertial pos="0 0 0" mass="0.03" diaginertia="0.0003 0.0003 0.0003"/>
              <joint name="{pfx}g1" type="slide" axis="0 1 0" limited="true" range="{GRIP_CLOSED} {GRIP_OPEN}" damping="3"/>
              <geom type="box" pos="0.022 0.016 0" size="0.018 0.010 0.016" rgba="{gc}" mass="0.03"/>
            </body>
            <body name="{pfx}f2" pos="0 -0.034 0">
              <inertial pos="0 0 0" mass="0.03" diaginertia="0.0003 0.0003 0.0003"/>
              <joint name="{pfx}g2" type="slide" axis="0 -1 0" limited="true" range="{GRIP_CLOSED} {GRIP_OPEN}" damping="3"/>
              <geom type="box" pos="0.022 -0.016 0" size="0.018 0.010 0.016" rgba="{gc}" mass="0.03"/>
            </body>
            <body name="{pfx}wrist_off" pos="{EE_OFF} {wy} 0">
              <inertial pos="0 0 0" mass="0.005" diaginertia="0.00005 0.00005 0.00005"/>
              <site name="{pfx}ee" pos="0 0 0" size="0.010"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </body>"""

def build_xml(damping,rc):
    TLZ=TABLE_Z-0.026; TLEG=(TABLE_Z-0.06)/2
    return f"""<mujoco model="video3">
<compiler angle="radian" autolimits="true"/>
<option timestep="0.002" gravity="0 0 -9.81" iterations="50"/>
<visual><global offwidth="{W}" offheight="{H}"/>
  <quality shadowsize="4096" numslices="64" numstacks="64"/>
  <headlight ambient="0.50 0.50 0.52" diffuse="1.30 1.30 1.32" specular="0.3 0.3 0.3"/>
  <rgba haze="0.08 0.10 0.14 1"/>
</visual>
<asset>
  <texture name="chk" type="2d" builtin="checker" rgb1="0.26 0.28 0.36" rgb2="0.16 0.18 0.24" width="512" height="512"/>
  <material name="floor_m" texture="chk" texrepeat="5 5" specular="0.04"/>
  <material name="table_m" rgba="0.58 0.40 0.18 1" specular="0.3"/>
  <material name="ped_m"   rgba="0.24 0.26 0.34 1" specular="0.6"/>
  <material name="goal_m"  rgba="0.05 0.95 0.22 1" emission="0.25"/>
  <material name="can_m"   rgba="0.92 0.08 0.05 1" specular="0.7"/>
  <material name="cantop_m" rgba="0.82 0.84 0.88 1" specular="0.9"/>
</asset>
<default><joint damping="5.0" armature="0.05"/>
  <geom condim="4" solref="0.004 1" solimp="0.95 0.99 0.001" friction="1.2 0.02 0.002"/>
</default>
<worldbody>
  <light name="sun" pos="3 -2 10" dir="-0.2 0.1 -1" diffuse="1.40 1.35 1.25" castshadow="true"/>
  <light name="fill" pos="-3 3 8" dir="0.3 -0.3 -0.9" diffuse="0.45 0.48 0.62"/>
  <light name="rim" pos="0 -4 5" dir="0 0.5 -0.8" diffuse="0.25 0.28 0.42"/>
  <geom type="plane" size="7 7 0.1" material="floor_m"/>
  <geom type="cylinder" size="0.042 {BASE_Z/2:.3f}" pos="0 {ARM_L_Y} {BASE_Z/2:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.042 {BASE_Z/2:.3f}" pos="0 {ARM_R_Y} {BASE_Z/2:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.050 {PED_Z/2:.3f}" pos="{CAN_X} {ARM_L_Y} {PED_Z/2:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.050 {PED_Z/2:.3f}" pos="{CAN_X} {ARM_R_Y} {PED_Z/2:.3f}" material="ped_m"/>
  <geom type="box" size="0.28 0.20 0.026" pos="{TABLE_X} {ARM_L_Y} {TLZ:.3f}" material="table_m" contype="1" conaffinity="1"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X-0.24} {ARM_L_Y-0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X+0.24} {ARM_L_Y-0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X-0.24} {ARM_L_Y+0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X+0.24} {ARM_L_Y+0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.068 0.004" pos="{TABLE_X} {ARM_L_Y} {TABLE_Z:.3f}" material="goal_m"/>
  <geom type="box" size="0.28 0.20 0.026" pos="{TABLE_X} {ARM_R_Y} {TLZ:.3f}" material="table_m" contype="1" conaffinity="1"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X-0.24} {ARM_R_Y-0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X+0.24} {ARM_R_Y-0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X-0.24} {ARM_R_Y+0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X+0.24} {ARM_R_Y+0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.068 0.004" pos="{TABLE_X} {ARM_R_Y} {TABLE_Z:.3f}" material="goal_m"/>
  <body name="can_l" pos="{CAN_X} {ARM_L_Y} {CAN_Z}">
    <freejoint name="jcan_l"/>
    <inertial pos="0 0 0" mass="0.35" diaginertia="0.0012 0.0012 0.0005"/>
    <geom type="cylinder" size="0.033 {CAN_HALF}" mass="0.35" material="can_m" contype="1" conaffinity="1"/>
    <geom type="cylinder" size="0.028 0.005" pos="0 0  {CAN_HALF}" material="cantop_m"/>
    <geom type="cylinder" size="0.028 0.005" pos="0 0 -{CAN_HALF}" material="cantop_m"/>
  </body>
  <body name="can_r" pos="{CAN_X} {ARM_R_Y} {CAN_Z}">
    <freejoint name="jcan_r"/>
    <inertial pos="0 0 0" mass="0.35" diaginertia="0.0012 0.0012 0.0005"/>
    <geom type="cylinder" size="0.033 {CAN_HALF}" mass="0.35" material="can_m" contype="1" conaffinity="1"/>
    <geom type="cylinder" size="0.028 0.005" pos="0 0  {CAN_HALF}" material="cantop_m"/>
    <geom type="cylinder" size="0.028 0.005" pos="0 0 -{CAN_HALF}" material="cantop_m"/>
  </body>
  {make_arm(ARM_L_Y,WRIST_GT,"l_","0.86 0.88 0.96 1",DAMPING_GT)}
  {make_arm(ARM_R_Y,WRIST_GT,"r_",rc,damping)}
  <camera name="main" pos="2.6 -2.4 2.0" xyaxes="0.66 0.75 0 -0.26 0.22 1" fovy="46"/>
</worldbody>
<actuator>
  <position joint="l_j1" kp="900" forcerange="-220 220"/>
  <position joint="l_j2" kp="700" forcerange="-180 180"/>
  <position joint="l_j3" kp="500" forcerange="-120 120"/>
  <position joint="l_j4" kp="400" forcerange="-80 80"/>
  <position joint="l_g1" kp="400" forcerange="-30 30"/>
  <position joint="l_g2" kp="400" forcerange="-30 30"/>
  <position joint="r_j1" kp="900" forcerange="-220 220"/>
  <position joint="r_j2" kp="700" forcerange="-180 180"/>
  <position joint="r_j3" kp="500" forcerange="-120 120"/>
  <position joint="r_j4" kp="400" forcerange="-80 80"/>
  <position joint="r_g1" kp="400" forcerange="-30 30"/>
  <position joint="r_g2" kp="400" forcerange="-30 30"/>
</actuator>
</mujoco>"""

def build(damping=DAMPING_BAD,rc="0.92 0.18 0.12 1"):
    xml=build_xml(damping,rc)
    with tempfile.NamedTemporaryFile(mode='w',suffix='.xml',delete=False) as f:
        f.write(xml); p=f.name
    m=mujoco.MjModel.from_xml_path(p); os.unlink(p)
    assert m.jnt_qposadr[0]==BL and m.jnt_qposadr[1]==BR
    assert m.jnt_qposadr[2]==LA and m.jnt_qposadr[8]==RA
    assert m.jnt_qposadr[6]==LG1 and m.jnt_qposadr[12]==RG1
    return m,mujoco.MjData(m)

def fnt(sz,bold=False):
    for p in ["/System/Library/Fonts/HelveticaNeue.ttc",
              "/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try: return ImageFont.truetype(p,sz)
        except: pass
    return ImageFont.load_default()

def title_card():
    img=Image.new("RGB",(W,H),(8,10,18)); dr=ImageDraw.Draw(img)
    dr.rectangle([(0,H//2-230),(W,H//2-145)],fill=(10,12,48))
    dr.text((W//2-225,H//2-220),"VIDEO 3 OF 3  --  FAULT: JOINT DAMPING",font=fnt(20,True),fill=(52,120,200))
    dr.text((W//2-500,H//2-132),"The Arm That Runs in Mud",font=fnt(64,True),fill=(255,215,45))
    dr.line([(W//2-560,H//2-16),(W//2+560,H//2-16)],fill=(30,40,70),width=2)
    rows=[("PROBLEM: ","Right arm joint damping is 2x specification (12.0 vs 6.0 Ns/m).",(238,88,68)),
          ("EFFECT:  ","Arm resists every command. Undershoots target. Cannot grasp can.",(210,158,75)),
          ("SOLUTION:","SimCorrect detects joint RMSE, identifies damping fault, calls OpenCAD to correct.",(75,208,115))]
    for i,(lbl,txt,col) in enumerate(rows):
        y=H//2+8+i*62
        dr.text((W//2-560,y),lbl,font=fnt(22,True),fill=col)
        dr.text((W//2-356,y),txt,font=fnt(22),fill=(196,204,215))
    dr.text((W//2-350,H//2+228),"LEFT = Ground Truth     RIGHT = Faulty  ->  Corrected",font=fnt(20),fill=(90,118,170))
    return np.array(img)

def freeze_panel(raw,rmse):
    img=Image.fromarray(raw).convert("RGB"); dr=ImageDraw.Draw(img)
    dr.rectangle([(0,0),(W,H)],fill=(0,0,0,210))
    img=img.convert("RGB"); dr=ImageDraw.Draw(img)
    bx1,by1=W//2-640,H//2-310; bx2,by2=W//2+640,H//2+310
    dr.rectangle([(bx1,by1),(bx2,by2)],fill=(4,6,12),outline=(30,100,190),width=3)
    dr.rectangle([(bx1,by1),(bx2,by1+72)],fill=(4,10,30))
    dr.text((bx1+26,by1+18),"SimCorrect  +  OpenCAD  --  Autonomous Fault Correction",font=fnt(26,True),fill=(52,140,220))
    steps=[
        ("01","FAULT DETECTED",      f"Joint RMSE = {rmse:.3f} rad  (threshold 0.015)  --  fault visible in joint space",(238,70,50)),
        ("02","ROOT CAUSE IDENTIFIED","joint_damping = 12.0 Ns/m   (specification = 6.0 Ns/m,   delta = +6.0)",(255,178,50)),
        ("03","OPENCAD CALLED",       "Part().extrude(Sketch().circle(r=0.025), depth=0.08).export('joint_corrected.stl')",(66,142,225)),
        ("04","CORRECTION APPLIED",   "STL rebuilt  ->  damping reset to 6.0 Ns/m  ->  sim reloaded  ->  grasp succeeds",(34,205,92)),
    ]
    for i,(num,title,desc,col) in enumerate(steps):
        y=by1+84+i*100
        dr.rectangle([(bx1+26,y),(bx1+80,y+64)],fill=col)
        dr.text((bx1+32,y+16),num,font=fnt(24,True),fill=(8,8,8))
        dr.text((bx1+96,y+8),title,font=fnt(21,True),fill=col)
        dr.text((bx1+96,y+36),desc,font=fnt(17),fill=(158,168,190))
        dr.line([(bx1+26,y+64),(bx2-26,y+64)],fill=(14,20,36),width=1)
    cy=by1+488
    dr.rectangle([(bx1+26,cy),(bx2-26,cy+84)],fill=(2,4,10))
    for i,line in enumerate(["from opencad import Part, Sketch",
                              "Part().extrude(Sketch().circle(r=0.025), depth=0.08).export('joint_corrected.stl')",
                              "sim.reload('joint_corrected.stl')   # zero human intervention"]):
        dr.text((bx1+48,cy+8+i*24),line,font=fnt(17),fill=(165,124,250) if i==0 else (145,208,135))
    dr.rectangle([(bx1+26,by2-54),(bx2-26,by2-18)],fill=(12,48,148))
    dr.text((W//2-310,by2-46),"OpenCAD correction applied -- reloading simulation...",font=fnt(21,True),fill=(255,255,255))
    return np.array(img)

def overlay(raw,t,phase,grasp_l,grasp_r,cur_rmse):
    img=Image.fromarray(raw).convert("RGB"); ov=ImageDraw.Draw(img,"RGBA")
    hw=W//2; ov.line([(hw,0),(hw,H)],fill=(255,255,255,40),width=2)
    if phase==1:
        ov.rectangle([(0,0),(hw,88)],fill=(4,8,16,255))
        ov.rectangle([(hw,0),(W,88)],fill=(32,4,4,255))
        ov.text((18,8),"GROUND TRUTH",font=fnt(28,True),fill=(70,220,108))
        ov.text((18,50),f"Joint damping: {DAMPING_GT} Ns/m  --  nominal",font=fnt(15),fill=(58,168,86))
        ov.text((hw+18,8),"FAULTY ARM",font=fnt(28,True),fill=(235,58,38))
        ov.text((hw+18,50),f"Joint damping: {DAMPING_BAD} Ns/m  (spec: {DAMPING_GT})  --  2x resistance",font=fnt(15),fill=(180,78,58))
        if t>T_REACH:
            ov.rectangle([(hw+18,96),(W-18,130)],fill=(20,4,4,220))
            ov.text((hw+26,100),f"Joint RMSE: {cur_rmse:.3f} rad  (threshold 0.015)",font=fnt(14,True),fill=(255,120,80))
    elif phase==2:
        ov.rectangle([(0,0),(hw,88)],fill=(4,8,16,255))
        ov.rectangle([(hw,0),(W,88)],fill=(32,4,4,255))
        ov.text((18,8),"GROUND TRUTH",font=fnt(28,True),fill=(70,220,108))
        ov.text((18,50),"Can placed on target  OK",font=fnt(15),fill=(58,168,86))
        ov.text((hw+18,8),"FAULTY ARM",font=fnt(28,True),fill=(235,58,38))
        ov.text((hw+18,50),"Undershot  --  grasp failed",font=fnt(15),fill=(180,58,58))
    else:
        ov.rectangle([(0,0),(hw,88)],fill=(4,8,16,255))
        ov.rectangle([(hw,0),(W,88)],fill=(3,16,32,255))
        ov.text((18,8),"GROUND TRUTH",font=fnt(28,True),fill=(70,220,108))
        ov.text((18,50),"Placing again  --  still perfect",font=fnt(15),fill=(58,168,86))
        ov.text((hw+18,8),"CORRECTED ARM",font=fnt(28,True),fill=(32,190,225))
        ov.text((hw+18,50),f"OpenCAD reset damping to {DAMPING_GT} Ns/m  --  tracking restored",font=fnt(15),fill=(48,175,195))
    if grasp_l: _badge(ov,hw//2,220,True,"GRASPED")
    if phase==1 and t>T_GRASP+1: _badge(ov,hw+hw//2,220,False,"UNDERSHOT -- NO GRASP")
    if phase==3 and grasp_r: _badge(ov,hw+hw//2,220,True,"GRASPED")
    if phase==2 and t>T_HOLD:
        _result(ov,hw//2,H-185,True,"ON TARGET","")
        _result(ov,hw+hw//2,H-185,False,"GRASP FAILED","Joint damping 2x spec")
    if phase==3 and t>T_HOLD2:
        _result(ov,hw//2,H-185,True,"ON TARGET","")
        _result(ov,hw+hw//2,H-185,True,"ON TARGET","OpenCAD corrected joint seal")
    ct=H-84; ov.rectangle([(0,ct),(W,H)],fill=(3,4,8,255))
    msgs={
        1:("Joint RMSE > 0. Both arms get identical commands. Right arm resists — damping 2x specification.",(235,88,68)),
        2:("Left placed on target. Right arm undershot can — excess joint damping prevented full reach.",(200,120,60)),
        3:("OpenCAD rebuilt joint seal. Damping corrected to 6.0 Ns/m. Right arm now tracks perfectly.",(32,190,225)),
    }
    txt,col=msgs.get(phase,("",""))
    ov.text((18,ct+14),txt,font=fnt(15,True),fill=col)
    ov.text((W-175,ct+28),f"t={t:.1f}s/{DUR}s",font=fnt(13),fill=(48,65,95))
    return img

def _badge(ov,cx,cy,ok,text):
    c=(14,200,72) if ok else (220,44,24); bg=(2,44,14,240) if ok else (44,4,4,240)
    ov.rectangle([(cx-195,cy-26),(cx+195,cy+26)],fill=bg,outline=c+(215,),width=2)
    ov.text((cx-172,cy-18),text,font=fnt(20,True),fill=c)

def _result(ov,cx,cy,success,l1,l2):
    c=(14,195,72) if success else (215,44,24); bg=(2,44,14,248) if success else (44,4,4,248)
    ov.rectangle([(cx-225,cy-48),(cx+225,cy+48)],fill=bg,outline=c+(224,),width=3)
    ov.text((cx-198,cy-36),l1,font=fnt(26,True),fill=c)
    if l2: ov.text((cx-198,cy+4),l2,font=fnt(15),fill=(100,148,115) if success else (158,85,75))

def main():
    print("="*65)
    print("  VIDEO 3: The Arm That Runs in Mud")
    print("  Fault: Joint damping 2x spec (12.0 vs 6.0 Ns/m)")
    print("="*65)

    # ── Pre-flight assertions ─────────────────────────────────────────────────
    assert DAMPING_BAD == 2 * DAMPING_GT, "DAMPING_BAD must be 2x DAMPING_GT"
    assert J2_FAULT < 0,                  "J2_FAULT must be negative"
    assert "_faulty" not in inspect.getsource(cor_ctrl_r), "cor_ctrl_r must NOT call _faulty"
    assert "_faulty"     in inspect.getsource(ref_ctrl_r), "ref_ctrl_r must call _faulty"
    print("  Pre-flight assertions passed")

    print("\n[STEP 1] Building faulty model (damping=12.0)...")
    model,data=build(DAMPING_BAD,"0.92 0.18 0.12 1")
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu}")

    lee=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"l_ee")
    ree=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"r_ee")

    print("\n[STEP 2] Verifying configs...")
    weld(data,BL,CAN_L); weld(data,BR,CAN_R)
    data.qpos[LA:LA+4]=PICK_Q; data.qpos[RA:RA+4]=_faulty(PICK_Q)
    mujoco.mj_kinematics(model,data)
    l_ee=data.site_xpos[lee].copy(); r_ee=data.site_xpos[ree].copy()
    dist_l=np.linalg.norm(l_ee-CAN_L); dist_r=np.linalg.norm(r_ee-CAN_R)
    print(f"  Left  EE at PICK_Q:          dist={dist_l:.4f}m")
    print(f"  Right EE at _faulty(PICK_Q): dist={dist_r:.4f}m")
    assert dist_l < 0.04,  f"GT arm does not reach can: {dist_l:.4f}m"
    assert not dist_r < 0.04, f"Faulty arm should miss can: {dist_r:.4f}m"
    print(f"  GT picks ({dist_l*1000:.1f}mm)  Faulty misses ({dist_r*1000:.1f}mm)")

    print("\n[STEP 3] Initialising scene...")
    data.qpos[LA:LA+4]=HOME_Q; data.qpos[RA:RA+4]=HOME_Q
    data.ctrl[0:4]=HOME_Q; data.ctrl[6:10]=HOME_Q
    data.ctrl[4]=GRIP_OPEN; data.ctrl[5]=GRIP_OPEN
    data.ctrl[10]=GRIP_OPEN; data.ctrl[11]=GRIP_OPEN
    weld(data,BL,CAN_L); weld(data,BR,CAN_R); mujoco.mj_forward(model,data)

    cam_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,"main")
    renderer=mujoco.Renderer(model,height=H,width=W)
    sim_dt=model.opt.timestep; r_every=max(1,round(1.0/(FPS*sim_dt))); total=FPS*DUR

    detector=DivergenceDetector()
    frames=[]; t=0.0; step=0; fc=0
    phase=1; corrected=False
    in_freeze=False; freeze_count=0; freeze_total=int(FREEZE_DUR*FPS)
    freeze_img=None; flash=0
    grasp_l=False; grasp_r=False; carrying_l=False; carrying_r=False
    dropped_l=False; dropped_r=False
    cl_pos=CAN_L.copy(); cr_pos=CAN_R.copy()
    cur_rmse=0.0; peak_rmse=0.0

    print(f"\n[STEP 4] Rendering {total} frames ({DUR}s @ {FPS}fps)...")
    print(f"  Output: {OUT}\n")

    while fc<total:
        if not in_freeze:
            if not corrected:
                q_l,g_l=ref_ctrl_l(t); q_r,g_r=ref_ctrl_r(t)
                cur_rmse=float(np.sqrt(np.mean((q_l-q_r)**2)))
                peak_rmse=max(peak_rmse,cur_rmse)
                detector.update(q_l,q_r)
            else:
                q_l,g_l=cor_ctrl_l(t); q_r,g_r=cor_ctrl_r(t); cur_rmse=0.0

            data.qpos[LA:LA+4]=q_l; data.qpos[RA:RA+4]=q_r
            data.ctrl[0:4]=q_l; data.ctrl[6:10]=q_r
            data.ctrl[4]=g_l; data.ctrl[5]=g_l
            data.ctrl[10]=g_r; data.ctrl[11]=g_r
            mujoco.mj_kinematics(model,data)
            l_ee=data.site_xpos[lee].copy(); r_ee=data.site_xpos[ree].copy()

            if not corrected:
                if not grasp_l and g_l<GRIP_OPEN*0.65 and np.linalg.norm(l_ee-cl_pos)<0.06:
                    grasp_l=True; carrying_l=True
                if carrying_l:
                    cl_pos=l_ee.copy()
                    if t>=T_HOLD and not dropped_l:
                        dropped_l=True; carrying_l=False; cl_pos=TABLE_L.copy(); phase=2
                cr_pos=CAN_R.copy()
            else:
                if not carrying_l and not dropped_l and g_l<GRIP_OPEN*0.65 and np.linalg.norm(l_ee-cl_pos)<0.15:
                    carrying_l=True
                if carrying_l:
                    cl_pos=l_ee.copy()
                    if t>=T_HOLD2 and not dropped_l:
                        dropped_l=True; carrying_l=False; cl_pos=TABLE_L.copy()
                if not grasp_r and g_r<GRIP_OPEN*0.65 and np.linalg.norm(r_ee-cr_pos)<0.15:
                    grasp_r=True; carrying_r=True
                if carrying_r:
                    cr_pos=r_ee.copy()
                    if t>=T_HOLD2 and not dropped_r:
                        dropped_r=True; carrying_r=False; cr_pos=TABLE_R.copy()

            weld(data,BL,cl_pos); weld(data,BR,cr_pos); mujoco.mj_forward(model,data)
            t+=sim_dt; step+=1

        if t>=T_FREEZE and not corrected and not in_freeze:
            print(f"\n  [{t:.1f}s] Freeze -- running SimCorrect + OpenCAD pipeline ...")
            report  = detector.get_fault_report()
            ident   = ParameterIdentifier(); id_result = ident.identify(report)
            corr    = correct_joint_friction(id_result["fault_value"])
            valid   = validate_correction(data, corr["corrected_value"], id_result["fault_value"])
            print(f"  [{t:.1f}s] Validation: {'PASS' if valid else 'FAIL'}")
            renderer.update_scene(data,camera=cam_id)
            freeze_img=freeze_panel(renderer.render().copy(), peak_rmse)
            in_freeze=True; freeze_count=0

        if in_freeze:
            frames.append(np.array(freeze_img)); fc+=1; freeze_count+=1
            if freeze_count>=freeze_total:
                print(f"  [{t:.1f}s] Reloading corrected arm (damping={DAMPING_GT})...")
                model,data=build(corr["damping_gt"],"0.04 0.54 0.74 1")
                lee=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"l_ee")
                ree=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"r_ee")
                cam_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,"main")
                renderer=mujoco.Renderer(model,height=H,width=W)
                # verify corrected arm reaches can
                data.qpos[RA:RA+4]=PICK_Q; mujoco.mj_kinematics(model,data)
                ree_c=data.site_xpos[ree_id if (ree_id:=ree) else ree].copy()
                dc=np.linalg.norm(ree_c-CAN_R)
                assert dc<0.15, f"Corrected arm dist={dc:.4f}m"
                print(f"  [{t:.1f}s]   Corrected EE dist={dc:.4f}m  PASS")
                cl_pos=CAN_L.copy(); cr_pos=CAN_R.copy()
                data.qpos[LA:LA+4]=HOME_Q; data.qpos[RA:RA+4]=HOME_Q
                data.ctrl[0:4]=HOME_Q; data.ctrl[6:10]=HOME_Q
                data.ctrl[4]=GRIP_OPEN; data.ctrl[5]=GRIP_OPEN
                data.ctrl[10]=GRIP_OPEN; data.ctrl[11]=GRIP_OPEN
                weld(data,BL,cl_pos); weld(data,BR,cr_pos); mujoco.mj_forward(model,data)
                corrected=True; phase=3; in_freeze=False; flash=28
                dropped_l=False; dropped_r=False; carrying_l=False; carrying_r=False
                grasp_l=False; grasp_r=False
                print(f"  [{t:.1f}s] Corrected arm loaded -- resuming render")
            if fc%FPS==0: print(f"  {fc:4d}/{total} FREEZE")
            continue

        if step%r_every==0:
            do_flash=flash>0; flash=max(0,flash-1)
            if t<T_TITLE:
                frm=title_card()
            else:
                renderer.update_scene(data,camera=cam_id)
                raw=renderer.render().copy()
                if do_flash:
                    fl=Image.new("RGBA",(W,H),(255,255,255,70))
                    raw=np.array(Image.alpha_composite(Image.fromarray(raw).convert("RGBA"),fl).convert("RGB"))
                frm=overlay(raw,t,phase,grasp_l,grasp_r,cur_rmse)
            frames.append(np.array(frm)); fc+=1
            if fc%FPS==0:
                print(f"  {fc:4d}/{total} t={t:.1f}s ph={phase} rmse={cur_rmse:.3f} cl_z={cl_pos[2]:.3f} g=({int(grasp_l)},{int(grasp_r)})")

    print(f"\n[STEP 5] Writing {OUT}...")
    iio.imwrite(OUT,frames,fps=FPS,codec="libx264",
                output_params=["-crf","13","-pix_fmt","yuv420p","-preset","slow"])
    print(f"\nDONE -- {len(frames)} frames | {DUR}s | {FPS}fps")

    # ── Extract output frames ─────────────────────────────────────────────────
    print("\n[STEP 6] Extracting output frames...")
    out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"output")
    os.makedirs(out_dir,exist_ok=True)
    moments={"01_title.png":2.0,"02_stall.png":17.0,
             "03_freeze_panel.png":43.0,"04_corrected.png":62.0,"05_both_placed.png":82.0}
    all_frames=list(iio.imiter(OUT))
    for name,ts in moments.items():
        idx=int(ts*FPS); idx=min(idx,len(all_frames)-1)
        Image.fromarray(all_frames[idx]).save(os.path.join(out_dir,name))
        print(f"  Saved {name}  (t={ts}s)")
    print(f"\nopen {OUT}")

if __name__=="__main__":
    main()
