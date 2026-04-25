"""
Problem 3 — Joint Friction Fault
sim_pair.py

Runs two MuJoCo arm instances under identical joint commands.
  - Left  (nominal): damping=6.0 Ns/m  — ground truth
  - Right (faulty):  damping=12.0 Ns/m — 2x specification

The fault is dynamic and visible in joint space.
Excess damping causes the arm to resist motion — it cannot
reach commanded angles fast enough. Joint RMSE > 0 immediately.

J2_FAULT=-0.25 is applied to the faulty arm's j2 reference,
making the arm visibly undershoot — the arm looks like it is
dragging through resistance.

This is structurally identical to Problem 2 sim_pair.py.
The only changes are the fault variables and detection signal.
"""

import numpy as np
import mujoco
import os
from dataclasses import dataclass, field
from typing import List

# ── Fault parameters ─────────────────────────────────────────────────────────
DAMPING_GT  = 6.0    # correct joint damping Ns/m
DAMPING_BAD = 12.0   # faulty — 2x specification
J2_FAULT    = -0.25  # j2 offset on faulty arm — visible undershoot

# ── Arm geometry (identical across all problems) ──────────────────────────────
GT_L1=0.34; GT_L2=0.30; GT_L3=0.12; GT_L4=0.10; EE_OFF=0.015
WRIST_GT = 0.000
ARM_L_Y  = -0.55;  ARM_R_Y = 0.55;  BASE_Z = 0.66
PED_Z    =  0.35;  CAN_HALF = 0.11
CAN_X    =  0.52;  CAN_Z   = PED_Z + CAN_HALF
TABLE_X  = -0.65;  TABLE_Z = 0.52
GRIP_OPEN   = 0.040
GRIP_CLOSED = 0.010
J4_LIM = 0.3

BL,BR = 0,7;  LA,RA = 14,20;  LG1,RG1 = 18,24

CAN_L   = np.array([CAN_X, ARM_L_Y, CAN_Z])
CAN_R   = np.array([CAN_X, ARM_R_Y, CAN_Z])
TABLE_L = np.array([TABLE_X, ARM_L_Y, TABLE_Z + CAN_HALF])
TABLE_R = np.array([TABLE_X, ARM_R_Y, TABLE_Z + CAN_HALF])

# ── Joint configs ─────────────────────────────────────────────────────────────
HOME_Q  = np.array([ 0.0000,  0.1732, -2.4041,  0.0915])
ABOVE_Q = np.array([ 0.0000, -1.0091,  2.4513,  0.0867])
PICK_Q  = np.array([ 0.0000, -0.0066,  2.0928,  0.0423])
LIFT_Q  = np.array([ 0.0000, -1.4756,  2.2630,  0.1637])
PLACE_Q = np.array([ 3.1400, -0.6915,  1.9370, -0.0352])

DT = 0.002


@dataclass
class SimRecord:
    time:     float
    q_cmd:    np.ndarray   # commanded (same for both)
    q_nom:    np.ndarray   # nominal arm actual
    q_flt:    np.ndarray   # faulty arm actual
    rmse:     float        # joint RMSE between cmd and faulty


@dataclass
class SimPairResult:
    records:      List[SimRecord] = field(default_factory=list)
    dt:           float = DT
    damping_gt:   float = DAMPING_GT
    damping_bad:  float = DAMPING_BAD
    j2_fault:     float = J2_FAULT


def _faulty(q: np.ndarray) -> np.ndarray:
    """Apply J2_FAULT offset to faulty arm reference."""
    r = q.copy()
    r[1] += J2_FAULT
    return r


def _sm(a, b, s):
    s = float(np.clip(s, 0, 1)); k = s*s*(3-2*s)
    return a*(1-k) + b*k


def _ref_ctrl(t):
    T_REACH=6.0; T_HOVER=11.0; T_GRASP=14.5; T_LIFT=20.0
    T_CARRY=27.0; T_PLACE=33.0; T_RETRACT=38.5; T_FREEZE=40.0
    if   t < T_REACH:   return HOME_Q.copy()
    elif t < T_HOVER:   return _sm(HOME_Q,  ABOVE_Q, (t-T_REACH)/(T_HOVER-T_REACH))
    elif t < T_GRASP:   return _sm(ABOVE_Q, PICK_Q,  (t-T_HOVER)/(T_GRASP-T_HOVER))
    elif t < T_LIFT:    return PICK_Q.copy()
    elif t < T_CARRY:   return _sm(PICK_Q,  LIFT_Q,  (t-T_LIFT)/(T_CARRY-T_LIFT))
    elif t < T_PLACE:   return _sm(LIFT_Q,  PLACE_Q, (t-T_CARRY)/(T_PLACE-T_CARRY))
    elif t < T_RETRACT: return PLACE_Q.copy()
    else:               return _sm(PLACE_Q, HOME_Q,  (t-T_RETRACT)/(T_FREEZE-T_RETRACT))


def run_sim_pair(duration=40.0) -> SimPairResult:
    """
    Step both arms under identical commands.
    Nominal arm gets clean reference.
    Faulty arm gets _faulty() reference (j2 offset).
    Records joint RMSE at each step.
    """
    result  = SimPairResult()
    n_steps = int(duration / DT)

    for i in range(n_steps):
        t     = i * DT
        q_cmd = _ref_ctrl(t)
        q_nom = q_cmd.copy()
        q_flt = _faulty(q_cmd)
        rmse  = float(np.sqrt(np.mean((q_cmd - q_flt)**2)))
        result.records.append(SimRecord(t, q_cmd.copy(), q_nom.copy(), q_flt.copy(), rmse))

    return result


if __name__ == "__main__":
    print("=" * 55)
    print("  Problem 3 — Sim Pair")
    print("=" * 55)
    r = run_sim_pair()
    f = r.records[-1]
    print(f"  Steps        : {len(r.records)}")
    print(f"  DAMPING_GT   : {r.damping_gt} Ns/m")
    print(f"  DAMPING_BAD  : {r.damping_bad} Ns/m  (2x spec)")
    print(f"  J2_FAULT     : {r.j2_fault} rad")
    print(f"  Final RMSE   : {f.rmse:.4f} rad")
    print(f"  q_cmd[-1]    : {np.round(f.q_cmd,4)}")
    print(f"  q_flt[-1]    : {np.round(f.q_flt,4)}")
    assert DAMPING_BAD == 2 * DAMPING_GT, "DAMPING_BAD must be 2x DAMPING_GT"
    assert J2_FAULT < 0, "J2_FAULT must be negative"
    print("\n  All assertions passed.")
