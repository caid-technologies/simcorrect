# Problem 3 — Joint Friction Fault

> *The arm looks geometrically perfect. It moves through mud.*

## The Fault

Joint damping is **2× above specification** across all joints.

| Parameter | Specification | Faulty | Delta |
|---|---|---|---|
| `joint_damping` | 6.0 Ns/m | 12.0 Ns/m | +6.0 |

Excess damping means every joint resists motion proportionally to velocity.
The controller sends correct torques — but the joints absorb them before
producing motion. The arm undershots every target angle. It cannot reach
the can. It moves like it is dragging through thick mud.

## Why This Matters

This is the most realistic fault class in industrial robotics.

- A new robot works perfectly on day 1
- Over months: seals age, lubricant degrades, contamination builds
- Damping increases gradually — no single failure event
- Static inspection shows nothing wrong — geometry is correct
- The arm *looks* fine at rest
- Under dynamic load, it underperforms and eventually fails tasks

Engineers often blame the controller, the path planner, or the payload.
The true cause — degraded joint mechanics — goes undiagnosed for months.

## What Makes Problem 3 Unique

| | Problem 1 | Problem 2 | Problem 3 |
|---|---|---|---|
| Fault type | Link length −37% | Wrist offset +150mm | Damping ×2 |
| Joint RMSE | **0** | **0** | **> 0** |
| Cartesian miss | Yes | Yes (Y-axis) | Yes (undershoot) |
| Visible at rest | Yes | Subtle | **No** |
| Detection signal | EE position | EE Cartesian Y | **Joint RMSE** |
| Correction target | CAD geometry | CAD geometry | **Joint seal** |

Problems 1 and 2: joint RMSE = 0. The fault is purely geometric.
Problem 3: joint RMSE > 0. The arm physically cannot reach commanded angles.
This is the key distinction — and why a different detection signal is needed.

## Detection
RMSE(t) = sqrt( mean( (q_commanded − q_actual)² ) )
alarm when RMSE > 0.015 rad
The signal fires immediately on motion start. No warmup period needed. Unlike Problems 1 and 2, the fault is visible in joint space — not just at the end-effector.

---

## Identification

`ParameterIdentifier` checks the fault report from `DivergenceDetector`:

- `joint_rmse > 0` confirms a dynamic fault, not a geometry fault
- The divergence pattern grows with velocity — matches `joint_damping` signature
- Rules out `joint_stiffness` which grows with position error
- Confidence: **0.96**

---

## Correction via OpenCAD

Unlike Problems 1 and 2 which regenerate link geometry, Problem 3 corrects the **joint seal** — the physical component responsible for damping. A degraded seal has excess contact area, increasing resistance. OpenCAD remanufactures it to specification and exports a corrected STL. SimCorrect reloads the simulation automatically. Zero human intervention.

```python
from opencad import Part, Sketch
Part().extrude(Sketch().circle(r=0.025), depth=0.08).export("joint_corrected.stl")
```

Called automatically at the freeze point in `render_demo.py`:

```python
report    = detector.get_fault_report()
id_result = ParameterIdentifier().identify(report)
corr      = correct_joint_friction(id_result["fault_value"])
validate_correction(corr["corrected_value"], id_result["fault_value"])
model, data = build(corr["damping_gt"], "0.04 0.54 0.74 1")
```

---

## Visual Fault Design

`J2_FAULT = −0.25 rad` is added to j2 of the faulty arm reference. The elbow visibly droops — the gripper arrives below and short of the can. The fault is unmistakable to bare eyes without any overlay. The corrected arm uses plain `PICK_Q` with no offset and looks identical to the ground truth arm.

---

## Pre-flight Assertions

The script verifies correctness before a single frame is rendered:

```python
assert DAMPING_BAD == 2 * DAMPING_GT        # fault is exactly 2x spec
assert J2_FAULT < 0                          # j2 offset is negative
assert "_faulty" not in source(cor_ctrl_r)   # corrected arm is clean
assert "_faulty"     in source(ref_ctrl_r)   # faulty arm has offset
assert dist_l < 0.04                         # ground truth picks can
assert not dist_r < 0.04                     # faulty arm misses can
assert dist_r2 < 0.15                        # corrected arm picks can
```

---

## Files

| File | Role |
|---|---|
| `render_demo.py` | Full 88s MuJoCo video — calls OpenCAD pipeline at freeze point |
| `sim_pair.py` | Runs nominal and faulty arms under identical commands |
| `divergence_detector.py` | Joint RMSE detector — alarms when RMSE > 0.015 rad |
| `parameter_identifier.py` | Identifies `joint_damping` as root cause with 0.96 confidence |
| `correction_and_validation.py` | Calls OpenCAD, exports STL, validates correction |

---

## Output Frames

| File | Timestamp | Content |
|---|---|---|
| `output/01_title.png` | t = 2s | Title card — The Arm That Runs in Mud |
| `output/02_stall.png` | t = 17s | Faulty arm drooping, missing can |
| `output/03_freeze_panel.png` | t = 43s | SimCorrect + OpenCAD diagnosis panel |
| `output/04_corrected.png` | t = 62s | Teal corrected arm grasping successfully |
| `output/05_both_placed.png` | t = 82s | Both arms placed on target |

---

## Run

```bash
cd Problem3_JointFriction

uv run --project .. python render_demo.py

uv run --project .. python sim_pair.py
uv run --project .. python divergence_detector.py
uv run --project .. python parameter_identifier.py
uv run --project .. python correction_and_validation.py
```

Video saves to `Problem3_JointFriction/output/Video3_JointFriction.mp4` by default. Set `SIMCORRECT_OUTPUT_DIR` to place generated videos and snapshots elsewhere.

---

## Push

```bash
cd /path/to/SimCorrect
git add Problem3_JointFriction/
git commit -m "Problem3: joint friction fault with OpenCAD correction"
git push origin main
```
