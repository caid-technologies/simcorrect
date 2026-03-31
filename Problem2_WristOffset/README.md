# Problem 2 — Wrist Offset Fault

## Overview

A robot arm's wrist has a **7mm lateral offset** from its CAD specification (`wrist_offset_y = 0.007m` vs correct `0.000m`, +7.7% error). Both arms receive **identical joint commands**. The ground truth arm grasps and places successfully. The faulty arm drifts 7mm sideways at every configuration — enough to cause a consistent grasp failure.

SimCorrect detects the divergence, isolates the wrist parameter, corrects it via OpenCAD, and verifies task success.

---

## Fault Specification

| Parameter | Ground Truth | Faulty | Error |
|---|---|---|---|
| `wrist_offset_y` | 0.000 m | 0.007 m | +7mm (+7.7%) |
| Lateral EE drift | ~0 mm | ~7.0 mm | — |
| Grasp success | ✓ | ✗ | — |

---

## Simulation Architecture

**3-link planar arm** with parallel-jaw gripper:
- `j1` — shoulder (hinge Y)
- `j2` — elbow (hinge Y)
- `j3` — wrist pitch (hinge Y) — compensates for vertical approach

**Verified kinematics:**
- Arm descends **vertically** onto can (`j1+j2+j3 = π/2` at all pick waypoints)
- Can on floor, placed on table via green target spot
- No arm/table collision on any trajectory waypoint

**Per-frame instrumentation:**
- EE→can distance (mm)
- Lateral drift (mm)
- Correction Δy needed (mm)
- Sim health check every 5s

---

## Scripts

| Script | Purpose |
|---|---|
| `sim_pair.py` | Phase 1 — dual simulation, divergence logging |
| `render_demo.py` | Full video: fault → freeze → OpenCAD → correction → success |

---

## Quickstart
```bash
cd Problem2_WristOffset
python sim_pair.py       # Phase 1: dual sim, confirmed 6.99mm lateral EE divergence
python render_demo.py    # Full video → ~/Desktop/Video2_WristOffset.mp4
```

**sim_pair.py output:**
```
Lateral (Y) EE error — mean: 6.99 mm  max: 7.04 mm
Joint state RMSE:      0.000011 rad  (expected ~0 — fault is geometric, not dynamic)
Divergence confirmed.
```

---

## OpenCAD Correction
```python
from opencad import Part, Sketch

Part('wrist').set_offset(y=0.000).export('wrist_corrected.stl')
sim.reload('wrist_corrected.stl')   # zero human intervention
```

---

## Key Properties

**Geometric fault, not dynamic.** Joint RMSE ≈ 0 — the same commands produce the same joint angles on both arms. The divergence is purely from the wrist body offset propagating through forward kinematics.

**Deterministic.** The 7mm lateral error is constant and reproducible across all configurations. This makes it highly tractable for correction.

**Subtle but catastrophic.** 7mm is invisible to the naked eye on a real robot. On a cylindrical can with radius 33mm, it causes the gripper to contact the rim rather than the body — consistent grasp failure.
