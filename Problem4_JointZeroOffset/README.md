# Problem 4 — Joint Zero Offset

> **"The Arm That Points the Wrong Way"**

## What is this fault?

Joint 1 is the base joint — it rotates the entire arm left and right around the vertical axis. During assembly or maintenance, the encoder on joint 1 was mounted **8 degrees (0.1396 rad)** off from its correct zero position.

The controller has no idea. It commands joint 1 to zero. The encoder reports zero. But the physical arm is already 8 degrees rotated from where the controller thinks it is. Every trajectory that follows is rotated by 8 degrees from the wrong starting point.

## The diagnostic signature

| Metric | Value |
|---|---|
| Rotational miss at 0.75m reach | **103mm** |
| Rotational miss at 0.375m reach | **52mm** |
| Scaling ratio | **2.0** (pure rotation — miss doubles with reach) |
| Joint RMSE | **0.000** |
| Fault class | **Geometric** |

The miss scales linearly with reach. This is the mathematical signature of a rotational error and what distinguishes it from every other fault class.

## How it differs from Problems 1, 2, 3

| Problem | Fault | Miss type | Joint RMSE |
|---|---|---|---|
| 1 | Link too short | Forward (fixed) | 0 |
| 2 | Wrist offset | Lateral (fixed) | 0 |
| 3 | Joint friction | Velocity-dependent lag | **> 0** |
| **4** | **Base rotation** | **Rotational (scales with reach)** | **0** |

## Correction

```python
from opencad import Part
Part('joint1').set_ref(0.0000).export('joint1_corrected.stl')
sim.reload('joint1_corrected.stl')  # zero human intervention
```

One number changes. Correction time: **0.28 seconds**.

## Output snapshots

### 01 — Title card
![Title](output/01_title.png)

### 02 — Rotational miss (t=17s)
![Rotational Miss](output/02_rotational_miss.png)

### 03 — OpenCAD freeze panel (t=43s)
![Freeze Panel](output/03_freeze_panel.png)

### 04 — Corrected arm grasping (t=62s)
![Corrected](output/04_corrected.png)

### 05 — Both cans placed on target (t=82s)
![Both Placed](output/05_both_placed.png)

## File structure
## Run

```bash
cd ~/simcorrect/Problem4_JointZeroOffset
python step0.py
python demo.py
python correction_and_validation.py
python render_demo.py
```
