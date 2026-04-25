# SimCorrect

## Fault Detection and Correction for Robot Simulation Models

Shreya Priya · Dean (Dien) Hu

---

## What is SimCorrect?

SimCorrect answers a deceptively simple question: what happens when your simulation model is wrong?

Not wrong in a noisy, probabilistic way. Wrong in a precise, physical way. A forearm link machined 80mm too long. A wrist bracket mounted 150mm off-center. A base encoder with an 8 degree zero offset. A gripper 60 grams heavier than its model says. A joint running with twice its specified friction.

In each case, the robot controller sees nothing wrong. Joint encoders report exactly what they expect. No alarms. No errors. And yet the robot fails its task every single cycle, because the simulation it was built on does not match the machine it is running on.

SimCorrect detects these faults, identifies the exact parameter responsible, corrects the model autonomously, and verifies the fix before the robot is deployed. No human writes the correction. No hardware is touched. The entire loop from detection through correction through validation closes in under 0.3 seconds.

---

## How It Works

Two simulation instances run side by side under identical joint commands. The left arm is ground truth, correctly modelled and correctly calibrated. The right arm carries an injected fault. SimCorrect monitors the divergence between them continuously.

When the faulty arm fails a task that the ground truth arm completes, the diagnostic pipeline begins. It measures the end-effector error, computes joint RMSE, and traces the divergence back to its source parameter through sensitivity analysis. The OpenCAD API then rebuilds the corrected geometry from first principles, exports the corrected MJCF, and reloads the simulation. The corrected arm re-executes the task. Success is verified programmatically before the result is accepted.

The same pipeline handles both geometric faults, where joints execute correctly but the geometry is wrong, and dynamics faults, where joints physically cannot reach their commanded positions because the physics model is wrong.

---

## The Five Fault Scenarios

### Problem 1 — Forearm Length Error

The right arm forearm link is 80mm longer than its CAD specification. The arm overshoots its target on every pick attempt, closing the gripper above the can rather than around it. The joint encoders report normal values because the joints are doing exactly what they are told. The fault lives in geometry, not in motion. SimCorrect detects the vertical end-effector overshoot, isolates the forearm length as the responsible parameter, and corrects the link geometry via OpenCAD.

### Problem 2 — Wrist Lateral Offset

The right arm wrist bracket is physically mounted 150mm off-center in the lateral axis. The arm executes every joint command with perfect precision and lands its gripper 250mm away from the can on every attempt. Joint RMSE is zero throughout. The fault exists entirely in Cartesian space and is invisible to any joint-level diagnostic. SimCorrect detects the consistent lateral drift, identifies the wrist offset parameter, corrects it, and the arm grasps correctly on the next attempt.

### Problem 3 — Joint Friction Fault

Joint friction is running at more than double its specified value. The arm stalls mid-trajectory and loses positional accuracy under load. Unlike the geometric faults, this one produces non-zero joint RMSE. The joints physically cannot reach their commanded positions because friction is consuming torque that was meant for motion. SimCorrect detects the velocity-dependent joint lag, identifies the friction coefficient as the fault source, and corrects it.

### Problem 4 — Base Encoder Offset

The base rotation joint has its encoder mounted 8 degrees off its correct zero position. Every trajectory the arm executes inherits this rotational error at the root. The arm moves with absolute precision relative to what it believes is forward, and misses its target by 103mm because its definition of forward is wrong. Joint RMSE is zero. The positional error scales linearly with reach distance. SimCorrect detects the rotational signature, identifies the encoder zero offset, and corrects the joint reference in 0.28 seconds.

### Problem 5 — Tool Mass Mismatch

The gripper physically weighs 0.160kg but the simulation model records it as 0.100kg. The controller calculates gravity compensation torques based on the modelled mass, sending insufficient torque to hold the arm against the real gravitational load. At full horizontal extension, the uncompensated torque is 0.44 Nm. The arm droops 55mm below its commanded position on every pick attempt. No encoder error is generated. No controller alarm fires. The robot just quietly misses, every cycle. SimCorrect detects the non-zero joint RMSE at extended poses, confirms the gravity-dependent signature that distinguishes mass mismatch from friction, estimates the mass delta analytically, and corrects the model via OpenCAD.

---

## Fault Coverage

Problem 1 detects forearm length error through vertical end-effector overshoot with zero joint RMSE and corrects in 0.28 seconds. Problem 2 detects wrist lateral offset through lateral end-effector drift with zero joint RMSE and corrects in 0.28 seconds. Problem 3 detects joint friction excess through velocity-dependent joint lag with non-zero joint RMSE and corrects in 0.28 seconds. Problem 4 detects encoder zero offset through rotational end-effector miss with zero joint RMSE and corrects in 0.28 seconds. Problem 5 detects tool mass mismatch through gravity-dependent joint droop with non-zero joint RMSE and corrects in 0.28 seconds.

Three of the five faults are completely invisible in joint space and undetectable by any onboard diagnostic. Two are detectable as joint errors but unidentifiable without the paired simulation approach. All five are fully corrected autonomously with no human intervention.

---

## OpenCAD

OpenCAD is the correction engine at the core of SimCorrect. It provides a programmatic interface for modifying simulation model parameters directly from fault identification results.

```python
from mjcf_correction import Part
Part('grip').set_mass(0.160).export('grip_corrected.xml')
Part('joint1').set_ref(0.0000).export('joint1_corrected.xml')
```

The correction record is written to an XML file, the simulation reloads the corrected model, and the validation pipeline confirms the fault is eliminated before the result is accepted.

The older `from opencad import Part` import remains as a compatibility facade, but new MJCF correction code should use `mjcf_correction` so it is not confused with the full OpenCAD package.

SimCorrect can also consume the CAID design artifact exported by OpenCAD 0.1.1. The artifact gives the correction loop stable parameter names instead of forcing each problem script to infer values from ad hoc files.

```python
from opencad import apply_parameter_patch, load_artifact, make_patch_from_identification

artifact = load_artifact("caid-design.json")
patch = make_patch_from_identification(artifact, identification_result)
corrected_artifact = apply_parameter_patch(artifact, patch)
```

The patch is structured JSON and can be sent back to OpenCAD to update named design parameters. If SimCorrect identifies an internal simulation target such as `link2_length`, the artifact can map it to a company-facing parameter such as `forearm_length` through a `kind="parameter"` simulation tag.

Problem 1 now uses this path when `CAID_DESIGN_ARTIFACT` points at an OpenCAD artifact. Its pure helper lives in `Problem1_ForearmLength/caid_loop.py`, so the artifact-to-patch behavior can be tested without MuJoCo or rendering.

The contract semantics are documented in [`docs/CAID_ARTIFACT_CONTRACT.md`](docs/CAID_ARTIFACT_CONTRACT.md).

---

## Installation

For development and contract tests, see `CONTRIBUTING.md`.

```bash
git clone https://github.com/caid-technologies/SimCorrect.git
cd SimCorrect
uv sync
uv sync --extra demo  # required for MuJoCo render demos
```

---

## Quickstart

```bash
cd Problem5_ToolMassMismatch
uv run --project .. python step0.py
uv run --project .. python demo.py
uv run --project .. python render_demo.py
```

---

## Repository Structure

The repository is organized into five self-contained problem folders, each representing one fault scenario. Shared contract and MJCF correction helpers live at the root. Every problem folder follows the same structure.

```text
SimCorrect/
├── caid_contract.py
├── mjcf_correction.py
├── opencad.py              # compatibility facade
├── Problem1_ForearmLength/
│   ├── render_demo.py
│   ├── sim_pair.py
│   ├── divergence_detector.py
│   ├── parameter_identifier.py
│   ├── correction_and_validation.py
│   ├── demo.py
│   ├── step0.py
│   ├── README.md
│   └── output/
├── Problem2_WristOffset/
├── Problem3_JointFriction/
├── Problem4_JointZeroOffset/
└── Problem5_ToolMassMismatch/
---

## Technical Stack

The physics simulation uses MuJoCo 3.x. CAD correction uses the OpenCAD API. Divergence detection uses end-effector trajectory analysis combined with joint RMSE monitoring. Parameter identification uses sensitivity-based geometric and dynamic analysis. Rendering uses the MuJoCo offscreen renderer with PIL and ffmpeg. The codebase is Python 3.10.

---

## Authors

Shreya Priya is a robotics and autonomy engineer focused on simulation-to-real transfer and autonomous correction pipelines.

Dean (Dien) Hu is a contributor to SimCorrect and the author of OpenCAD.

---

## License

MIT License
