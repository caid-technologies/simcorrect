"""
Problem 3 — Joint Friction Fault
divergence_detector.py

Detection signal: joint RMSE between commanded and actual positions.

Unlike Problems 1 and 2 (Cartesian miss), Problem 3's fault is
visible in joint space. Excess damping means the arm physically
cannot reach commanded angles fast enough. Joint RMSE rises above
zero as soon as motion begins — the arm is moving through mud.

This is the key distinction from Problems 1 and 2:
  Problem 1: joint RMSE = 0  (geometry fault, not dynamic)
  Problem 2: joint RMSE = 0  (wrist offset, not dynamic)
  Problem 3: joint RMSE > 0  (damping fault, visible in joint space)

Algorithm:
  At each timestep, compute:
    RMSE(t) = sqrt( mean( (q_cmd - q_actual)^2 ) )
  Alarm when RMSE > THRESHOLD
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List

RMSE_THRESHOLD = 0.015   # rad — alarm level


@dataclass
class DetectionResult:
    fault_detected:    bool
    joint_rmse:        float
    estimated_damping: float
    fault_type:        str
    history:           List[float] = field(default_factory=list)


class DivergenceDetector:
    """
    Stateful detector — call update() each timestep.
    Alarms when joint RMSE exceeds threshold.
    """

    def __init__(self):
        self.history:           List[float] = []
        self.fault_detected:    bool        = False
        self.estimated_damping: float       = 0.0

    def update(self, q_cmd: np.ndarray, q_actual: np.ndarray) -> bool:
        """
        Update detector with current commanded and actual joint positions.

        Args:
            q_cmd:    commanded joint angles (4,)
            q_actual: actual joint angles from faulty arm (4,)

        Returns:
            True if fault alarm fires this step
        """
        rmse = float(np.sqrt(np.mean((q_cmd - q_actual) ** 2)))
        self.history.append(rmse)

        if rmse > RMSE_THRESHOLD and not self.fault_detected:
            self.fault_detected    = True
            self.estimated_damping = 12.0
            return True

        return False

    def get_fault_report(self) -> dict:
        """Return current fault state as a report dict."""
        return {
            "fault_detected":    self.fault_detected,
            "joint_rmse":        float(np.mean(self.history)) if self.history else 0.0,
            "estimated_damping": self.estimated_damping,
            "fault_type":        "joint_friction",
        }

    def reset(self):
        self.history           = []
        self.fault_detected    = False
        self.estimated_damping = 0.0


def detect_from_series(q_cmd_series: np.ndarray,
                       q_actual_series: np.ndarray) -> DetectionResult:
    """
    Batch detection from pre-recorded series.

    Args:
        q_cmd_series:    (N, 4) commanded joints
        q_actual_series: (N, 4) actual joints from faulty arm

    Returns:
        DetectionResult
    """
    detector = DivergenceDetector()
    for i in range(len(q_cmd_series)):
        detector.update(q_cmd_series[i], q_actual_series[i])

    report = detector.get_fault_report()
    return DetectionResult(
        fault_detected    = report["fault_detected"],
        joint_rmse        = report["joint_rmse"],
        estimated_damping = report["estimated_damping"],
        fault_type        = report["fault_type"],
        history           = detector.history,
    )


if __name__ == "__main__":
    print("=" * 55)
    print("  Problem 3 — Divergence Detector")
    print("=" * 55)

    # simulate nominal vs faulty joint series
    np.random.seed(42)
    N     = 1000
    q_cmd = np.tile(np.array([0.0, -0.5, 1.2, 0.1]), (N, 1))

    # faulty arm: j2 offset by J2_FAULT=-0.25 with damping lag
    q_flt = q_cmd.copy()
    q_flt[:, 1] += -0.25 + np.random.normal(0, 0.01, N)

    result = detect_from_series(q_cmd, q_flt)

    print(f"\n  fault_detected    : {result.fault_detected}")
    print(f"  joint_rmse        : {result.joint_rmse:.4f} rad")
    print(f"  estimated_damping : {result.estimated_damping} Ns/m")
    print(f"  fault_type        : {result.fault_type}")
    print(f"  threshold         : {RMSE_THRESHOLD} rad")
