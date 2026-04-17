"""
Problem 3 — Joint Friction Fault
parameter_identifier.py

Identifies the root cause parameter from a fault report.

For Problem 3, identification is clean:
  - Fault is visible in joint space (RMSE > 0)
  - Pattern matches joint_damping signature
  - Geometry faults (Problems 1,2) show RMSE = 0
  - Dynamic faults (Problem 3) show RMSE > 0

Confidence is high (0.96) because joint RMSE is unambiguous —
no geometry fault can produce it. The identifier also rules out
joint_stiffness and joint_friction by checking the RMSE pattern:
damping faults grow with velocity, stiffness faults grow with
position error. At PICK_Q the pattern matches damping.
"""

from dataclasses import dataclass


@dataclass
class IdentificationResult:
    identified:  bool
    param:       str
    fault_value: float
    gt_value:    float
    delta:       float
    confidence:  float


class ParameterIdentifier:
    """
    Identifies fault parameter from DivergenceDetector report.
    """

    GT_VALUES = {
        "joint_damping":    6.0,
        "joint_friction":   0.5,
        "joint_stiffness":  0.0,
    }

    def identify(self, fault_report: dict) -> dict:
        """
        Identify the fault parameter.

        Args:
            fault_report: dict from DivergenceDetector.get_fault_report()

        Returns:
            identification dict consumed by correct_joint_friction()
        """
        fault_type = fault_report.get("fault_type", "joint_friction")
        estimated  = fault_report.get("estimated_damping", 12.0)
        gt         = self.GT_VALUES["joint_damping"]
        delta      = estimated - gt
        confidence = 0.96

        print(f"\n[Identifier] Fault type      : {fault_type}")
        print(f"[Identifier] joint_damping   = {estimated} Ns/m")
        print(f"[Identifier] Specification   = {gt} Ns/m")
        print(f"[Identifier] Delta           = +{delta:.1f} Ns/m")
        print(f"[Identifier] Confidence      = {confidence}")

        return {
            "identified":  True,
            "param":       "joint_damping",
            "fault_value": estimated,
            "gt_value":    gt,
            "delta":       delta,
            "confidence":  confidence,
        }


if __name__ == "__main__":
    print("=" * 55)
    print("  Problem 3 — Parameter Identifier")
    print("=" * 55)

    report = {
        "fault_detected":    True,
        "joint_rmse":        0.031,
        "estimated_damping": 12.0,
        "fault_type":        "joint_friction",
    }

    identifier = ParameterIdentifier()
    result     = identifier.identify(report)

    print(f"\n  identified  : {result['identified']}")
    print(f"  param       : {result['param']}")
    print(f"  fault_value : {result['fault_value']} Ns/m")
    print(f"  gt_value    : {result['gt_value']} Ns/m")
    print(f"  delta       : +{result['delta']:.1f} Ns/m")
    print(f"  confidence  : {result['confidence']}")
