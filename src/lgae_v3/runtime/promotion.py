"""Promotion gates (Phase 46).

A model/policy cannot be promoted to a higher maturity level until all
required gates pass. Levels are ordered:

  EXPERIMENTAL  -> initial implementation, no gates required
  CANDIDATE     -> safety gate passes
  QUALIFIED     -> safety + scientific + performance gates pass
  PRODUCTION    -> all gates pass + signed release checkpoint

Promotion is explicit and auditable. A gate that did not run is NOT the same
as a gate that passed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from .qualification import SafetyQualificationReport
from .scientific_qualification import ScientificQualificationReport
from .performance_qualification import PerformanceQualificationReport


class PromotionLevel(IntEnum):
    EXPERIMENTAL = 0
    CANDIDATE = 1
    QUALIFIED = 2
    PRODUCTION = 3


@dataclass(frozen=True, slots=True)
class GateStatus:
    """Status of one promotion gate."""
    name: str
    ran: bool
    passed: bool
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def status_str(self) -> str:
        if not self.ran:
            return "not_run"
        return "pass" if self.passed else "fail"

    def to_log(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ran": bool(self.ran),
            "passed": bool(self.passed),
            "status": self.status_str,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class PromotionReport:
    """Aggregate promotion decision report."""
    current_level: PromotionLevel
    target_level: PromotionLevel
    gates: list[GateStatus] = field(default_factory=list)
    signed_checkpoint: str | None = None

    @property
    def all_required_gates_passed(self) -> bool:
        """True only if every required gate ran and passed."""
        return all(g.ran and g.passed for g in self.gates)

    @property
    def promotion_approved(self) -> bool:
        if not self.all_required_gates_passed:
            return False
        if self.target_level == PromotionLevel.PRODUCTION:
            return self.signed_checkpoint is not None
        return True

    def to_log(self) -> dict[str, Any]:
        return {
            "current_level": self.current_level.name,
            "target_level": self.target_level.name,
            "promotion_approved": self.promotion_approved,
            "all_required_gates_passed": self.all_required_gates_passed,
            "signed_checkpoint": self.signed_checkpoint,
            "gates": [g.to_log() for g in self.gates],
        }


def evaluate_promotion(
    *,
    current_level: PromotionLevel,
    target_level: PromotionLevel,
    safety_report: SafetyQualificationReport | None = None,
    scientific_report: ScientificQualificationReport | None = None,
    performance_report: PerformanceQualificationReport | None = None,
    signed_checkpoint: str | None = None,
) -> PromotionReport:
    """Evaluate whether a model/policy can be promoted from current to target.

    Required gates by target level:
      CANDIDATE:  safety
      QUALIFIED:  safety + scientific + performance
      PRODUCTION: safety + scientific + performance + signed checkpoint
    """
    if target_level <= current_level:
        return PromotionReport(
            current_level=current_level, target_level=target_level,
            gates=[GateStatus(name="noop", ran=True, passed=True)],
        )

    gates: list[GateStatus] = []

    # Safety gate is required for all promotions above EXPERIMENTAL.
    if target_level >= PromotionLevel.CANDIDATE:
        if safety_report is not None:
            gates.append(GateStatus(
                name="safety", ran=True, passed=safety_report.all_passed,
                evidence=safety_report.to_log(),
            ))
        else:
            gates.append(GateStatus(name="safety", ran=False, passed=False))

    # Scientific + performance gates required for QUALIFIED and above.
    if target_level >= PromotionLevel.QUALIFIED:
        if scientific_report is not None:
            gates.append(GateStatus(
                name="scientific", ran=True, passed=scientific_report.all_gates_passed,
                evidence=scientific_report.to_log(),
            ))
        else:
            gates.append(GateStatus(name="scientific", ran=False, passed=False))
        if performance_report is not None:
            # v5.11-RC Phase 18: Performance gate requires PASS, not just
            # measurement. The previous check (len(measured_tiers) > 0)
            # passed even when all tiers FAILED.
            from .performance_qualification import MeasurementStatus, ScaleTier
            required_tiers = [ScaleTier.S]
            if target_level >= PromotionLevel.PRODUCTION:
                required_tiers = [ScaleTier.S, ScaleTier.M]
            passed = all(
                performance_report.result_for(tier) is not None
                and performance_report.result_for(tier).qualification_status == MeasurementStatus.PASS
                for tier in required_tiers
            )
            gates.append(GateStatus(
                name="performance", ran=True, passed=passed,
                evidence=performance_report.to_log(),
            ))
        else:
            gates.append(GateStatus(name="performance", ran=False, passed=False))

    return PromotionReport(
        current_level=current_level,
        target_level=target_level,
        gates=gates,
        signed_checkpoint=signed_checkpoint,
    )


def assert_promotion(report: PromotionReport) -> PromotionReport:
    """Raise if promotion is not approved. Returns the report on success."""
    if not report.promotion_approved:
        failed = [g.name for g in report.gates if not (g.ran and g.passed)]
        raise PromotionGateError(
            f"promotion from {report.current_level.name} to {report.target_level.name} denied; "
            f"gates not passed: {failed}"
        )
    return report


class PromotionGateError(RuntimeError):
    """Raised when a promotion gate does not pass."""
