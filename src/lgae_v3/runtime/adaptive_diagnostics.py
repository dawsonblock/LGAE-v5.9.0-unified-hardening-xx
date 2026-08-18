"""Adaptive geometric diagnostics cascade (Phase 5).

Do not compute expensive geometry uniformly. Diagnostics escalate through
four levels based on a decision function of risk, uncertainty, disagreement,
and mutation authority:

  Level 0 (cheap):     degree statistics, local clustering, Forman curvature,
                       cheap connectivity proxies
  Level 1 (local):     LLY, local spectral quantities, approximate effective
                       resistance
  Level 2 (structural): Sinkhorn ORC, sheaf/connection diagnostics, local
                       persistent-topology tests
  Level 3 (exact):     exact transport, exact/reference spectral calculation,
                       exhaustive invariant verification

High-risk structural mutations automatically demand stronger certification.

This module is an escalation policy + orchestration layer. It does NOT
re-implement the geometry operators; callers register evaluator bundles per
level. The existing ``AdaptiveCurvatureCascade`` remains the curvature-specific
escalation engine and is reused inside Level 0..3 curvature bundles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable

from ..mutations import MutationAuthorityLevel


class DiagnosticLevel(IntEnum):
    L0_CHEAP = 0
    L1_LOCAL = 1
    L2_STRUCTURAL = 2
    L3_EXACT = 3


@dataclass(frozen=True, slots=True)
class DiagnosticEscalationPolicy:
    """Thresholds mapping (risk, uncertainty, disagreement, authority) -> level.

    All inputs are normalized to [0, 1] except authority, which is the
    ``MutationAuthorityLevel`` enum. The selected level is the maximum over
    the per-signal thresholds so that any single high signal can force
    escalation.
    """
    risk_l1: float = 0.3
    risk_l2: float = 0.6
    risk_l3: float = 0.85
    uncertainty_l1: float = 0.35
    uncertainty_l2: float = 0.65
    uncertainty_l3: float = 0.9
    disagreement_l1: float = 0.3
    disagreement_l2: float = 0.6
    disagreement_l3: float = 0.85
    # Authority-driven minimums: structural mutations require at least L1,
    # high-impact at least L2 (added in Phase 7), irreversible at least L3.
    authority_min_level: dict[MutationAuthorityLevel, DiagnosticLevel] = field(
        default_factory=lambda: {
            MutationAuthorityLevel.REVERSIBLE: DiagnosticLevel.L0_CHEAP,
            MutationAuthorityLevel.STRUCTURAL: DiagnosticLevel.L1_LOCAL,
            MutationAuthorityLevel.HIGH_IMPACT: DiagnosticLevel.L2_STRUCTURAL,
            MutationAuthorityLevel.IRREVERSIBLE: DiagnosticLevel.L3_EXACT,
        }
    )

    def level_for(
        self,
        *,
        risk: float = 0.0,
        uncertainty: float = 0.0,
        disagreement: float = 0.0,
        authority: MutationAuthorityLevel = MutationAuthorityLevel.REVERSIBLE,
    ) -> DiagnosticLevel:
        """Select the diagnostic level as f(risk, uncertainty, disagreement, authority)."""
        for v in (risk, uncertainty, disagreement):
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError("risk/uncertainty/disagreement must lie in [0,1]")
        level = DiagnosticLevel.L0_CHEAP
        if risk >= self.risk_l3 or uncertainty >= self.uncertainty_l3 or disagreement >= self.disagreement_l3:
            level = DiagnosticLevel.L3_EXACT
        elif risk >= self.risk_l2 or uncertainty >= self.uncertainty_l2 or disagreement >= self.disagreement_l2:
            level = max(level, DiagnosticLevel.L2_STRUCTURAL)
        elif risk >= self.risk_l1 or uncertainty >= self.uncertainty_l1 or disagreement >= self.disagreement_l1:
            level = max(level, DiagnosticLevel.L1_LOCAL)
        # Authority-driven minimum.
        auth_min = self.authority_min_level.get(authority, DiagnosticLevel.L0_CHEAP)
        return max(level, auth_min)


@dataclass(slots=True)
class DiagnosticResult:
    """Outcome of one diagnostic cascade evaluation."""
    level: DiagnosticLevel
    metrics: dict[str, Any] = field(default_factory=dict)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    escalated: bool = False

    @property
    def is_exact(self) -> bool:
        return self.level == DiagnosticLevel.L3_EXACT

    def to_log(self) -> dict[str, Any]:
        return {
            "level": int(self.level),
            "level_name": self.level.name,
            "escalated": bool(self.escalated),
            "metrics": self.metrics,
            "evaluations": self.evaluations,
        }


# A level evaluator returns a dict of metric name -> value.
LevelEvaluator = Callable[[], dict[str, Any]]


class DiagnosticCascade:
    """Adaptive L0..L3 diagnostic escalation.

    Callers register an evaluator bundle per level. ``evaluate()`` selects the
    level via the escalation policy and runs every level up to and including
    the selected one (so cheap diagnostics are always available and the
    cascade accumulates evidence). The result records which levels ran.
    """

    def __init__(
        self,
        evaluators: dict[DiagnosticLevel, LevelEvaluator],
        policy: DiagnosticEscalationPolicy | None = None,
    ) -> None:
        self.evaluators: dict[DiagnosticLevel, LevelEvaluator] = dict(evaluators)
        self.policy = policy or DiagnosticEscalationPolicy()

    def register(self, level: DiagnosticLevel, fn: LevelEvaluator) -> None:
        self.evaluators[level] = fn

    def evaluate(
        self,
        *,
        risk: float = 0.0,
        uncertainty: float = 0.0,
        disagreement: float = 0.0,
        authority: MutationAuthorityLevel = MutationAuthorityLevel.REVERSIBLE,
        force_level: DiagnosticLevel | None = None,
    ) -> DiagnosticResult:
        target = DiagnosticLevel(force_level) if force_level is not None else self.policy.level_for(
            risk=risk, uncertainty=uncertainty, disagreement=disagreement, authority=authority,
        )
        metrics: dict[str, Any] = {}
        evaluations: list[dict[str, Any]] = []
        for level in DiagnosticLevel:
            if level > target:
                break
            fn = self.evaluators.get(level)
            if fn is None:
                continue
            out = fn()
            evaluations.append({"level": int(level), "name": level.name, "metrics": out})
            metrics.update(out)
        return DiagnosticResult(
            level=target, metrics=metrics, evaluations=evaluations,
            escalated=target > DiagnosticLevel.L0_CHEAP,
        )
