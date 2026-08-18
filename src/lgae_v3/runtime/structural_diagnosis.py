"""Phase 12, 13, 14: Structural Diagnosis, Local Search & Attention Budget.

Structural diagnosis provides a semantic layer mapping geometric/graph indicators
to actionable diagnoses that condition candidate proposal generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch import Tensor

from ..types import GraphBuffers


class DiagnosisType(str, Enum):
    OVERSQUASHING = "OVERSQUASHING"
    UNDERCONNECTED_REGION = "UNDERCONNECTED_REGION"
    EXCESSIVE_DENSITY = "EXCESSIVE_DENSITY"
    STRUCTURAL_BOTTLENECK = "STRUCTURAL_BOTTLENECK"
    HUB_OVERLOAD = "HUB_OVERLOAD"
    COMMUNITY_FRAGMENTATION = "COMMUNITY_FRAGMENTATION"
    TRANSPORT_INSTABILITY = "TRANSPORT_INSTABILITY"
    SPECTRAL_DEGRADATION = "SPECTRAL_DEGRADATION"
    HIGH_EPISTEMIC_REGION = "HIGH_EPISTEMIC_REGION"
    REDUNDANT_PATHS = "REDUNDANT_PATHS"


@dataclass(frozen=True, slots=True)
class StructuralDiagnosis:
    """A first-class structural diagnosis emitted during reasoning."""
    diagnosis_type: DiagnosisType
    severity: float  # [0.0, 1.0]
    confidence: float  # [0.0, 1.0]
    affected_nodes: tuple[int, ...] = ()
    affected_edges: tuple[tuple[int, int], ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis_type": self.diagnosis_type.value,
            "severity": float(self.severity),
            "confidence": float(self.confidence),
            "affected_nodes": list(self.affected_nodes),
            "affected_edges": [list(e) for e in self.affected_edges],
            "evidence": self.evidence,
        }


class StructuralDiagnoser:
    """Diagnoses structural deficits from graph topology, curvature, and audit data."""

    def diagnose(
        self,
        graph: GraphBuffers,
        audit: Any,
        epistemic_uncertainty: float = 0.0,
    ) -> list[StructuralDiagnosis]:
        diagnoses: list[StructuralDiagnosis] = []

        # 1. Check Spectral Gap / Oversquashing
        gap = getattr(audit, "spectral_gap", None)
        if gap is not None and gap < 0.15:
            severity = min(1.0, (0.15 - float(gap)) / 0.15)
            diagnoses.append(StructuralDiagnosis(
                diagnosis_type=DiagnosisType.OVERSQUASHING,
                severity=round(severity, 4),
                confidence=0.85,
                affected_nodes=(0, 1),
                evidence={"spectral_gap": float(gap)},
            ))

        # 2. Check Negative Curvature (Ricci min)
        ricci_min = getattr(audit, "ricci_min", None)
        if ricci_min is not None and ricci_min < -0.3:
            severity = min(1.0, abs(float(ricci_min)))
            diagnoses.append(StructuralDiagnosis(
                diagnosis_type=DiagnosisType.STRUCTURAL_BOTTLENECK,
                severity=round(severity, 4),
                confidence=0.8,
                evidence={"ricci_min": float(ricci_min)},
            ))

        # 3. Check High Epistemic Uncertainty Region
        if epistemic_uncertainty > 0.4:
            diagnoses.append(StructuralDiagnosis(
                diagnosis_type=DiagnosisType.HIGH_EPISTEMIC_REGION,
                severity=min(1.0, float(epistemic_uncertainty)),
                confidence=0.75,
                evidence={"epistemic_uncertainty": float(epistemic_uncertainty)},
            ))

        # 4. Check Density
        n_edges = float(graph.valid.sum().item()) if hasattr(graph, "valid") and hasattr(graph.valid, "sum") else 0.0
        cap = float(len(graph.valid)) if hasattr(graph, "valid") else max(1.0, n_edges * 2)
        if n_edges > 0 and (n_edges / cap) > 0.8:
            diagnoses.append(StructuralDiagnosis(
                diagnosis_type=DiagnosisType.EXCESSIVE_DENSITY,
                severity=round((n_edges / cap), 4),
                confidence=0.9,
                evidence={"density": n_edges / cap},
            ))

        return diagnoses


class StructuralAttentionBudget:
    """Calculates region-wise attention priorities to allocate compute budgets."""

    def __init__(
        self,
        alpha_severity: float = 0.4,
        beta_uncertainty: float = 0.3,
        gamma_utility: float = 0.2,
        delta_recent_change: float = 0.1,
    ) -> None:
        self.alpha = alpha_severity
        self.beta = beta_uncertainty
        self.gamma = gamma_utility
        self.delta = delta_recent_change

    def compute_region_priority(
        self,
        severity: float,
        uncertainty: float,
        utility_impact: float = 0.0,
        recent_change: float = 0.0,
    ) -> float:
        """Priority(r) = alpha * Severity + beta * Uncertainty + gamma * UtilityImpact + delta * RecentChange."""
        score = (
            self.alpha * severity
            + self.beta * uncertainty
            + self.gamma * utility_impact
            + self.delta * recent_change
        )
        return round(float(score), 4)
