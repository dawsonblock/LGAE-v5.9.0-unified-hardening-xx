"""Phase 2 contract: ReasoningResult.

Output of the reason() phase: diagnostics, uncertainty, and structural deficits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import PhaseResult


@dataclass(frozen=True, slots=True)
class StructuralDeficit:
    """A diagnosed structural problem in the graph.

    Attributes:
        deficit_type: category (oversquashing, over_smoothing, etc.)
        location: edge or region identifier
        severity: 0.0 to 1.0
        confidence: 0.0 to 1.0
        evidence: supporting diagnostic data
    """
    deficit_type: str
    location: str
    severity: float
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DiagnosticBundle:
    """Aggregated diagnostics from the adaptive cascade."""
    curvature_anomalies: tuple[dict[str, Any], ...] = ()
    spectral_anomalies: tuple[dict[str, Any], ...] = ()
    sheaf_inconsistencies: tuple[dict[str, Any], ...] = ()
    topology_bottlenecks: tuple[dict[str, Any], ...] = ()
    transport_errors: tuple[dict[str, Any], ...] = ()
    diagnostic_level: str = "L0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "curvature_anomalies": list(self.curvature_anomalies),
            "spectral_anomalies": list(self.spectral_anomalies),
            "sheaf_inconsistencies": list(self.sheaf_inconsistencies),
            "topology_bottlenecks": list(self.topology_bottlenecks),
            "transport_errors": list(self.transport_errors),
            "diagnostic_level": self.diagnostic_level,
        }


@dataclass(frozen=True, slots=True)
class ReasoningResult(PhaseResult):
    """Output of the reason() phase.

    Attributes:
        diagnostics: aggregated diagnostic bundle
        epistemic_uncertainty: model uncertainty estimate
        aleatoric_uncertainty: data uncertainty estimate
        ood_score: out-of-distribution score (higher = more OOD)
        deficits: tuple of diagnosed structural deficits
        diagnoses: first-class structural diagnoses (Phase 12)
    """
    diagnostics: DiagnosticBundle = field(default_factory=DiagnosticBundle)
    epistemic_uncertainty: float = 0.0
    aleatoric_uncertainty: float = 0.0
    ood_score: float = 0.0
    deficits: tuple[StructuralDeficit, ...] = ()
    diagnoses: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "state_version": self.state_version,
            "state_hash": self.state_hash,
            "diagnostics": self.diagnostics.to_dict(),
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "ood_score": self.ood_score,
            "deficits": [
                {"deficit_type": d.deficit_type, "location": d.location,
                 "severity": d.severity, "confidence": d.confidence}
                for d in self.deficits
            ],
            "diagnoses": [
                d.to_dict() if hasattr(d, "to_dict") else str(d)
                for d in self.diagnoses
            ],
        }
