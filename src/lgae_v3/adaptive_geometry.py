"""v5.7 adaptive geometry runtime primitives.

Selects geometric work according to declared dependency footprints, numerical
health, ambiguity, and decision risk. Exact operators remain authoritative.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, TypeVar
import time
import torch
from torch import Tensor
from .cache_coherence import ChangeKind, SpatialCacheDependency


@dataclass(frozen=True, slots=True)
class OperatorDependencyFootprint:
    name: str
    changes: ChangeKind
    support_radius: int | None
    exact_reference: bool = False

    def as_cache_dependency(self) -> SpatialCacheDependency:
        return SpatialCacheDependency(self.changes, radius=self.support_radius)


DEFAULT_OPERATOR_FOOTPRINTS: dict[str, OperatorDependencyFootprint] = {
    "forman": OperatorDependencyFootprint("forman", ChangeKind.TOPOLOGY | ChangeKind.WEIGHTS, 1),
    "lly": OperatorDependencyFootprint("lly", ChangeKind.TOPOLOGY | ChangeKind.WEIGHTS | ChangeKind.LENGTHS, 2),
    "ollivier_sinkhorn": OperatorDependencyFootprint("ollivier_sinkhorn", ChangeKind.TOPOLOGY | ChangeKind.WEIGHTS | ChangeKind.LENGTHS, 2),
    "ollivier_exact": OperatorDependencyFootprint("ollivier_exact", ChangeKind.TOPOLOGY | ChangeKind.WEIGHTS | ChangeKind.LENGTHS, 2, True),
    "spectral_gap": OperatorDependencyFootprint("spectral_gap", ChangeKind.TOPOLOGY | ChangeKind.WEIGHTS, None, True),
    "persistent_homology": OperatorDependencyFootprint("persistent_homology", ChangeKind.TOPOLOGY | ChangeKind.WEIGHTS | ChangeKind.LENGTHS, None, True),
    "sheaf_transport": OperatorDependencyFootprint("sheaf_transport", ChangeKind.TOPOLOGY | ChangeKind.WEIGHTS | ChangeKind.FIBERS | ChangeKind.GAUGE, 1),
}


class DependencyRegistry:
    def __init__(self) -> None:
        self._items = dict(DEFAULT_OPERATOR_FOOTPRINTS)

    def register(self, footprint: OperatorDependencyFootprint) -> None:
        self._items[footprint.name] = footprint

    def get(self, name: str) -> OperatorDependencyFootprint:
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(f"unknown geometry operator dependency: {name}") from exc

    def cache_dependency(self, name: str) -> SpatialCacheDependency:
        return self.get(name).as_cache_dependency()


@dataclass(frozen=True, slots=True)
class OrthogonalityHealth:
    error: float
    action: str
    repaired: Tensor | None = None


def orthogonality_error(W: Tensor) -> Tensor:
    if W.ndim < 2 or W.shape[-1] != W.shape[-2]:
        raise ValueError("orthogonality monitor requires square matrices")
    d = W.shape[-1]
    eye = torch.eye(d, device=W.device, dtype=W.dtype)
    return torch.linalg.matrix_norm(W.transpose(-1, -2) @ W - eye, ord="fro", dim=(-2, -1))


def monitor_orthogonality(
    W: Tensor,
    *,
    warn_threshold: float = 1e-5,
    repair_threshold: float = 1e-3,
) -> OrthogonalityHealth:
    if not (0 <= warn_threshold <= repair_threshold):
        raise ValueError("require 0 <= warn_threshold <= repair_threshold")
    err = float(orthogonality_error(W).max().detach().cpu())
    if err <= warn_threshold:
        return OrthogonalityHealth(err, "healthy", None)
    if err <= repair_threshold:
        return OrthogonalityHealth(err, "warn", None)
    U, _, Vh = torch.linalg.svd(W, full_matrices=False)
    repaired = U @ Vh
    return OrthogonalityHealth(err, "repaired", repaired)


class CurvatureStage(str, Enum):
    FORM = "forman"
    LLY = "lly"
    SINKHORN = "ollivier_sinkhorn"
    EXACT = "ollivier_exact"


@dataclass(frozen=True, slots=True)
class GeometryEstimate:
    stage: CurvatureStage
    value: float
    ambiguity: float = 0.0
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class CascadePolicy:
    ambiguity_threshold: float = 0.20
    high_risk_threshold: float = 0.75
    exact_risk_threshold: float = 0.95
    max_latency_ms: float | None = None


@dataclass(slots=True)
class CascadeResult:
    selected: GeometryEstimate
    evaluations: list[GeometryEstimate] = field(default_factory=list)
    escalated: bool = False


class AdaptiveCurvatureCascade:
    """Cheap-to-expensive curvature escalation with exact-reference endpoint.

    Evaluators return ``(value, ambiguity)``. Ambiguity is normalized to [0,1].
    Runtime latency is measured here so policy can enforce a geometry budget.
    """
    ORDER = (CurvatureStage.FORM, CurvatureStage.LLY, CurvatureStage.SINKHORN, CurvatureStage.EXACT)

    def __init__(self, evaluators: dict[CurvatureStage, Callable[[], tuple[float, float]]], policy: CascadePolicy | None = None) -> None:
        self.evaluators = dict(evaluators)
        self.policy = policy or CascadePolicy()

    def _run(self, stage: CurvatureStage) -> GeometryEstimate:
        fn = self.evaluators.get(stage)
        if fn is None:
            raise KeyError(f"missing evaluator for {stage.value}")
        t0 = time.perf_counter()
        value, ambiguity = fn()
        latency = (time.perf_counter() - t0) * 1000.0
        ambiguity = float(ambiguity)
        if not (0.0 <= ambiguity <= 1.0):
            raise ValueError("geometry ambiguity must lie in [0,1]")
        return GeometryEstimate(stage, float(value), ambiguity, latency)

    def evaluate(self, *, risk: float = 0.0, require_exact: bool = False) -> CascadeResult:
        risk = float(risk)
        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk must lie in [0,1]")
        records: list[GeometryEstimate] = []
        for idx, stage in enumerate(self.ORDER):
            rec = self._run(stage); records.append(rec)
            if require_exact:
                if stage is CurvatureStage.EXACT:
                    return CascadeResult(rec, records, True)
                continue
            must_escalate = rec.ambiguity > self.policy.ambiguity_threshold
            if risk >= self.policy.high_risk_threshold and stage in {CurvatureStage.FORM, CurvatureStage.LLY}:
                must_escalate = True
            if risk >= self.policy.exact_risk_threshold and stage is not CurvatureStage.EXACT:
                must_escalate = True
            if self.policy.max_latency_ms is not None and sum(r.latency_ms for r in records) >= self.policy.max_latency_ms:
                return CascadeResult(rec, records, idx > 0)
            if not must_escalate or stage is CurvatureStage.EXACT:
                return CascadeResult(rec, records, idx > 0)
        return CascadeResult(records[-1], records, True)
