"""Target definitions for v6.0-exp4.

Preserves multiple views of the outcome:

- realized_delta: raw ΔU
- sign(realized_delta): binary
- normalized_delta: ΔU / max(ε, |U_before|)
- utility_bucket: discretized
- candidate-relative rank: within candidate set

For risk, preserves components:
- instability risk
- constraint-margin risk
- OOD risk
- topology-fragmentation risk
- rollback/failure risk

For cost, uses actual measured cost:
- wall-clock time
- shadow evaluation count
- structural complexity increase
- edge count change
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import math
import numpy as np


class TargetType(str, Enum):
    """Types of prediction targets."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    RANKING = "ranking"


@dataclass(frozen=True, slots=True)
class TargetDefinition:
    """Definition of a prediction target."""
    name: str
    target_type: TargetType
    description: str
    units: str = ""
    lower_bound: float | None = None
    upper_bound: float | None = None

    def to_log(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_type": self.target_type.value,
            "description": self.description,
            "units": self.units,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


# Canonical target definitions.
TARGET_REALIZED_DELTA = TargetDefinition(
    name="realized_delta",
    target_type=TargetType.REGRESSION,
    description="Raw realized utility delta",
    units="utility",
)

TARGET_SIGN_DELTA = TargetDefinition(
    name="sign_delta",
    target_type=TargetType.CLASSIFICATION,
    description="Sign of realized utility delta (1 if ΔU > 0, else 0)",
)

TARGET_NORMALIZED_DELTA = TargetDefinition(
    name="normalized_delta",
    target_type=TargetType.REGRESSION,
    description="ΔU / max(ε, |U_before|)",
    units="fraction",
)

TARGET_UTILITY_BUCKET = TargetDefinition(
    name="utility_bucket",
    target_type=TargetType.CLASSIFICATION,
    description="Discretized utility delta bucket",
)

TARGET_CANDIDATE_RANK = TargetDefinition(
    name="candidate_rank",
    target_type=TargetType.RANKING,
    description="Candidate-relative rank within candidate set",
)

TARGET_RISK = TargetDefinition(
    name="risk",
    target_type=TargetType.REGRESSION,
    description="Aggregate risk score",
    units="risk",
    lower_bound=0.0,
)

TARGET_COST = TargetDefinition(
    name="cost",
    target_type=TargetType.REGRESSION,
    description="Measured compute cost",
    units="compute",
    lower_bound=0.0,
)

# Risk components.
RISK_COMPONENTS = (
    TargetDefinition("instability_risk", TargetType.REGRESSION, "Structural instability risk"),
    TargetDefinition("constraint_margin_risk", TargetType.REGRESSION, "Constraint margin violation risk"),
    TargetDefinition("ood_risk", TargetType.REGRESSION, "Out-of-distribution risk"),
    TargetDefinition("fragmentation_risk", TargetType.REGRESSION, "Topology fragmentation risk"),
    TargetDefinition("rollback_risk", TargetType.REGRESSION, "Rollback/failure risk"),
)

# Cost components.
COST_COMPONENTS = (
    TargetDefinition("wall_clock_seconds", TargetType.REGRESSION, "Wall-clock time", "seconds", 0.0),
    TargetDefinition("shadow_executions", TargetType.REGRESSION, "Shadow evaluation count", "count", 0.0),
    TargetDefinition("complexity_increase", TargetType.REGRESSION, "Structural complexity increase", "delta", 0.0),
    TargetDefinition("edge_count_change", TargetType.REGRESSION, "Edge count change", "delta"),
)


# ---------------------------------------------------------------------------
# Target transforms.
# ---------------------------------------------------------------------------

def compute_sign_delta(realized_delta: float) -> int:
    """Compute sign(ΔU) as binary {0, 1}."""
    return 1 if realized_delta > 0 else 0


def compute_normalized_delta(realized_delta: float, utility_before: float, eps: float = 1e-6) -> float:
    """Compute normalized ΔU = ΔU / max(ε, |U_before|)."""
    return float(realized_delta) / max(eps, abs(float(utility_before)))


def compute_utility_bucket(realized_delta: float, thresholds: tuple[float, ...] = (-0.1, -0.01, 0.01, 0.1)) -> int:
    """Discretize ΔU into buckets.

    Buckets:
        0: strongly negative (ΔU < -0.1)
        1: weakly negative (-0.1 ≤ ΔU < -0.01)
        2: neutral (-0.01 ≤ ΔU ≤ 0.01)
        3: weakly positive (0.01 < ΔU ≤ 0.1)
        4: strongly positive (ΔU > 0.1)
    """
    for i, t in enumerate(thresholds):
        if realized_delta < t:
            return i
    return len(thresholds)


def compute_candidate_ranks(realized_deltas: list[float]) -> list[int]:
    """Compute candidate-relative ranks (0 = best).

    Returns ranks such that rank 0 corresponds to the highest ΔU.
    """
    n = len(realized_deltas)
    if n == 0:
        return []
    # Sort by delta descending, then map back to original indices.
    indexed = sorted(range(n), key=lambda i: -realized_deltas[i])
    ranks = [0] * n
    for rank, idx in enumerate(indexed):
        ranks[idx] = rank
    return ranks


def compute_pairwise_labels(realized_deltas: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Generate pairwise ranking labels.

    For each pair (i, j) from the same state:
        y = 1[ΔU_i > ΔU_j]

    Returns:
        pairs: (n_pairs, 2) array of (i, j) index pairs.
        labels: (n_pairs,) array of {0, 1} labels.
    """
    n = len(realized_deltas)
    pairs = []
    labels = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append([i, j])
            labels.append(1 if realized_deltas[i] > realized_deltas[j] else 0)
    if not pairs:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float)
    return np.array(pairs, dtype=int), np.array(labels, dtype=float)


def aggregate_risk(risk_components: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Aggregate risk components into a single risk score.

    Args:
        risk_components: Dictionary of component name → value.
        weights: Optional weights for each component. If None, equal weights.

    Returns:
        Weighted sum of risk components.
    """
    if not risk_components:
        return 0.0
    if weights is None:
        weights = {k: 1.0 / len(risk_components) for k in risk_components}
    total = 0.0
    for comp, val in risk_components.items():
        w = weights.get(comp, 0.0)
        total += w * float(val)
    return max(0.0, total)


def aggregate_cost(cost_components: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Aggregate cost components into a single cost score."""
    return aggregate_risk(cost_components, weights)  # same logic


# ---------------------------------------------------------------------------
# Target set.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TargetSet:
    """A set of targets for a single training/evaluation run."""
    regression_targets: tuple[str, ...] = ()
    classification_targets: tuple[str, ...] = ()
    ranking_targets: tuple[str, ...] = ()

    @property
    def all_targets(self) -> tuple[str, ...]:
        return self.regression_targets + self.classification_targets + self.ranking_targets

    def to_log(self) -> dict[str, Any]:
        return {
            "regression_targets": list(self.regression_targets),
            "classification_targets": list(self.classification_targets),
            "ranking_targets": list(self.ranking_targets),
        }


# Default target set for exp4.
DEFAULT_TARGETS = TargetSet(
    regression_targets=("realized_delta", "normalized_delta", "risk", "cost"),
    classification_targets=("sign_delta", "utility_bucket"),
    ranking_targets=("candidate_rank",),
)
