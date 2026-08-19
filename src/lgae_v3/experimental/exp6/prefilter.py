"""Phase 7-8: Learned candidate prefilter with UCB pruning.

The learned model scores candidates and reduces the number that
need exact evaluation. Uses uncertainty-aware optimistic bounds
to avoid discarding unfamiliar but potentially strong actions.

Architecture:
    1000 candidates
        ↓
    world model scores (delta, uncertainty, risk)
        ↓
    UCB ranking: UCB(a) = ΔU(a) + κ·σ(a)
        ↓
    top-K candidates
        ↓
    exact shadow evaluation
        ↓
    v5.11 governor
        ↓
    CommitChannel

Primary metric: OracleRecall@K
Did the learned filter preserve the actually best candidate?
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Candidate:
    """A structural mutation candidate."""
    action_type: str
    action_target: dict[str, Any]
    z_t: np.ndarray  # state before
    a_t: np.ndarray  # encoded action
    # Filled by the learned scorer.
    predicted_delta: np.ndarray | None = None
    predicted_uncertainty: float = 0.0
    predicted_utility: float = 0.0
    ucb_score: float = 0.0
    # Filled by exact evaluation.
    exact_utility: float | None = None
    exact_delta: np.ndarray | None = None


@dataclass
class PrefilterResult:
    """Result of candidate prefiltering."""
    n_total: int = 0
    n_retained: int = 0
    retained_indices: list[int] = field(default_factory=list)
    oracle_best_index: int = -1
    oracle_best_retained: bool = False
    recall_at_k: dict[int, float] = field(default_factory=dict)
    best_retained_utility: float = 0.0
    exact_evaluations_saved: float = 0.0
    mean_predicted_utility: float = 0.0
    mean_uncertainty: float = 0.0

    def to_log(self) -> dict[str, Any]:
        return {
            "n_total": int(self.n_total),
            "n_retained": int(self.n_retained),
            "oracle_best_retained": bool(self.oracle_best_retained),
            "recall_at_k": {str(k): float(v) for k, v in self.recall_at_k.items()},
            "best_retained_utility": float(self.best_retained_utility),
            "exact_evaluations_saved": float(self.exact_evaluations_saved),
            "mean_predicted_utility": float(self.mean_predicted_utility),
            "mean_uncertainty": float(self.mean_uncertainty),
        }


def score_candidates(
    candidates: list[Candidate],
    base_model: Any,
    calibration: Any,
    *,
    kappa: float = 1.0,
) -> None:
    """Score candidates using the calibrated world model.

    Fills in:
    - predicted_delta
    - predicted_uncertainty (from ensemble if available)
    - predicted_utility (from delta)
    - ucb_score

    UCB(a) = ΔU(a) + κ·σ(a)

    The UCB score prevents discarding high-uncertainty candidates
    that might be strong.
    """
    if not candidates:
        return

    # Batch predict.
    z_batch = np.array([c.z_t for c in candidates])
    a_batch = np.array([c.a_t for c in candidates])

    # Raw delta predictions.
    raw_deltas = base_model.predict_raw_batch(z_batch, a_batch)

    # Apply calibration.
    adapted_deltas = calibration.apply_batch(raw_deltas)

    # Uncertainty (from ensemble if available, else 0).
    uncertainties = np.zeros(len(candidates))
    if hasattr(base_model, 'predict_uncertainty_batch'):
        uncertainties = base_model.predict_uncertainty_batch(z_batch, a_batch)

    for i, cand in enumerate(candidates):
        cand.predicted_delta = adapted_deltas[i]
        cand.predicted_uncertainty = float(uncertainties[i])
        # Utility proxy: mean of delta vector (higher = better).
        cand.predicted_utility = float(np.mean(adapted_deltas[i]))
        # UCB score.
        cand.ucb_score = cand.predicted_utility + kappa * cand.predicted_uncertainty


def prefilter_candidates(
    candidates: list[Candidate],
    *,
    k_values: list[int] | None = None,
) -> PrefilterResult:
    """Filter candidates by UCB score and compute oracle recall.

    Args:
        candidates: Scored candidates (must have ucb_score and exact_utility).
        k_values: K values for recall@K computation.

    Returns:
        PrefilterResult with recall metrics.
    """
    if k_values is None:
        k_values = [10, 25, 50, 100]

    n = len(candidates)
    if n == 0:
        return PrefilterResult()

    result = PrefilterResult(n_total=n)

    # Sort by UCB score (descending).
    sorted_indices = sorted(range(n), key=lambda i: candidates[i].ucb_score, reverse=True)

    # Find oracle best (by exact utility).
    exact_utils = [c.exact_utility for c in candidates if c.exact_utility is not None]
    if exact_utils:
        oracle_best_idx = max(range(n), key=lambda i: candidates[i].exact_utility or -1e9)
        result.oracle_best_index = oracle_best_idx
    else:
        oracle_best_idx = -1

    # Compute recall@K.
    for k in k_values:
        k_actual = min(k, n)
        top_k_set = set(sorted_indices[:k_actual])
        if oracle_best_idx >= 0:
            result.recall_at_k[k] = 1.0 if oracle_best_idx in top_k_set else 0.0
        else:
            result.recall_at_k[k] = 0.0

    # Default: retain top-K (use largest K from k_values).
    default_k = min(max(k_values), n)
    result.retained_indices = sorted_indices[:default_k]
    result.n_retained = default_k

    # Check if oracle best is retained.
    if oracle_best_idx >= 0:
        result.oracle_best_retained = oracle_best_idx in set(result.retained_indices)

    # Best retained utility.
    retained_utils = [candidates[i].exact_utility for i in result.retained_indices
                      if candidates[i].exact_utility is not None]
    if retained_utils:
        result.best_retained_utility = max(retained_utils)

    # Exact evaluations saved.
    result.exact_evaluations_saved = 1.0 - (default_k / max(n, 1))

    # Mean stats.
    result.mean_predicted_utility = float(np.mean([c.predicted_utility for c in candidates]))
    result.mean_uncertainty = float(np.mean([c.predicted_uncertainty for c in candidates]))

    return result


def compute_oracle_recall(
    candidates: list[Candidate],
    k_values: list[int] | None = None,
) -> dict[int, float]:
    """Compute oracle recall at each K value.

    Recall@K = 1 if the oracle-best candidate is in the top-K by UCB score.
    """
    if k_values is None:
        k_values = [10, 25, 50, 100]

    n = len(candidates)
    if n == 0:
        return {k: 0.0 for k in k_values}

    # Sort by UCB score.
    sorted_indices = sorted(range(n), key=lambda i: candidates[i].ucb_score, reverse=True)

    # Find oracle best.
    exact_utils = [c.exact_utility for c in candidates if c.exact_utility is not None]
    if not exact_utils:
        return {k: 0.0 for k in k_values}

    oracle_best_idx = max(range(n), key=lambda i: candidates[i].exact_utility or -1e9)

    recall = {}
    for k in k_values:
        k_actual = min(k, n)
        top_k_set = set(sorted_indices[:k_actual])
        recall[k] = 1.0 if oracle_best_idx in top_k_set else 0.0

    return recall
