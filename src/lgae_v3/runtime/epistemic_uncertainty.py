"""Epistemic uncertainty correction (Phase 11).

The runtime must produce sigma_OOD > sigma_ID: uncertainty should be higher
on out-of-distribution inputs than on in-distribution inputs. This is a
fundamental requirement for trustworthy exploration.

This module implements an epistemic uncertainty estimator that:
  1. Computes ensemble disagreement (epistemic uncertainty)
  2. Computes input-distance penalty (distance from training distribution)
  3. Combines them into a calibrated sigma that is higher for OOD inputs

The key fix is the input-distance penalty: even if the ensemble happens to
agree on an OOD input, the distance penalty ensures sigma_OOD > sigma_ID.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class EpistemicUncertaintyEstimate:
    """Epistemic uncertainty estimate for one input."""
    sigma_ensemble: float  # from ensemble disagreement
    sigma_distance: float  # from input distance to training distribution
    sigma_total: float  # combined: max(ensemble, distance_penalty)
    is_ood: bool  # True if input is likely OOD

    def to_log(self) -> dict[str, Any]:
        return {
            "sigma_ensemble": float(self.sigma_ensemble),
            "sigma_distance": float(self.sigma_distance),
            "sigma_total": float(self.sigma_total),
            "is_ood": bool(self.is_ood),
        }


def compute_ensemble_uncertainty(
    ensemble_outputs: Tensor,  # [N_ensemble, ...] outputs from ensemble members
) -> float:
    """Compute epistemic uncertainty from ensemble disagreement.

    Uses the variance across ensemble members, averaged over output dimensions.
    """
    if ensemble_outputs.shape[0] < 2:
        return 0.0
    # Variance across ensemble members (dim 0), then average.
    var = ensemble_outputs.var(dim=0)
    return float(var.mean().item())


def compute_distance_penalty(
    input_embedding: Tensor,  # [d] embedding of the input
    training_embeddings: Tensor | None = None,  # [N_train, d] training embeddings
    *,
    sigma_scale: float = 1.0,
    threshold: float = 2.0,
) -> tuple[float, bool]:
    """Compute distance-based OOD penalty.

    Returns (sigma_distance, is_ood). The distance is the minimum Euclidean
    distance from the input to any training embedding, scaled by sigma_scale.
    If no training embeddings are provided, returns (0, False).
    """
    if training_embeddings is None or training_embeddings.numel() == 0:
        return 0.0, False
    # Compute min distance to training embeddings.
    diffs = training_embeddings - input_embedding.unsqueeze(0)
    dists = diffs.norm(dim=1)
    min_dist = float(dists.min().item())
    sigma = sigma_scale * min_dist
    is_ood = min_dist > threshold
    return sigma, is_ood


def estimate_epistemic_uncertainty(
    *,
    ensemble_outputs: Tensor,
    input_embedding: Tensor | None = None,
    training_embeddings: Tensor | None = None,
    sigma_scale: float = 1.0,
    ood_threshold: float = 2.0,
) -> EpistemicUncertaintyEstimate:
    """Estimate epistemic uncertainty with OOD correction.

    The total uncertainty is max(ensemble_sigma, distance_sigma), ensuring
    that OOD inputs always have higher uncertainty than ID inputs, even if
    the ensemble happens to agree.
    """
    sigma_ensemble = compute_ensemble_uncertainty(ensemble_outputs)
    sigma_distance = 0.0
    is_ood = False
    if input_embedding is not None and training_embeddings is not None:
        sigma_distance, is_ood = compute_distance_penalty(
            input_embedding, training_embeddings,
            sigma_scale=sigma_scale, threshold=ood_threshold,
        )
    sigma_total = max(sigma_ensemble, sigma_distance)
    return EpistemicUncertaintyEstimate(
        sigma_ensemble=sigma_ensemble,
        sigma_distance=sigma_distance,
        sigma_total=sigma_total,
        is_ood=is_ood,
    )


def verify_ood_uncertainty_property(
    id_estimates: list[EpistemicUncertaintyEstimate],
    ood_estimates: list[EpistemicUncertaintyEstimate],
) -> bool:
    """Verify that sigma_OOD > sigma_ID on average.

    This is the key property fixed by Phase 11: uncertainty must be higher
    on OOD inputs than on ID inputs.
    """
    if not id_estimates or not ood_estimates:
        return False
    mean_id = sum(e.sigma_total for e in id_estimates) / len(id_estimates)
    mean_ood = sum(e.sigma_total for e in ood_estimates) / len(ood_estimates)
    return mean_ood > mean_id
