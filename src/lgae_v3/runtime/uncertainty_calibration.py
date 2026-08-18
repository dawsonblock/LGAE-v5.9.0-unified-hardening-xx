"""Uncertainty calibration (Phase 12).

Calibration measures whether predicted uncertainties are well-calibrated:
a model that says "90% confident" should be right 90% of the time. This
module implements three standard calibration metrics:

  - ECE (Expected Calibration Error): weighted average of |accuracy - confidence|
    across bins
  - NLL (Negative Log-Likelihood): penalizes confident wrong predictions
  - Brier score: mean squared error between predicted probabilities and outcomes

A well-calibrated model has low ECE, low NLL, and low Brier score. The
scientific gate (Phase 47) uses these metrics to verify that uncertainty
estimates are trustworthy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """Calibration metrics for a set of predictions."""
    ece: float  # Expected Calibration Error
    nll: float  # Negative Log-Likelihood
    brier: float  # Brier score
    bin_count: int
    bin_accuracies: list[float] = field(default_factory=list)
    bin_confidences: list[float] = field(default_factory=list)
    bin_counts: list[int] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "ece": float(self.ece),
            "nll": float(self.nll),
            "brier": float(self.brier),
            "bin_count": int(self.bin_count),
            "bin_accuracies": [float(a) for a in self.bin_accuracies],
            "bin_confidences": [float(c) for c in self.bin_confidences],
            "bin_counts": [int(c) for c in self.bin_counts],
        }


def expected_calibration_error(
    confidences: Tensor,  # [N] predicted confidence (probability of predicted class)
    correctness: Tensor,  # [N] binary: 1 if correct, 0 if wrong
    n_bins: int = 10,
) -> tuple[float, list[float], list[float], list[int]]:
    """Compute Expected Calibration Error.

    Returns (ece, bin_accuracies, bin_confidences, bin_counts).
    """
    n = len(confidences)
    if n == 0:
        return 0.0, [], [], []
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs: list[float] = []
    bin_confs: list[float] = []
    bin_counts: list[int] = []
    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:  # include 0.0 in first bin
            mask = mask | (confidences == lo)
        count = int(mask.sum().item())
        bin_counts.append(count)
        if count == 0:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            continue
        bin_acc = float(correctness[mask].float().mean().item())
        bin_conf = float(confidences[mask].mean().item())
        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)
        ece += (count / n) * abs(bin_acc - bin_conf)
    return ece, bin_accs, bin_confs, bin_counts


def negative_log_likelihood(
    probabilities: Tensor,  # [N] predicted probability of correct class
    eps: float = 1e-7,
) -> float:
    """Compute NLL. Lower is better."""
    probs = torch.clamp(probabilities, min=eps, max=1.0 - eps)
    return float(-torch.log(probs).mean().item())


def brier_score(
    probabilities: Tensor,  # [N] predicted probability of the positive class
    outcomes: Tensor,  # [N] binary outcomes (0 or 1)
) -> float:
    """Compute Brier score. Lower is better (0 = perfect)."""
    return float(((probabilities - outcomes) ** 2).mean().item())


def compute_calibration_metrics(
    *,
    confidences: Tensor,
    correctness: Tensor,
    probabilities: Tensor | None = None,
    outcomes: Tensor | None = None,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute all calibration metrics.

    If ``probabilities`` and ``outcomes`` are provided, NLL and Brier are
    computed. Otherwise, they are estimated from confidences and correctness.
    """
    ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(
        confidences, correctness, n_bins=n_bins,
    )
    if probabilities is not None and outcomes is not None:
        nll = negative_log_likelihood(probabilities)
        brier = brier_score(probabilities, outcomes)
    else:
        # Estimate NLL and Brier from confidences and correctness.
        nll = negative_log_likelihood(
            torch.where(correctness.bool(), confidences, 1 - confidences)
        )
        brier = brier_score(confidences, correctness.float())
    return CalibrationMetrics(
        ece=ece, nll=nll, brier=brier, bin_count=n_bins,
        bin_accuracies=bin_accs, bin_confidences=bin_confs, bin_counts=bin_counts,
    )


def is_well_calibrated(metrics: CalibrationMetrics, *, ece_threshold: float = 0.1) -> bool:
    """Check if a model is well-calibrated (ECE below threshold)."""
    return metrics.ece < ece_threshold
