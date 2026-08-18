"""Calibration measurement: ECE, Brier score, reliability, interval coverage.

The runtime later needs to know when not to trust the world model.
Exp4 is where that trust signal begins.

For classification:
- Expected Calibration Error (ECE)
- Brier score
- Reliability curves

For regression:
- Prediction interval coverage
- Calibration of standardized residuals
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np


@dataclass(slots=True)
class CalibrationReport:
    """Calibration report for a set of predictions."""
    metric: str  # "ece", "brier", "interval_coverage", "std_residual_calibration"
    value: float
    n_samples: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": float(self.value),
            "n_samples": int(self.n_samples),
            "details": dict(self.details),
        }


@dataclass(slots=True)
class ReliabilityCurve:
    """Reliability curve for classification calibration."""
    bin_centers: list[float]
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]
    n_bins: int

    def to_log(self) -> dict[str, Any]:
        return {
            "bin_centers": list(self.bin_centers),
            "bin_accuracies": list(self.bin_accuracies),
            "bin_confidences": list(self.bin_confidences),
            "bin_counts": [int(c) for c in self.bin_counts],
            "n_bins": int(self.n_bins),
        }


def expected_calibration_error(
    probabilities: list[float],
    labels: list[int],
    n_bins: int = 10,
) -> CalibrationReport:
    """Compute Expected Calibration Error (ECE).

    ECE = Σ (|bin_i| / N) * |acc(bin_i) - conf(bin_i)|

    A perfectly calibrated model has ECE = 0.
    """
    n = len(probabilities)
    if n == 0:
        return CalibrationReport("ece", 0.0, 0)
    probs = np.array(probabilities)
    labs = np.array(labels)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_centers = []
    bin_accs = []
    bin_confs = []
    bin_counts = []
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:  # include 1.0 in last bin
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        count = int(mask.sum())
        if count > 0:
            acc = float(np.mean(labs[mask]))
            conf = float(np.mean(probs[mask]))
            ece += (count / n) * abs(acc - conf)
            bin_centers.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))
            bin_accs.append(acc)
            bin_confs.append(conf)
            bin_counts.append(count)
        else:
            bin_centers.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)
    return CalibrationReport("ece", float(ece), n, {
        "reliability_curve": ReliabilityCurve(
            bin_centers=bin_centers,
            bin_accuracies=bin_accs,
            bin_confidences=bin_confs,
            bin_counts=bin_counts,
            n_bins=n_bins,
        ).to_log(),
    })


def brier_score(probabilities: list[float], labels: list[int]) -> CalibrationReport:
    """Compute Brier score.

    BS = (1/N) * Σ (p_i - y_i)^2

    Lower is better. Perfect prediction has BS = 0.
    """
    n = len(probabilities)
    if n == 0:
        return CalibrationReport("brier", 0.0, 0)
    probs = np.array(probabilities)
    labs = np.array(labels, dtype=float)
    bs = float(np.mean((probs - labs) ** 2))
    return CalibrationReport("brier", bs, n)


def reliability_curve(
    probabilities: list[float],
    labels: list[int],
    n_bins: int = 10,
) -> ReliabilityCurve:
    """Compute reliability curve for calibration visualization."""
    n = len(probabilities)
    if n == 0:
        return ReliabilityCurve([], [], [], [], 0)
    probs = np.array(probabilities)
    labs = np.array(labels)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers = []
    accs = []
    confs = []
    counts = []
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        count = int(mask.sum())
        centers.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))
        if count > 0:
            accs.append(float(np.mean(labs[mask])))
            confs.append(float(np.mean(probs[mask])))
        else:
            accs.append(0.0)
            confs.append(0.0)
        counts.append(count)
    return ReliabilityCurve(centers, accs, confs, counts, n_bins)


def prediction_interval_coverage(
    means: list[float],
    uncertainties: list[float],
    targets: list[float],
    z: float = 1.96,  # 95% interval
) -> CalibrationReport:
    """Compute prediction interval coverage.

    For a well-calibrated model with 95% intervals, ~95% of targets
    should fall within [mean - z*unc, mean + z*unc].
    """
    n = len(means)
    if n == 0:
        return CalibrationReport("interval_coverage", 0.0, 0)
    means_arr = np.array(means)
    uncs = np.array(uncertainties)
    targets_arr = np.array(targets)
    lower = means_arr - z * uncs
    upper = means_arr + z * uncs
    covered = np.sum((targets_arr >= lower) & (targets_arr <= upper))
    coverage = float(covered) / n
    return CalibrationReport("interval_coverage", coverage, n, {
        "z": float(z),
        "expected_coverage": float(safe_sigmoid(z) * 2 - 1),  # approx
        "actual_coverage": coverage,
    })


def standardized_residual_calibration(
    means: list[float],
    uncertainties: list[float],
    targets: list[float],
) -> CalibrationReport:
    """Check calibration of standardized residuals.

    For a well-calibrated model, standardized residuals
    (target - mean) / uncertainty should be approximately N(0, 1).
    """
    n = len(means)
    if n == 0:
        return CalibrationReport("std_residual_calibration", 0.0, 0)
    means_arr = np.array(means)
    uncs = np.array(uncertainties)
    targets_arr = np.array(targets)
    # Avoid division by zero.
    safe_uncs = np.where(uncs > 1e-10, uncs, 1.0)
    residuals = (targets_arr - means_arr) / safe_uncs
    # Should be N(0, 1): mean ~0, std ~1.
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))
    # Calibration error: how far from N(0, 1)?
    cal_error = abs(res_mean) + abs(res_std - 1.0)
    return CalibrationReport("std_residual_calibration", float(cal_error), n, {
        "residual_mean": res_mean,
        "residual_std": res_std,
        "ideal_mean": 0.0,
        "ideal_std": 1.0,
    })


def calibration_drift(
    val_calibration: float,
    heldout_calibration: float,
) -> float:
    """Measure calibration drift between validation and held-out.

    A large degradation means the model's confidence cannot be trusted
    OOD even if raw ranking performance remains decent.
    """
    return float(heldout_calibration) - float(val_calibration)


def safe_sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)
