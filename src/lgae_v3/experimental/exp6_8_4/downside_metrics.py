"""Downside-aware metrics for exp6.8.4.

Standard precision doesn't capture asymmetric payoffs.
A planner can have mediocre sign accuracy but positive risk-adjusted
expected advantage if correct overrides have large positive value
and incorrect overrides have limited downside.

Metrics:
  - MeanOverrideAdvantage: E[A* | override]
  - DownsideProb: P(A* < -tau | override)
  - CVaR_neg_5: E[A* | A* <= P5(A*)] (conditional expectation of worst 5%)
  - RiskAdjustedScore: E[A | override] - lambda * DownsideRisk
  - Spearman correlation (ranking quality)
"""
from __future__ import annotations

import numpy as np
from typing import Optional
from scipy.stats import spearmanr


def compute_spearman_correlation(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> float:
    """Spearman rank correlation — measures ranking quality."""
    if len(predicted) < 3:
        return 0.0
    if np.std(predicted) < 1e-10 or np.std(actual) < 1e-10:
        return 0.0
    try:
        return float(spearmanr(predicted, actual).correlation)
    except Exception:
        return 0.0


def compute_downside_probability(
    true_advantages: list[float],
    used_learned: list[bool],
    tau: float = 0.0,
) -> float:
    """P(A* < -tau | override) — probability of large negative advantage."""
    overrides = [a for a, used in zip(true_advantages, used_learned) if used]
    if not overrides:
        return 0.0
    count = sum(1 for a in overrides if a < -tau)
    return count / len(overrides)


def compute_cvar_negative(
    true_advantages: list[float],
    used_learned: list[bool],
    percentile: float = 5.0,
) -> float:
    """CVaR^-_p = E[A* | A* <= P_p(A*), override]

    The conditional expectation of the worst p% of override advantages.
    Lower (more negative) is worse.
    """
    overrides = [a for a, used in zip(true_advantages, used_learned) if used]
    if not overrides:
        return 0.0
    arr = np.array(overrides)
    threshold = np.percentile(arr, percentile)
    tail = arr[arr <= threshold]
    if len(tail) == 0:
        return float(threshold)
    return float(np.mean(tail))


def compute_risk_adjusted_score(
    true_advantages: list[float],
    used_learned: list[bool],
    lambda_risk: float = 1.0,
    tau: float = 0.0,
) -> float:
    """Risk-adjusted score: E[A | override] - lambda * DownsideRisk.

    A useful planner can have mediocre sign accuracy but positive
    risk-adjusted expected advantage.
    """
    overrides = [a for a, used in zip(true_advantages, used_learned) if used]
    if not overrides:
        return 0.0

    mean_adv = float(np.mean(overrides))
    downside = compute_downside_probability(true_advantages, used_learned, tau)

    # Downside risk = P(A < -tau) * |CVaR_neg_5|
    cvar_neg = compute_cvar_negative(true_advantages, used_learned, 5.0)
    downside_risk = downside * abs(cvar_neg)

    return mean_adv - lambda_risk * downside_risk


def compute_learning_curve_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    true_advantages: list[float],
    used_learned: list[bool],
) -> dict:
    """Compute all metrics for a single (model, target, features, N) cell."""
    from ..exp6_8_3.risk_metrics import (
        compute_override_precision, compute_override_coverage,
        compute_mean_override_advantage, compute_regret_metrics,
        compute_cvar,
    )

    return {
        "spearman": compute_spearman_correlation(predicted, actual),
        "override_precision": compute_override_precision(true_advantages, used_learned),
        "coverage": compute_override_coverage(used_learned),
        "mean_override_advantage": compute_mean_override_advantage(true_advantages, used_learned),
        "downside_prob": compute_downside_probability(true_advantages, used_learned),
        "cvar_neg_5": compute_cvar_negative(true_advantages, used_learned, 5.0),
        "risk_adjusted_score": compute_risk_adjusted_score(true_advantages, used_learned),
        "n_samples": len(actual),
        "n_overrides": int(sum(used_learned)),
    }
