"""Risk-aware planning metrics for exp6.8.1.

Primary metrics:
  - NormalizedPlanningRegret (primary)
  - MedianRegret
  - P95Regret, P99Regret
  - P(Regret > tau) — probability of large regret

Secondary metrics:
  - NonGreedyRecoveryRate (with ActionIdentity)
  - SearchSavings
  - Coverage (fraction where learned planner was used)
"""
from __future__ import annotations

import numpy as np
from typing import List


def compute_regret_distribution(
    exact_vals: list[float],
    model_vals: list[float],
) -> np.ndarray:
    """Compute per-task regret distribution.

    regret_i = |Q*_i - Q_model_i|
    """
    regrets = []
    for ev, mv in zip(exact_vals, model_vals):
        regrets.append(abs(ev - mv))
    return np.array(regrets)


def compute_normalized_regret_distribution(
    exact_vals: list[float],
    model_vals: list[float],
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute normalized regret distribution.

    norm_regret_i = |Q*_i - Q_model_i| / (|Q*_i| + eps)
    """
    regrets = []
    for ev, mv in zip(exact_vals, model_vals):
        regrets.append(abs(ev - mv) / (abs(ev) + eps))
    return np.array(regrets)


def compute_risk_metrics(regrets: np.ndarray) -> dict:
    """Compute risk-aware metrics from a regret distribution.

    Returns:
      - mean_regret
      - median_regret
      - p95_regret
      - p99_regret
      - p_regret_gt_1: P(regret > 1.0)
      - p_regret_gt_5: P(regret > 5.0)
      - p_regret_gt_10: P(regret > 10.0)
      - max_regret
    """
    if len(regrets) == 0:
        return {
            "mean_regret": 0.0, "median_regret": 0.0,
            "p95_regret": 0.0, "p99_regret": 0.0,
            "p_regret_gt_1": 0.0, "p_regret_gt_5": 0.0,
            "p_regret_gt_10": 0.0, "max_regret": 0.0,
        }

    return {
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "p95_regret": float(np.percentile(regrets, 95)),
        "p99_regret": float(np.percentile(regrets, 99)),
        "p_regret_gt_1": float(np.mean(regrets > 1.0)),
        "p_regret_gt_5": float(np.mean(regrets > 5.0)),
        "p_regret_gt_10": float(np.mean(regrets > 10.0)),
        "max_regret": float(np.max(regrets)),
    }


def compute_coverage_risk_curve(
    coverage_results: list[list[dict]],
) -> dict:
    """Aggregate coverage-vs-risk curves across tasks.

    Input: list of per-task coverage sweeps (each from run_coverage_sweep).
    Output: for each tau_sigma, aggregated metrics.
    """
    if not coverage_results:
        return {}

    n_thresholds = len(coverage_results[0])
    curve = {}

    for i in range(n_thresholds):
        tau_sigma = coverage_results[0][i]["tau_sigma"]
        recoveries = []
        regrets = []
        norm_regrets = []
        savings = []
        used_learned = []

        for task_results in coverage_results:
            if i < len(task_results):
                r = task_results[i]
                recoveries.append(r["recovery"])
                regrets.append(r["regret"])
                norm_regrets.append(r["normalized_regret"])
                savings.append(r["savings"])
                used_learned.append(1.0 if r["used_learned"] else 0.0)

        regrets_arr = np.array(regrets)
        curve[tau_sigma] = {
            "tau_sigma": tau_sigma,
            "coverage": float(np.mean(used_learned)) if used_learned else 0.0,
            "recovery_rate": float(np.mean(recoveries)) if recoveries else 0.0,
            "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
            "median_regret": float(np.median(regrets)) if regrets else 0.0,
            "p95_regret": float(np.percentile(regrets, 95)) if len(regrets) > 0 else 0.0,
            "p99_regret": float(np.percentile(regrets, 99)) if len(regrets) > 0 else 0.0,
            "mean_norm_regret": float(np.mean(norm_regrets)) if norm_regrets else 0.0,
            "mean_savings": float(np.mean(savings)) if savings else 0.0,
            "n_tasks": len(recoveries),
        }

    return curve
