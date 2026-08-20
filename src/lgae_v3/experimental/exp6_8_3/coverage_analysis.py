"""Coverage-vs-safety analysis for exp6.8.3.

Sweep alpha levels to generate the coverage-versus-safety frontier.
For each alpha, measure:
  - coverage (fraction of decisions where learned is used)
  - override_precision (P(A* > 0 | override))
  - mean/median/P95/CVaR95 regret
  - mean override advantage

The operating point is chosen on CALIBRATION only, then frozen.
"""
from __future__ import annotations

import numpy as np
from typing import Optional
from .risk_metrics import (
    compute_override_precision, compute_false_override_rate,
    compute_override_coverage, compute_mean_override_advantage,
    compute_regret_metrics, compute_cvar,
)


def compute_coverage_safety_curve(
    true_advantages: list[float],
    regret_values: list[float],
    used_learned_by_alpha: dict[float, list[bool]],
) -> dict[float, dict]:
    """Compute coverage-vs-safety metrics for each alpha level.

    Returns {alpha: {coverage, precision, false_override_rate,
                     mean_advantage, mean_regret, median_regret,
                     p95_regret, cvar95}}.
    """
    curve = {}
    for alpha, used in used_learned_by_alpha.items():
        precision = compute_override_precision(true_advantages, used)
        false_rate = compute_false_override_rate(true_advantages, used)
        coverage = compute_override_coverage(used)
        mean_adv = compute_mean_override_advantage(true_advantages, used)

        # Regret for selected actions (only for tasks where we overrode).
        override_regrets = np.array([
            r for r, u in zip(regret_values, used) if u
        ]) if used else np.array([])

        if len(override_regrets) > 0:
            regret_metrics = compute_regret_metrics(override_regrets)
        else:
            regret_metrics = {
                "mean": 0.0, "median": 0.0, "p95": 0.0, "cvar95": 0.0,
            }

        curve[alpha] = {
            "alpha": alpha,
            "coverage": round(coverage, 4),
            "override_precision": round(precision, 4),
            "false_override_rate": round(false_rate, 4),
            "mean_override_advantage": round(mean_adv, 4),
            "mean_regret": round(regret_metrics["mean"], 4),
            "median_regret": round(regret_metrics["median"], 4),
            "p95_regret": round(regret_metrics["p95"], 4),
            "cvar95": round(regret_metrics["cvar95"], 4),
            "n_overrides": int(sum(used)),
            "n_total": len(used),
        }

    return curve


def select_operating_point(
    coverage_curve: dict[float, dict],
    min_precision: float = 0.95,
    min_coverage: float = 0.10,
    max_p95_regret: float = None,
    max_cvar95: float = None,
) -> tuple[Optional[float], dict]:
    """Select the operating alpha from the coverage curve.

    Choose the highest-coverage alpha that satisfies all constraints.
    Search from aggressive (high alpha) to conservative (low alpha).
    """
    sorted_alphas = sorted(coverage_curve.keys(), reverse=True)

    for alpha in sorted_alphas:
        metrics = coverage_curve[alpha]
        if metrics["override_precision"] < min_precision:
            continue
        if metrics["coverage"] < min_coverage:
            continue
        if max_p95_regret is not None and metrics["p95_regret"] > max_p95_regret:
            continue
        if max_cvar95 is not None and metrics["cvar95"] > max_cvar95:
            continue
        return alpha, metrics

    return None, {}
