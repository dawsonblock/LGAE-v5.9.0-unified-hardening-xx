"""Risk metrics for exp6.8.3.

Primary metrics:
  - OverridePrecision = P(A* > 0 | override)
  - FalseOverrideRate = P(A* <= 0 | override)
  - OverrideCoverage = #overrides / #decisions
  - MeanOverrideAdvantage = E[A* | override]

Regret metrics:
  - mean, median, P90, P95, P99, max, CVaR95
  - normalized regret

Calibration diagnostics:
  - uncertainty-error correlation
  - monotonic risk-by-confidence deciles
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def compute_override_precision(
    true_advantages: list[float],
    used_learned: list[bool],
) -> float:
    """OverridePrecision = P(A* > 0 | override).

    The percentage of learned overrides that were genuinely better.
    """
    overrides = [a for a, used in zip(true_advantages, used_learned) if used]
    if not overrides:
        return 0.0
    correct = sum(1 for a in overrides if a > 0)
    return correct / len(overrides)


def compute_false_override_rate(
    true_advantages: list[float],
    used_learned: list[bool],
) -> float:
    """FalseOverrideRate = P(A* <= 0 | override)."""
    overrides = [a for a, used in zip(true_advantages, used_learned) if used]
    if not overrides:
        return 0.0
    false = sum(1 for a in overrides if a <= 0)
    return false / len(overrides)


def compute_override_coverage(used_learned: list[bool]) -> float:
    """Coverage = #overrides / #decisions."""
    if not used_learned:
        return 0.0
    return sum(used_learned) / len(used_learned)


def compute_mean_override_advantage(
    true_advantages: list[float],
    used_learned: list[bool],
) -> float:
    """MeanOverrideAdvantage = E[A* | override]."""
    overrides = [a for a, used in zip(true_advantages, used_learned) if used]
    if not overrides:
        return 0.0
    return float(np.mean(overrides))


def compute_cvar(regrets: np.ndarray, percentile: float = 95.0) -> float:
    """CVaR_p = E[regret | regret >= P_p]."""
    if len(regrets) == 0:
        return 0.0
    threshold = np.percentile(regrets, percentile)
    tail = regrets[regrets >= threshold]
    if len(tail) == 0:
        return float(threshold)
    return float(np.mean(tail))


def compute_regret_metrics(regrets: np.ndarray) -> dict:
    """Full regret profile."""
    if len(regrets) == 0:
        return {
            "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0,
            "p99": 0.0, "max": 0.0, "cvar95": 0.0,
        }
    return {
        "mean": float(np.mean(regrets)),
        "median": float(np.median(regrets)),
        "p90": float(np.percentile(regrets, 90)),
        "p95": float(np.percentile(regrets, 95)),
        "p99": float(np.percentile(regrets, 99)),
        "max": float(np.max(regrets)),
        "cvar95": compute_cvar(regrets, 95),
    }


def compute_normalized_regret(
    exact_vals: list[float],
    selected_vals: list[float],
) -> np.ndarray:
    """NormalizedRegret = (Q* - Q(a)) / (|Q*| + epsilon)."""
    result = []
    for q_star, q_sel in zip(exact_vals, selected_vals):
        regret = abs(q_star - q_sel)
        norm = regret / (abs(q_star) + 1e-6)
        result.append(norm)
    return np.array(result)


def compute_uncertainty_error_correlation(
    uncertainties: list[float],
    errors: list[float],
) -> dict:
    """Correlation between predicted uncertainty and actual error."""
    if len(uncertainties) < 3 or len(errors) < 3:
        return {"correlation": 0.0, "spearman": 0.0, "n_samples": 0}

    u = np.array(uncertainties)
    e = np.array(errors)

    if np.std(u) < 1e-10 or np.std(e) < 1e-10:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(u, e)[0, 1])

    try:
        from scipy.stats import spearmanr
        spearman = float(spearmanr(u, e).correlation)
    except Exception:
        spearman = 0.0

    return {
        "correlation": pearson,
        "spearman": spearman,
        "n_samples": len(uncertainties),
    }


def compute_confidence_decile_analysis(
    lcb_values: list[float],
    true_advantages: list[float],
    used_learned: list[bool],
) -> dict:
    """Bucket decisions by predicted LCB confidence.

    Higher LCB should correspond to lower false-override rate and
    lower regret. This checks whether the conformal calibration
    is meaningful.
    """
    if len(lcb_values) < 10:
        return {"deciles": [], "is_monotonic": True}

    lcb = np.array(lcb_values)
    adv = np.array(true_advantages)
    used = np.array(used_learned)

    # Sort by LCB (higher = more confident).
    order = np.argsort(lcb)
    lcb_sorted = lcb[order]
    adv_sorted = adv[order]
    used_sorted = used[order]

    n = len(lcb_sorted)
    decile_size = max(1, n // 10)

    deciles = []
    for d in range(10):
        start = d * decile_size
        end = min((d + 1) * decile_size, n)
        if start >= n:
            break
        lcb_d = lcb_sorted[start:end]
        adv_d = adv_sorted[start:end]
        used_d = used_sorted[start:end]

        n_overrides = int(np.sum(used_d))
        if n_overrides > 0:
            override_precision = float(np.mean(adv_d[used_d] > 0))
        else:
            override_precision = 1.0  # no overrides = no false positives

        deciles.append({
            "decile": d,
            "mean_lcb": float(np.mean(lcb_d)),
            "n_overrides": n_overrides,
            "override_precision": override_precision,
            "mean_advantage": float(np.mean(adv_d)),
            "n": len(adv_d),
        })

    # Check monotonicity: precision should increase with LCB.
    precisions = [d["override_precision"] for d in deciles if d["n_overrides"] > 0]
    is_monotonic = all(
        precisions[i] <= precisions[i + 1] + 1e-6
        for i in range(len(precisions) - 1)
    ) if len(precisions) > 1 else True

    return {"deciles": deciles, "is_monotonic": is_monotonic}


def compute_bootstrap_ci(
    values: list[float],
    n_boot: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.

    Returns (mean, lower, upper).
    """
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values)
    rng = np.random.RandomState(42)
    boot_means = []
    n = len(arr)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        boot_means.append(float(np.mean(arr[idx])))
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, alpha * 100))
    hi = float(np.percentile(boot_means, (1 - alpha) * 100))
    return float(np.mean(arr)), lo, hi
