"""Extended risk metrics for exp6.8.2.

Adds:
  - CVaR_p (Conditional Value at Risk)
  - Risk-by-uncertainty deciles (monotonicity check)
  - Uncertainty-error correlation
"""
from __future__ import annotations

import numpy as np
from ..exp6_8_1.risk_metrics import compute_risk_metrics


def compute_cvar(regrets: np.ndarray, percentile: float = 95.0) -> float:
    """Compute Conditional Value at Risk.

    CVaR_p = E[regret | regret >= P_p]
    """
    if len(regrets) == 0:
        return 0.0
    threshold = np.percentile(regrets, percentile)
    tail = regrets[regrets >= threshold]
    if len(tail) == 0:
        return float(threshold)
    return float(np.mean(tail))


def compute_extended_risk_metrics(regrets: np.ndarray) -> dict:
    """Compute full risk profile including CVaR."""
    base = compute_risk_metrics(regrets)
    base["cvar95"] = compute_cvar(regrets, 95)
    base["cvar99"] = compute_cvar(regrets, 99)
    return base


def compute_uncertainty_error_correlation(
    uncertainties: list[float],
    errors: list[float],
) -> dict:
    """Compute correlation between predicted uncertainty and actual error.

    A well-calibrated uncertainty metric should positively correlate
    with prediction error.

    Returns:
      - correlation: Pearson correlation
      - spearman: Spearman rank correlation
      - n_samples: count
    """
    if len(uncertainties) < 3 or len(errors) < 3:
        return {"correlation": 0.0, "spearman": 0.0, "n_samples": 0}

    u = np.array(uncertainties)
    e = np.array(errors)

    # Pearson.
    if np.std(u) < 1e-10 or np.std(e) < 1e-10:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(u, e)[0, 1])

    # Spearman (rank correlation).
    from scipy.stats import spearmanr
    try:
        spearman = float(spearmanr(u, e).correlation)
    except Exception:
        spearman = 0.0

    return {
        "correlation": pearson,
        "spearman": spearman,
        "n_samples": len(uncertainties),
    }


def compute_risk_by_uncertainty_deciles(
    uncertainties: list[float],
    regrets: list[float],
) -> dict:
    """Compute risk metrics within each uncertainty decile.

    For a well-calibrated uncertainty metric, regret should increase
    monotonically from low-uncertainty to high-uncertainty deciles.

    Returns:
      - deciles: list of {decile, mean_uncertainty, mean_regret, median_regret, n}
      - is_monotonic: whether mean regret increases monotonically
    """
    if len(uncertainties) < 10:
        return {"deciles": [], "is_monotonic": True}

    u = np.array(uncertainties)
    r = np.array(regrets)

    # Sort by uncertainty.
    order = np.argsort(u)
    u_sorted = u[order]
    r_sorted = r[order]

    n = len(u_sorted)
    decile_size = max(1, n // 10)

    deciles = []
    for d in range(10):
        start = d * decile_size
        end = min((d + 1) * decile_size, n)
        if start >= n:
            break
        u_decile = u_sorted[start:end]
        r_decile = r_sorted[start:end]
        deciles.append({
            "decile": d,
            "mean_uncertainty": float(np.mean(u_decile)),
            "mean_regret": float(np.mean(r_decile)),
            "median_regret": float(np.median(r_decile)),
            "p95_regret": float(np.percentile(r_decile, 95)) if len(r_decile) > 1 else float(r_decile[0]),
            "n": len(r_decile),
        })

    # Check monotonicity: mean regret should increase with uncertainty.
    mean_regrets = [d["mean_regret"] for d in deciles]
    is_monotonic = all(
        mean_regrets[i] <= mean_regrets[i + 1] + 1e-6
        for i in range(len(mean_regrets) - 1)
    )

    return {
        "deciles": deciles,
        "is_monotonic": is_monotonic,
    }
