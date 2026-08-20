"""Split-conformal calibration for exp6.8.3.

On the CALIBRATION split, compute residuals:
  r_i = |A_i* - A_hat_i|

For desired confidence 1-alpha, compute the empirical quantile:
  q_{1-alpha} = ceil((n+1)(1-alpha)/n)th sorted residual

Then construct:
  LCB_A = A_hat - q_{1-alpha}

The arbitration rule is:
  override only if LCB_A > 0
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def compute_conformal_quantile(
    residuals: np.ndarray,
    alpha: float,
) -> float:
    """Compute the split-conformal quantile for confidence 1-alpha.

    q_{1-alpha} = the ceil((n+1)(1-alpha))-th smallest residual.

    This guarantees that with probability >= 1-alpha:
      |A* - A_hat| <= q_{1-alpha}
    """
    n = len(residuals)
    if n == 0:
        return float("inf")

    # Sort residuals.
    sorted_r = np.sort(residuals)

    # The (1-alpha) quantile with finite-sample coverage guarantee.
    # Index: ceil((n+1)(1-alpha)) - 1 (0-indexed)
    idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    idx = max(0, min(idx, n - 1))

    return float(sorted_r[idx])


def calibrate_conformal(
    y_cal: np.ndarray,
    y_hat_cal: np.ndarray,
    alphas: list[float] = None,
) -> dict[float, float]:
    """Compute conformal quantiles for multiple alpha levels.

    Returns {alpha: q_{1-alpha}} mapping.
    """
    if alphas is None:
        alphas = [0.20, 0.10, 0.05, 0.025, 0.01]

    residuals = np.abs(y_cal - y_hat_cal)
    quantiles = {}

    for alpha in alphas:
        q = compute_conformal_quantile(residuals, alpha)
        quantiles[alpha] = q

    return quantiles


def compute_lcb_advantage(
    y_hat: float,
    conformal_quantile: float,
) -> float:
    """Compute the lower confidence bound on advantage.

    LCB_A = A_hat - q_{1-alpha}
    """
    return float(y_hat - conformal_quantile)


def select_operating_alpha(
    calibration_results: dict[float, dict],
    min_precision: float = 0.95,
    min_coverage: float = 0.10,
    max_p95_regret: float = None,
    max_cvar95: float = None,
) -> tuple[float, dict]:
    """Select the operating alpha from calibration results only.

    Choose the highest-coverage alpha (lowest confidence requirement)
    that satisfies:
      - override_precision >= min_precision
      - coverage >= min_coverage
      - P95 regret <= max_p95_regret (if specified)
      - CVaR95 <= max_cvar95 (if specified)

    Returns (best_alpha, selection_metrics).
    """
    # Sort alphas from low to high (low alpha = high confidence = conservative).
    sorted_alphas = sorted(calibration_results.keys())

    best_alpha = None
    best_metrics = None

    # We want the highest coverage, which means the lowest confidence
    # requirement, which means the highest alpha. But we also need
    # precision >= min_precision. So we search from high alpha (aggressive)
    # to low alpha (conservative), and pick the first that passes.
    for alpha in reversed(sorted_alphas):
        metrics = calibration_results[alpha]
        precision = metrics.get("override_precision", 0.0)
        coverage = metrics.get("coverage", 0.0)
        p95_reg = metrics.get("p95_regret", float("inf"))
        cvar95 = metrics.get("cvar95", float("inf"))

        if precision < min_precision:
            continue
        if coverage < min_coverage:
            continue
        if max_p95_regret is not None and p95_reg > max_p95_regret:
            continue
        if max_cvar95 is not None and cvar95 > max_cvar95:
            continue

        best_alpha = alpha
        best_metrics = metrics
        break

    return best_alpha, best_metrics


def compute_conformalized_quantile_intervals(
    q05_pred: np.ndarray,
    q95_pred: np.ndarray,
    y_cal: np.ndarray,
    q05_cal: np.ndarray,
    q95_cal: np.ndarray,
    alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Conformalized quantile regression intervals.

    For CQR, the interval is:
      [q05_pred - q_lo, q95_pred + q_hi]

    where q_lo and q_hi are conformal corrections based on calibration.
    """
    # Conformal correction: max of (q05_cal - y_cal, y_cal - q95_cal)
    lo_resid = q05_cal - y_cal
    hi_resid = y_cal - q95_cal
    scores = np.maximum(lo_resid, hi_resid)

    n = len(scores)
    idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    idx = max(0, min(idx, n - 1))
    q_corr = float(np.sort(scores)[idx])

    lower = q05_pred - q_corr
    upper = q95_pred + q_corr

    return lower, upper
