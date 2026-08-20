"""Target transformations for exp6.8.4.

The advantage A* = Q_L - Q_B may be badly conditioned.
Test multiple target representations:

  T1_raw: A* as-is
  T2_normalized: A* / (|Q_B| + epsilon)
  T3_sign: sign(A*) in {-1, 0, +1}
  T4_ordinal: 5-class ordinal {strongly worse, slightly worse, tied, slightly better, strongly better}
  T5_downside: A* but with downside clipped (asymmetric loss)
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def transform_raw(advantages: np.ndarray, baseline_qs: np.ndarray = None) -> np.ndarray:
    """T1: Raw advantage."""
    return advantages.astype(np.float32)


def transform_normalized(advantages: np.ndarray, baseline_qs: np.ndarray) -> np.ndarray:
    """T2: Normalized advantage A / (|Q_B| + epsilon)."""
    return (advantages / (np.abs(baseline_qs) + 1e-6)).astype(np.float32)


def transform_sign(advantages: np.ndarray, baseline_qs: np.ndarray = None) -> np.ndarray:
    """T3: Sign of advantage."""
    return np.sign(advantages).astype(np.float32)


def transform_ordinal(
    advantages: np.ndarray,
    baseline_qs: np.ndarray = None,
    thresholds: tuple = (0.0, 0.0),
) -> np.ndarray:
    """T4: Ordinal target with 5 classes.

    0: strongly worse (A < -tau_neg)
    1: slightly worse (-tau_neg <= A < 0)
    2: approximately tied (A == 0)
    3: slightly better (0 < A <= tau_pos)
    4: strongly better (A > tau_pos)

    Thresholds are set to the 25th and 75th percentiles of |A|.
    """
    if len(advantages) == 0:
        return np.array([], dtype=np.float32)

    abs_a = np.abs(advantages)
    if np.std(abs_a) < 1e-10:
        tau = 1.0
    else:
        tau = float(np.percentile(abs_a[abs_a > 0], 50)) if np.any(abs_a > 0) else 1.0

    result = np.zeros(len(advantages), dtype=np.float32)
    for i, a in enumerate(advantages):
        if a < -tau:
            result[i] = 0.0
        elif a < 0:
            result[i] = 1.0
        elif a == 0:
            result[i] = 2.0
        elif a <= tau:
            result[i] = 3.0
        else:
            result[i] = 4.0
    return result


def transform_downside(
    advantages: np.ndarray,
    baseline_qs: np.ndarray = None,
    clip_neg: float = None,
) -> np.ndarray:
    """T5: Downside-adjusted advantage.

    Clips large negative advantages to limit the influence of
    catastrophic-but-rare failures on the regression target.

    A_downside = max(A, -clip_neg) if clip_neg is specified
    Otherwise clips at the 5th percentile.
    """
    if len(advantages) == 0:
        return np.array([], dtype=np.float32)
    if clip_neg is None:
        clip_neg = float(np.percentile(advantages, 5))
    return np.maximum(advantages, clip_neg).astype(np.float32)


TARGET_TRANSFORMS = {
    "T1_raw": transform_raw,
    "T2_normalized": transform_normalized,
    "T3_sign": transform_sign,
    "T4_ordinal": transform_ordinal,
    "T5_downside": transform_downside,
}


def apply_target_transform(
    name: str,
    advantages: np.ndarray,
    baseline_qs: np.ndarray = None,
) -> np.ndarray:
    """Apply a target transform by name."""
    fn = TARGET_TRANSFORMS.get(name, transform_raw)
    if name in ("T2_normalized",):
        if baseline_qs is None:
            return transform_raw(advantages)
        return fn(advantages, baseline_qs)
    elif name in ("T4_ordinal",):
        return fn(advantages)
    elif name in ("T5_downside",):
        return fn(advantages)
    else:
        return fn(advantages)


def is_classification_target(name: str) -> bool:
    """Whether the target is a classification problem."""
    return name in ("T3_sign", "T4_ordinal")
