"""Shadow transfer analysis for exp7.3.

Measures whether shadow advantage predicts full-set advantage.
Key metric: ShadowTransferCorrelation = corr(ΔJ_shadow, ΔJ_full).

Also produces a confusion table:
  shadow positive / full positive  (true positive)
  shadow positive / full negative  (false positive — the problem)
  shadow negative / full positive  (false negative)
  shadow negative / full negative  (true negative)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ShadowTransferResult:
    """Result of shadow transfer analysis."""
    shadow_batch_size: int = 0
    n_mutations: int = 0
    shadow_advantages: list[float] = field(default_factory=list)
    full_advantages: list[float] = field(default_factory=list)
    correlation: float = 0.0
    # Confusion table
    tp: int = 0  # shadow+, full+
    fp: int = 0  # shadow+, full-  (false positive — bad mutations applied)
    fn: int = 0  # shadow-, full+  (false negative — missed good mutations)
    tn: int = 0  # shadow-, full-
    precision: float = 0.0  # tp / (tp + fp)
    recall: float = 0.0     # tp / (tp + fn)
    f1: float = 0.0

    def to_dict(self) -> dict:
        return {
            "shadow_batch_size": self.shadow_batch_size,
            "n_mutations": self.n_mutations,
            "correlation": round(self.correlation, 4),
            "confusion": {
                "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            },
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def compute_shadow_transfer(
    shadow_advantages: list[float],
    full_advantages: list[float],
    shadow_batch_size: int,
    threshold: float = 0.0,
) -> ShadowTransferResult:
    """Compute shadow transfer correlation and confusion table.

    Args:
        shadow_advantages: ΔJ measured on shadow batch for each mutation
        full_advantages: ΔJ measured on full task set for each mutation
        shadow_batch_size: size of shadow batch used
        threshold: advantage threshold for positive/negative classification

    Returns:
        ShadowTransferResult with correlation and confusion table
    """
    n = min(len(shadow_advantages), len(full_advantages))
    if n == 0:
        return ShadowTransferResult(shadow_batch_size=shadow_batch_size)

    shadow = np.array(shadow_advantages[:n])
    full = np.array(full_advantages[:n])

    # Correlation.
    if n > 1 and np.std(shadow) > 1e-10 and np.std(full) > 1e-10:
        corr = float(np.corrcoef(shadow, full)[0, 1])
    else:
        corr = 0.0

    # Confusion table.
    shadow_positive = shadow > threshold
    full_positive = full > threshold

    tp = int(np.sum(shadow_positive & full_positive))
    fp = int(np.sum(shadow_positive & ~full_positive))
    fn = int(np.sum(~shadow_positive & full_positive))
    tn = int(np.sum(~shadow_positive & ~full_positive))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return ShadowTransferResult(
        shadow_batch_size=shadow_batch_size,
        n_mutations=n,
        shadow_advantages=shadow_advantages[:n],
        full_advantages=full_advantages[:n],
        correlation=corr,
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1,
    )


def sweep_shadow_batch_sizes(
    shadow_advantages_by_size: dict[int, list[float]],
    full_advantages: list[float],
    sizes: list[int] = None,
) -> list[ShadowTransferResult]:
    """Sweep shadow batch sizes and compute transfer for each.

    Args:
        shadow_advantages_by_size: {batch_size: [advantages]}
        full_advantages: full-set advantages (same for all sizes)
        sizes: batch sizes to test

    Returns:
        List of ShadowTransferResult, one per batch size
    """
    if sizes is None:
        sizes = [5, 10, 20, 50]

    results = []
    for size in sizes:
        shadow_advs = shadow_advantages_by_size.get(size, [])
        if not shadow_advs:
            continue
        result = compute_shadow_transfer(shadow_advs, full_advantages, size)
        results.append(result)

    return results
