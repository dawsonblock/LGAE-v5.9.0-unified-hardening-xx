"""Pairwise model comparison for exp6.8.3.

Compares three arbitration systems:
  1. Baseline only (greedy + certified spectral)
  2. exp6.8.2 ensemble-LCB
  3. exp6.8.3 conformal advantage-LCB

This module provides the comparison framework, not the models themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ArbitrationComparison:
    """Comparison of arbitration systems."""
    baseline_metrics: dict = field(default_factory=dict)
    ensemble_lcb_metrics: dict = field(default_factory=dict)
    conformal_lcb_metrics: dict = field(default_factory=dict)
    improvement: dict = field(default_factory=dict)

    def compute_improvement(self) -> None:
        """Compute improvement of conformal over baseline and ensemble."""
        for key in ["override_precision", "coverage", "mean_regret",
                     "median_regret", "p95_regret", "cvar95"]:
            b = self.baseline_metrics.get(key, 0.0)
            e = self.ensemble_lcb_metrics.get(key, 0.0)
            c = self.conformal_lcb_metrics.get(key, 0.0)
            self.improvement[f"conformal_vs_baseline_{key}"] = c - b
            self.improvement[f"conformal_vs_ensemble_{key}"] = c - e


def compare_arbitration_systems(
    baseline_results: dict,
    ensemble_results: dict,
    conformal_results: dict,
) -> ArbitrationComparison:
    """Compare the three arbitration systems."""
    comp = ArbitrationComparison(
        baseline_metrics=baseline_results,
        ensemble_lcb_metrics=ensemble_results,
        conformal_lcb_metrics=conformal_results,
    )
    comp.compute_improvement()
    return comp
