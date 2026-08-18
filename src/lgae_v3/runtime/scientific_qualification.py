"""Scientific qualification gate (Phase 47).

The scientific gate is independent of engineering/safety gates. It must
establish, across held-out graph families, with statistically meaningful
improvement over multiple seeds:

  Regret_learned < Regret_best_baseline   (not merely on average)
  sigma_OOD > sigma_ID                     (or another demonstrated OOD metric)
  rho(predicted_IG, realized_IG) > 0       (information gain calibration)

Each metric is evaluated with a seed-disaggregated distribution, not a single
number. The gate does not pass on average alone; it requires improvement that
is consistent across seeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .baseline_competition import CompetitionReport


@dataclass(frozen=True, slots=True)
class ScientificMetric:
    """One scientific metric with seed-disaggregated values."""
    name: str
    values_by_seed: dict[int, float]
    threshold: float | None = None
    direction: str = "lower"  # "lower" or "higher" is better

    @property
    def mean(self) -> float:
        v = list(self.values_by_seed.values())
        return sum(v) / len(v) if v else 0.0

    @property
    def n_seeds(self) -> int:
        return len(self.values_by_seed)

    @property
    def all_seeds_pass(self) -> bool:
        """True only if every seed meets the threshold (not merely on average)."""
        if self.threshold is None:
            return True
        if self.direction == "lower":
            return all(v < self.threshold for v in self.values_by_seed.values())
        return all(v > self.threshold for v in self.values_by_seed.values())

    @property
    def mean_passes(self) -> bool:
        if self.threshold is None:
            return True
        if self.direction == "lower":
            return self.mean < self.threshold
        return self.mean > self.threshold

    def to_log(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values_by_seed": {int(k): float(v) for k, v in sorted(self.values_by_seed.items())},
            "mean": float(self.mean),
            "n_seeds": int(self.n_seeds),
            "threshold": self.threshold,
            "direction": self.direction,
            "all_seeds_pass": self.all_seeds_pass,
            "mean_passes": self.mean_passes,
        }


@dataclass(slots=True)
class ScientificQualificationReport:
    """Aggregate scientific qualification report."""
    regret_learned: ScientificMetric | None = None
    regret_best_baseline: ScientificMetric | None = None
    sigma_ood: ScientificMetric | None = None
    sigma_id: ScientificMetric | None = None
    ig_correlation: ScientificMetric | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def regret_gate_passed(self) -> bool:
        """Regret_learned < Regret_best_baseline on every seed (not merely average)."""
        if self.regret_learned is None or self.regret_best_baseline is None:
            return False
        if self.regret_learned.n_seeds == 0:
            return False
        seeds = set(self.regret_learned.values_by_seed.keys())
        baseline_seeds = set(self.regret_best_baseline.values_by_seed.keys())
        if seeds != baseline_seeds:
            return False
        return all(
            self.regret_learned.values_by_seed[s] < self.regret_best_baseline.values_by_seed[s]
            for s in seeds
        )

    @property
    def ood_gate_passed(self) -> bool:
        """sigma_OOD > sigma_ID on every seed."""
        if self.sigma_ood is None or self.sigma_id is None:
            return False
        seeds = set(self.sigma_ood.values_by_seed.keys())
        id_seeds = set(self.sigma_id.values_by_seed.keys())
        if seeds != id_seeds or not seeds:
            return False
        return all(
            self.sigma_ood.values_by_seed[s] > self.sigma_id.values_by_seed[s]
            for s in seeds
        )

    @property
    def ig_gate_passed(self) -> bool:
        """rho(predicted_IG, realized_IG) > 0 on every seed."""
        if self.ig_correlation is None:
            return False
        return self.ig_correlation.all_seeds_pass and self.ig_correlation.threshold == 0.0

    @property
    def all_gates_passed(self) -> bool:
        return self.regret_gate_passed and self.ood_gate_passed and self.ig_gate_passed

    def to_log(self) -> dict[str, Any]:
        return {
            "all_gates_passed": self.all_gates_passed,
            "regret_gate": {
                "passed": self.regret_gate_passed,
                "learned": self.regret_learned.to_log() if self.regret_learned else None,
                "best_baseline": self.regret_best_baseline.to_log() if self.regret_best_baseline else None,
            },
            "ood_gate": {
                "passed": self.ood_gate_passed,
                "sigma_ood": self.sigma_ood.to_log() if self.sigma_ood else None,
                "sigma_id": self.sigma_id.to_log() if self.sigma_id else None,
            },
            "ig_gate": {
                "passed": self.ig_gate_passed,
                "ig_correlation": self.ig_correlation.to_log() if self.ig_correlation else None,
            },
            "metadata": self.metadata,
        }


def assert_scientific_gate(report: ScientificQualificationReport) -> None:
    """Raise if the scientific gate did not pass."""
    if not report.all_gates_passed:
        failed = []
        if not report.regret_gate_passed:
            failed.append("regret_learned < regret_best_baseline (all seeds)")
        if not report.ood_gate_passed:
            failed.append("sigma_OOD > sigma_ID (all seeds)")
        if not report.ig_gate_passed:
            failed.append("rho(predicted_IG, realized_IG) > 0 (all seeds)")
        raise ScientificGateError(f"scientific gate failed: {failed}")


class ScientificGateError(RuntimeError):
    """Raised when the scientific qualification gate does not pass."""
