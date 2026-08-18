"""v6 evaluation metrics.

The scientific gate asks:

    LGAE_v6 > LGAE_v5.11 > Strong Baselines?

Not merely in raw performance. We measure:

- **Performance per compute**: utility gain divided by FLOPs or wall-clock.
- **Adaptation speed**: steps to reach a utility threshold.
- **OOD generalization**: performance drop from train to held-out families.
- **Mutation count**: how many structural changes were needed.
- **Topology complexity**: edge/node count of the resulting graph.
- **Failure rate**: fraction of rejected/catastrophic mutations.
- **Calibration**: correlation between predicted and realized ΔU.

Each metric is computed per-seed and reported as a distribution, not a
single number. The gate requires consistency across seeds, not merely
mean improvement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class V6Metric:
    """A single v6 metric value for one seed/run.

    Metrics are always reported per-seed to enable seed-disaggregated
    analysis. Aggregation happens at the report level.
    """
    name: str
    value: float
    seed: int
    split: str  # "train", "validation", "held_out", or "all"
    units: str = ""
    direction: str = "higher"  # "higher" or "lower" is better

    def to_log(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": float(self.value),
            "seed": int(self.seed),
            "split": self.split,
            "units": self.units,
            "direction": self.direction,
        }


@dataclass(slots=True)
class V6MetricReport:
    """Aggregated metric report across seeds and splits."""
    name: str
    metrics: list[V6Metric] = field(default_factory=list)
    direction: str = "higher"
    units: str = ""

    def add(self, metric: V6Metric) -> None:
        self.metrics.append(metric)

    @property
    def values(self) -> list[float]:
        return [m.value for m in self.metrics]

    @property
    def mean(self) -> float:
        v = self.values
        return float(np.mean(v)) if v else 0.0

    @property
    def median(self) -> float:
        v = self.values
        return float(np.median(v)) if v else 0.0

    @property
    def std(self) -> float:
        v = self.values
        return float(np.std(v)) if v else 0.0

    @property
    def min(self) -> float:
        v = self.values
        return float(np.min(v)) if v else 0.0

    @property
    def max(self) -> float:
        v = self.values
        return float(np.max(v)) if v else 0.0

    def by_split(self, split: str) -> list[float]:
        return [m.value for m in self.metrics if m.split == split]

    def mean_by_split(self, split: str) -> float:
        v = self.by_split(split)
        return float(np.mean(v)) if v else 0.0

    def to_log(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "units": self.units,
            "n_measurements": len(self.metrics),
            "mean": float(self.mean),
            "median": float(self.median),
            "std": float(self.std),
            "min": float(self.min),
            "max": float(self.max),
            "by_split": {
                s: {
                    "mean": float(np.mean(self.by_split(s))) if self.by_split(s) else 0.0,
                    "n": len(self.by_split(s)),
                }
                for s in ("train", "validation", "held_out", "all")
                if self.by_split(s)
            },
            "per_seed": [
                {"seed": m.seed, "value": m.value, "split": m.split}
                for m in self.metrics
            ],
        }


# ---------------------------------------------------------------------------
# Concrete metric computations.
# ---------------------------------------------------------------------------

def adaptation_speed_metric(
    utilities: Sequence[float],
    threshold: float,
    seed: int,
    split: str = "all",
) -> V6Metric:
    """Steps to reach a utility threshold for the first time.

    Lower is better. If the threshold is never reached, returns len(utilities).
    """
    for i, u in enumerate(utilities):
        if u >= threshold:
            return V6Metric(
                name="adaptation_speed",
                value=float(i),
                seed=seed,
                split=split,
                units="steps",
                direction="lower",
            )
    return V6Metric(
        name="adaptation_speed",
        value=float(len(utilities)),
        seed=seed,
        split=split,
        units="steps",
        direction="lower",
    )


def performance_per_compute_metric(
    utility_gain: float,
    compute_cost: float,
    seed: int,
    split: str = "all",
) -> V6Metric:
    """Utility gain per unit of compute.

    Higher is better. Compute cost can be FLOPs, wall-clock seconds, or
    candidate evaluations.
    """
    cost = max(compute_cost, 1e-10)
    return V6Metric(
        name="performance_per_compute",
        value=float(utility_gain) / cost,
        seed=seed,
        split=split,
        units="utility_per_compute_unit",
        direction="higher",
    )


def ood_generalization_metric(
    train_performance: float,
    held_out_performance: float,
    seed: int,
) -> V6Metric:
    """OOD generalization: ratio of held-out to train performance.

    Higher is better (1.0 = no degradation, <1.0 = degradation).
    """
    train = max(abs(train_performance), 1e-10)
    ratio = float(held_out_performance) / train
    return V6Metric(
        name="ood_generalization",
        value=ratio,
        seed=seed,
        split="held_out",
        units="ratio",
        direction="higher",
    )


def mutation_count_metric(
    n_mutations: int,
    seed: int,
    split: str = "all",
) -> V6Metric:
    """Number of structural mutations committed.

    Lower is better (efficiency): same utility with fewer mutations.
    """
    return V6Metric(
        name="mutation_count",
        value=float(n_mutations),
        seed=seed,
        split=split,
        units="count",
        direction="lower",
    )


def topology_complexity_metric(
    n_edges: int,
    n_nodes: int,
    seed: int,
    split: str = "all",
) -> V6Metric:
    """Topology complexity: edge-to-node ratio.

    Lower is better (simpler resulting structure for same utility).
    """
    nodes = max(n_nodes, 1)
    return V6Metric(
        name="topology_complexity",
        value=float(n_edges) / nodes,
        seed=seed,
        split=split,
        units="edges_per_node",
        direction="lower",
    )


def failure_rate_metric(
    n_rejected: int,
    n_total: int,
    seed: int,
    split: str = "all",
) -> V6Metric:
    """Fraction of proposed mutations that were rejected by governance.

    Lower is better (better proposal quality).
    """
    total = max(n_total, 1)
    return V6Metric(
        name="failure_rate",
        value=float(n_rejected) / total,
        seed=seed,
        split=split,
        units="fraction",
        direction="lower",
    )


def calibration_metric(
    predicted: Sequence[float],
    realized: Sequence[float],
    seed: int,
    split: str = "all",
) -> V6Metric:
    """Calibration: Pearson correlation between predicted and realized ΔU.

    Higher is better (1.0 = perfect calibration, 0.0 = no correlation).
    """
    if len(predicted) < 2 or len(predicted) != len(realized):
        return V6Metric(
            name="calibration",
            value=0.0,
            seed=seed,
            split=split,
            units="correlation",
            direction="higher",
        )
    pred = np.array(predicted, dtype=np.float64)
    real = np.array(realized, dtype=np.float64)
    if pred.std() < 1e-10 or real.std() < 1e-10:
        return V6Metric(
            name="calibration",
            value=0.0,
            seed=seed,
            split=split,
            units="correlation",
            direction="higher",
        )
    corr = float(np.corrcoef(pred, real)[0, 1])
    return V6Metric(
        name="calibration",
        value=corr,
        seed=seed,
        split=split,
        units="correlation",
        direction="higher",
    )


# Convenience aliases for the metric computation functions.
AdaptationSpeedMetric = adaptation_speed_metric
PerformancePerComputeMetric = performance_per_compute_metric
OODGeneralizationMetric = ood_generalization_metric
MutationCountMetric = mutation_count_metric
TopologyComplexityMetric = topology_complexity_metric
FailureRateMetric = failure_rate_metric
CalibrationMetric = calibration_metric


def aggregate_metrics(metrics: list[V6Metric]) -> dict[str, V6MetricReport]:
    """Aggregate a list of V6Metrics into reports by metric name."""
    reports: dict[str, V6MetricReport] = {}
    for m in metrics:
        if m.name not in reports:
            reports[m.name] = V6MetricReport(name=m.name, direction=m.direction, units=m.units)
        reports[m.name].add(m)
    return reports
