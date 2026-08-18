"""True OOD qualification (Phase 25).

Evaluates a policy on held-out graph families that were never seen during
training. The key requirement is that the policy generalizes, not merely
memorizes. The OOD gate requires:

  sigma_OOD > sigma_ID   (uncertainty is higher on unseen families)
  OR another demonstrated OOD metric (e.g. regret_OOD < threshold)

This module builds on the curriculum generator (Phase 24) and the scientific
qualification gate (Phase 47).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .curriculum import CurriculumEntry, CurriculumGenerator, GraphFamily, generate_graph
from .scientific_qualification import ScientificMetric, ScientificQualificationReport


@dataclass(frozen=True, slots=True)
class OODEvaluationResult:
    """Result of evaluating a policy on one graph family."""
    family: GraphFamily
    seed: int
    n_nodes: int
    metric_value: float
    is_held_out: bool

    def to_log(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "seed": int(self.seed),
            "n_nodes": int(self.n_nodes),
            "metric_value": float(self.metric_value),
            "is_held_out": bool(self.is_held_out),
        }


@dataclass(slots=True)
class OODQualificationReport:
    """Aggregate OOD qualification report."""
    id_results: list[OODEvaluationResult] = field(default_factory=list)
    ood_results: list[OODEvaluationResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id_metric_values(self) -> list[float]:
        return [r.metric_value for r in self.id_results]

    @property
    def ood_metric_values(self) -> list[float]:
        return [r.metric_value for r in self.ood_results]

    @property
    def sigma_id(self) -> float:
        import statistics
        vals = self.id_metric_values
        return statistics.stdev(vals) if len(vals) >= 2 else 0.0

    @property
    def sigma_ood(self) -> float:
        import statistics
        vals = self.ood_metric_values
        return statistics.stdev(vals) if len(vals) >= 2 else 0.0

    @property
    def mean_id(self) -> float:
        vals = self.id_metric_values
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_ood(self) -> float:
        vals = self.ood_metric_values
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def ood_gate_passed(self) -> bool:
        """sigma_OOD > sigma_ID (uncertainty is higher on unseen families)."""
        return self.sigma_ood > self.sigma_id

    def to_log(self) -> dict[str, Any]:
        return {
            "id_results": [r.to_log() for r in self.id_results],
            "ood_results": [r.to_log() for r in self.ood_results],
            "sigma_id": float(self.sigma_id),
            "sigma_ood": float(self.sigma_ood),
            "mean_id": float(self.mean_id),
            "mean_ood": float(self.mean_ood),
            "ood_gate_passed": self.ood_gate_passed,
            "metadata": self.metadata,
        }


def evaluate_ood(
    *,
    metric_fn: Callable[[Any, CurriculumEntry], float],
    train_families: list[GraphFamily] | None = None,
    held_out_families: list[GraphFamily] | None = None,
    n_nodes: int = 20,
    n_seeds: int = 3,
    base_seed: int = 42,
) -> OODQualificationReport:
    """Evaluate a policy on ID (train) and OOD (held-out) graph families.

    ``metric_fn(graph, entry) -> float`` computes the metric (e.g. regret,
    uncertainty, error) for one graph. The report aggregates ID vs OOD.
    """
    gen = CurriculumGenerator(seed=base_seed)
    split = gen.generate_split(
        n_nodes=n_nodes,
        train_families=train_families,
        held_out_families=held_out_families,
        n_seeds=n_seeds,
    )
    report = OODQualificationReport()
    for entry in split["train"]:
        graph = generate_graph(entry)
        val = float(metric_fn(graph, entry))
        report.id_results.append(OODEvaluationResult(
            family=entry.family, seed=entry.seed, n_nodes=entry.n_nodes,
            metric_value=val, is_held_out=False,
        ))
    for entry in split["held_out"]:
        graph = generate_graph(entry)
        val = float(metric_fn(graph, entry))
        report.ood_results.append(OODEvaluationResult(
            family=entry.family, seed=entry.seed, n_nodes=entry.n_nodes,
            metric_value=val, is_held_out=True,
        ))
    return report


def to_scientific_report(ood_report: OODQualificationReport) -> ScientificQualificationReport:
    """Convert an OOD report into the scientific qualification format.

    Maps sigma_OOD and sigma_ID to seed-disaggregated metrics (one per family
    seed) so the scientific gate can evaluate them.
    """
    # Group by seed for seed-disaggregated metrics.
    ood_by_seed: dict[int, list[float]] = {}
    id_by_seed: dict[int, list[float]] = {}
    for r in ood_report.ood_results:
        ood_by_seed.setdefault(r.seed, []).append(r.metric_value)
    for r in ood_report.id_results:
        id_by_seed.setdefault(r.seed, []).append(r.metric_value)
    # Average across families within each seed.
    sigma_ood = {seed: sum(vals) / len(vals) for seed, vals in ood_by_seed.items() if vals}
    sigma_id = {seed: sum(vals) / len(vals) for seed, vals in id_by_seed.items() if vals}
    return ScientificQualificationReport(
        sigma_ood=ScientificMetric("sigma_ood", sigma_ood),
        sigma_id=ScientificMetric("sigma_id", sigma_id),
    )
