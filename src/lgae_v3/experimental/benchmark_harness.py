"""v6 benchmark harness.

Orchestrates the evaluation of baselines and learned policies across the
frozen graph family splits. Produces comparable results with per-seed
disaggregation.

Usage::

    registry = get_frozen_registry()
    harness = V6BenchmarkHarness(registry)
    results = harness.run_all_baselines(n_steps=5, seed=42)
    report = harness.summarize(results)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np

from ..runtime.curriculum import CurriculumEntry, generate_graph as gen_graph
from ..config import ResearchConfig
from .graph_families import FrozenGraphFamilyRegistry
from .baselines import BaselineResult, ALL_V6_BASELINES
from .metrics import (
    V6Metric,
    V6MetricReport,
    adaptation_speed_metric,
    performance_per_compute_metric,
    ood_generalization_metric,
    mutation_count_metric,
    topology_complexity_metric,
    failure_rate_metric,
    aggregate_metrics,
)


@dataclass(slots=True)
class BenchmarkRunResult:
    """Result of running one baseline on one graph instance."""
    baseline_name: str
    family: str
    split: str  # "train", "validation", "held_out"
    seed: int
    n_nodes: int
    result: BaselineResult
    metrics: list[V6Metric] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "family": self.family,
            "split": self.split,
            "seed": int(self.seed),
            "n_nodes": int(self.n_nodes),
            "result": self.result.to_log(),
            "metrics": [m.to_log() for m in self.metrics],
        }


class V6BenchmarkHarness:
    """Orchestrates benchmark evaluation across frozen graph families.

    This is the primary entry point for v6 experimental evaluation. It:

    1. Generates graphs from the frozen train/validation/held-out splits.
    2. Runs each baseline (or learned policy) on each graph.
    3. Computes v6 metrics for each run.
    4. Returns results in a format suitable for the experiment registry.
    """

    def __init__(
        self,
        registry: FrozenGraphFamilyRegistry | None = None,
        config: ResearchConfig | None = None,
    ) -> None:
        self.registry = registry or FrozenGraphFamilyRegistry()
        self.config = config or self._default_config()

    @staticmethod
    def _default_config() -> ResearchConfig:
        cfg = ResearchConfig()
        cfg.fiber.d_base = 2
        cfg.fiber.d_max = 6
        cfg.fiber.spawn_width = 1
        cfg.fiber.gauge_dim = 0
        cfg.audit.orc_backend = "exact_lp"
        cfg.audit.persistent_homology_enabled = False
        cfg.audit.entropic_nodes = 0
        cfg.audit.bakry_nodes = 0
        cfg.audit.cde_nodes = 0
        cfg.audit.exact_lly_top_k = 0
        cfg.audit.orc_top_k = 0
        cfg.mutation.shadow_horizons = [1, 2]
        cfg.mutation.curvature_ema_enabled = False
        return cfg

    def run_baseline(
        self,
        baseline_name: str,
        entry: CurriculumEntry,
        split: str,
        n_steps: int,
        seed: int,
    ) -> BenchmarkRunResult:
        """Run one baseline on one graph instance."""
        baseline = ALL_V6_BASELINES[baseline_name]
        graph = gen_graph(entry)
        result = baseline.run(graph, self.config, seed=seed, n_steps=n_steps)
        metrics = self._compute_metrics(result, split, seed)
        return BenchmarkRunResult(
            baseline_name=baseline_name,
            family=entry.family.value,
            split=split,
            seed=seed,
            n_nodes=entry.n_nodes,
            result=result,
            metrics=metrics,
        )

    def run_all_baselines(
        self,
        n_steps: int = 5,
        seed: int = 42,
        baselines: list[str] | None = None,
        splits: list[str] | None = None,
    ) -> list[BenchmarkRunResult]:
        """Run all (or a subset of) baselines on all graph instances.

        Args:
            n_steps: Number of adaptation steps per graph.
            seed: Base random seed (per-entry seeds are derived deterministically).
            baselines: Subset of baseline names to run. None = all.
            splits: Subset of split names ("train", "validation", "held_out").
                None = all.
        """
        baseline_names = baselines or list(ALL_V6_BASELINES.keys())
        split_names = splits or ["train", "validation", "held_out"]
        results: list[BenchmarkRunResult] = []
        all_entries = self.registry.all_entries()
        for split in split_names:
            entries = all_entries.get(split, [])
            for entry in entries:
                # Deterministic per-entry seed.
                import hashlib
                entry_seed = seed + int.from_bytes(
                    hashlib.sha256(entry.family_id.encode()).digest()[:4], "big"
                ) % 1000
                for bname in baseline_names:
                    results.append(
                        self.run_baseline(bname, entry, split, n_steps, entry_seed)
                    )
        return results

    @staticmethod
    def _compute_metrics(
        result: BaselineResult,
        split: str,
        seed: int,
    ) -> list[V6Metric]:
        """Compute v6 metrics from a baseline result."""
        metrics: list[V6Metric] = []
        # Final utility.
        metrics.append(V6Metric(
            name="final_utility",
            value=result.final_utility,
            seed=seed,
            split=split,
            units="spectral_gap",
            direction="higher",
        ))
        # Adaptation speed (steps to reach 90% of final utility).
        if result.utility_history:
            threshold = 0.9 * result.final_utility
            metrics.append(adaptation_speed_metric(
                result.utility_history, threshold, seed, split,
            ))
        # Performance per compute.
        if result.utility_history and result.compute_cost > 0:
            gain = result.final_utility - result.utility_history[0]
            metrics.append(performance_per_compute_metric(
                gain, result.compute_cost, seed, split,
            ))
        # Mutation count.
        metrics.append(mutation_count_metric(
            result.n_mutations, seed, split,
        ))
        # Topology complexity.
        metrics.append(topology_complexity_metric(
            result.final_n_edges, result.final_n_nodes, seed, split,
        ))
        # Failure rate.
        total = result.n_mutations + result.n_rejected
        if total > 0:
            metrics.append(failure_rate_metric(
                result.n_rejected, total, seed, split,
            ))
        return metrics

    def summarize(self, results: list[BenchmarkRunResult]) -> dict[str, Any]:
        """Summarize benchmark results by baseline and split."""
        all_metrics: list[V6Metric] = []
        for r in results:
            all_metrics.extend(r.metrics)
        reports = aggregate_metrics(all_metrics)
        # Per-baseline summary.
        by_baseline: dict[str, dict[str, Any]] = {}
        for r in results:
            if r.baseline_name not in by_baseline:
                by_baseline[r.baseline_name] = {
                    "n_runs": 0,
                    "mean_final_utility": [],
                    "splits": set(),
                }
            by_baseline[r.baseline_name]["n_runs"] += 1
            by_baseline[r.baseline_name]["mean_final_utility"].append(
                r.result.final_utility
            )
            by_baseline[r.baseline_name]["splits"].add(r.split)
        for bname, info in by_baseline.items():
            info["mean_final_utility"] = float(np.mean(info["mean_final_utility"]))
            info["splits"] = sorted(info["splits"])
        return {
            "n_results": len(results),
            "by_baseline": by_baseline,
            "metric_reports": {name: rep.to_log() for name, rep in reports.items()},
        }
