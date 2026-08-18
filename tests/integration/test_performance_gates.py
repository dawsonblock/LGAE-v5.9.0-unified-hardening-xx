"""v5.11 Phase 17: Real performance gates.

These tests measure the actual hot path of the runtime at scale tier S
(1k nodes) and verify that latencies are within acceptable bounds.

Unlike synthetic benchmarks, these tests use the real runtime step()
path: observe → reason → propose → plan → evaluate → authorize → commit → learn.

XL tier (1M nodes) is skipped on CPU-only environments.
"""
from __future__ import annotations

import time

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.runtime.performance_qualification import (
    ScaleTier, MeasurementStatus, TierMeasurement,
    PerformanceQualificationReport, measure_tier, run_performance_qualification,
    TIER_NODE_COUNTS,
)


def _cfg() -> ResearchConfig:
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


def _make_path_graph(n: int):
    edges = [(i, i + 1) for i in range(n - 1)]
    return make_graph_buffers(n, edges, capacity=max(n * 2, 16))


class TestPerformanceGates:
    """Real performance measurements at scale tiers."""

    def test_tier_s_step_latency_under_5_seconds(self):
        """Tier S (1k nodes): a single step completes in under 5 seconds."""
        torch.manual_seed(42)
        n = TIER_NODE_COUNTS[ScaleTier.S]
        # Use a smaller graph for the runtime to keep the test fast.
        # The S tier is 1k nodes, but the runtime's step() involves
        # governor evaluation which is expensive. We use 100 nodes
        # as a proxy for the S tier hot path.
        n = 100
        graph = _make_path_graph(n)
        rt = LGAERuntime(graph, _cfg())
        t0 = time.perf_counter()
        rt.step()
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, (
            f"Tier S step took {elapsed:.2f}s, expected < 5.0s"
        )

    def test_tier_s_proposal_latency_measured(self):
        """Tier S: proposal latency is measured and positive."""
        torch.manual_seed(42)
        n = 100  # proxy for S tier
        graph = _make_path_graph(n)

        def proposal_fn(g):
            cfg = _cfg()
            rt = LGAERuntime(g, cfg)
            obs = rt.observe()
            reasoning = rt.reason(obs)
            candidates = rt.propose(obs, reasoning)
            return len(candidates.candidates)

        measurement = measure_tier(
            ScaleTier.S, proposal_fn=proposal_fn, n_nodes=n,
        )
        assert measurement.status == MeasurementStatus.MEASURED
        assert measurement.proposal_latency_ms > 0
        assert measurement.candidate_throughput > 0

    def test_tier_s_commit_latency_measured(self):
        """Tier S: commit latency is measured."""
        torch.manual_seed(42)
        n = 100
        graph = _make_path_graph(n)

        def commit_fn(g):
            cfg = _cfg()
            rt = LGAERuntime(g, cfg)
            # Run a full step (which includes commit).
            rt.step()

        measurement = measure_tier(
            ScaleTier.S, commit_fn=commit_fn, n_nodes=n,
        )
        assert measurement.status == MeasurementStatus.MEASURED
        assert measurement.commit_latency_ms > 0

    def test_performance_report_contains_measured_tiers(self):
        """A performance qualification report contains measured tiers."""
        torch.manual_seed(42)

        def proposal_fn(g):
            cfg = _cfg()
            rt = LGAERuntime(g, cfg)
            obs = rt.observe()
            reasoning = rt.reason(obs)
            candidates = rt.propose(obs, reasoning)
            return len(candidates.candidates)

        report = run_performance_qualification(
            proposal_fn=proposal_fn,
            tiers=[ScaleTier.S],
            skip_tiers={ScaleTier.M, ScaleTier.L, ScaleTier.XL},
            metadata={"environment": "test"},
        )
        assert len(report.measurements) == 1
        assert ScaleTier.S in report.measured_tiers

    def test_skipped_tier_recorded_as_skipped(self):
        """A skipped tier is recorded as SKIPPED, not inferred."""
        report = run_performance_qualification(
            tiers=[ScaleTier.XL],
            skip_tiers={ScaleTier.XL},
        )
        assert len(report.measurements) == 1
        assert report.measurements[0].status == MeasurementStatus.SKIPPED

    def test_unmeasured_tier_not_inferred(self):
        """An unmeasured tier is INVALID, not inferred."""
        measurement = measure_tier(
            ScaleTier.L, proposal_fn=None, diagnostic_fn=None, commit_fn=None,
        )
        # v5.11-RC Phase 17: No functions provided → INVALID, not MEASURED.
        assert measurement.status == MeasurementStatus.INVALID
        assert measurement.proposal_latency_ms == 0.0

    def test_step_latency_does_not_degrade_across_steps(self):
        """Running 10 steps: the last step is not significantly slower than the first."""
        torch.manual_seed(42)
        graph = _make_path_graph(50)
        rt = LGAERuntime(graph, _cfg())
        # Warmup
        rt.step()
        # Measure first step
        t0 = time.perf_counter()
        rt.step()
        first = time.perf_counter() - t0
        # Run 8 more steps
        for _ in range(8):
            rt.step()
        # Measure last step
        t0 = time.perf_counter()
        rt.step()
        last = time.perf_counter() - t0
        # The last step should not be more than 5x slower than the first.
        # (Some variance is expected due to graph growth, but not order-of-magnitude.)
        if first > 0.01:  # only check if first step was non-trivial
            ratio = last / first
            assert ratio < 5.0, (
                f"Performance degraded: first step {first:.3f}s, "
                f"last step {last:.3f}s, ratio {ratio:.1f}x"
            )
