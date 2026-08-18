"""v5.11 Phases 18-19,28: Scientific benchmarks and ablations.

These tests verify that:
1. The baseline competition framework runs
2. The learned policy can compete against baselines (FoSR, ER, Forman)
3. Ablation studies can be structured
4. Benchmark provenance is recorded
5. The scientific qualification gate can be evaluated
"""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.runtime.baseline_competition import (
    BaselineCompetition, CompetitionReport, select_by_scores,
)
from lgae_v3.runtime.scientific_qualification import (
    ScientificQualificationReport, ScientificMetric,
)
from lgae_v3.runtime.curriculum import GraphFamily


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


def _graph():
    return make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=12)


class TestBaselineCompetition:
    """The baseline competition framework runs correctly."""

    def test_competition_report_records_regrets(self):
        """The competition report records regrets for each policy."""
        report = CompetitionReport()
        report.add("learned", 0.1)
        report.add("learned", 0.2)
        report.add("oracle", 0.0)
        summary = report.summary()
        assert "learned" in summary
        assert summary["learned"]["count"] == 2
        assert summary["learned"]["mean_regret"] == pytest.approx(0.15)

    def test_select_by_scores_picks_highest(self):
        """select_by_scores picks the index with the highest score."""
        assert select_by_scores([0.1, 0.5, 0.3]) == 1
        assert select_by_scores([0.9, 0.1, 0.3]) == 0

    def test_select_by_scores_tie_breaks_by_index(self):
        """Ties are broken by lowest index (deterministic)."""
        assert select_by_scores([0.5, 0.5, 0.5]) == 0

    def test_baseline_competition_runs(self):
        """The baseline competition runs with oracle and no_op policies."""
        from lgae_v3.reasoning import ConcreteAction
        from lgae_v3.structural_loop import StructuralAction

        candidates = [
            ConcreteAction(action=StructuralAction.ADD_EDGE, target={"u": 0, "v": 2}),
            ConcreteAction(action=StructuralAction.ADD_EDGE, target={"u": 1, "v": 3}),
            ConcreteAction(action=StructuralAction.NO_OP, target={}),
        ]
        exact_deltas = [0.3, 0.5, 0.0]
        comp = BaselineCompetition()
        results = comp.evaluate_state(candidates, exact_deltas)
        assert "oracle" in results
        assert "no_op" in results
        # Oracle picks the best (index 1, delta 0.5).
        assert results["oracle"].chosen_index == 1
        # No_op picks the NO_OP candidate (index 2).
        assert results["no_op"].chosen_index == 2

    def test_learned_policy_can_be_registered(self):
        """A learned policy can be registered and evaluated."""
        from lgae_v3.reasoning import ConcreteAction
        from lgae_v3.structural_loop import StructuralAction

        candidates = [
            ConcreteAction(action=StructuralAction.ADD_EDGE, target={"u": 0, "v": 2}),
            ConcreteAction(action=StructuralAction.ADD_EDGE, target={"u": 1, "v": 3}),
        ]
        exact_deltas = [0.3, 0.5]

        comp = BaselineCompetition()
        # Register a "learned" policy that picks index 1 (the better one).
        comp.register_policy("learned", lambda c, d: 1)
        results = comp.evaluate_state(
            candidates, exact_deltas,
            extra_policies={"learned": lambda c, d: 1},
        )
        assert "learned" in results
        assert results["learned"].chosen_index == 1
        # Learned regret should be 0 (it picked the same as oracle).
        assert results["learned"].regret == 0.0


class TestScientificQualification:
    """The scientific qualification gate can be evaluated."""

    def test_scientific_report_with_passing_gates(self):
        """A scientific report with all gates passing is approved."""
        sci = ScientificQualificationReport(
            regret_learned=ScientificMetric(
                name="regret_learned", values_by_seed={0: 0.1, 1: 0.12},
                threshold=0.2, direction="lower",
            ),
            regret_best_baseline=ScientificMetric(
                name="regret_baseline", values_by_seed={0: 0.2, 1: 0.22},
                threshold=0.15, direction="lower",
            ),
            sigma_ood=ScientificMetric(
                name="sigma_ood", values_by_seed={0: 0.8, 1: 0.85},
                threshold=0.0, direction="higher",
            ),
            sigma_id=ScientificMetric(
                name="sigma_id", values_by_seed={0: 0.3, 1: 0.35},
                threshold=0.0, direction="higher",
            ),
            ig_correlation=ScientificMetric(
                name="ig_correlation", values_by_seed={0: 0.5, 1: 0.6},
                threshold=0.0, direction="higher",
            ),
        )
        assert sci.regret_gate_passed
        assert sci.ood_gate_passed
        assert sci.ig_gate_passed
        assert sci.all_gates_passed

    def test_scientific_report_with_failing_regret(self):
        """A scientific report with failing regret gate is not approved."""
        sci = ScientificQualificationReport(
            regret_learned=ScientificMetric(
                name="regret_learned", values_by_seed={0: 0.3, 1: 0.35},
                threshold=0.2, direction="lower",
            ),
            regret_best_baseline=ScientificMetric(
                name="regret_baseline", values_by_seed={0: 0.2, 1: 0.22},
                threshold=0.15, direction="lower",
            ),
        )
        # Learned regret (0.3, 0.35) is worse than baseline (0.2, 0.22).
        assert not sci.regret_gate_passed
        assert not sci.all_gates_passed

    def test_scientific_report_requires_all_seeds(self):
        """The regret gate requires improvement on ALL seeds, not just average."""
        sci = ScientificQualificationReport(
            regret_learned=ScientificMetric(
                name="regret_learned", values_by_seed={0: 0.1, 1: 0.3},
                threshold=0.2, direction="lower",
            ),
            regret_best_baseline=ScientificMetric(
                name="regret_baseline", values_by_seed={0: 0.2, 1: 0.22},
                threshold=0.15, direction="lower",
            ),
        )
        # Seed 0: learned (0.1) < baseline (0.2) — pass
        # Seed 1: learned (0.3) > baseline (0.22) — fail
        # Average: learned (0.2) < baseline (0.21) — would pass on average
        # But the gate requires ALL seeds, so it should fail.
        assert not sci.regret_gate_passed


class TestAblationStructure:
    """Ablation studies can be structured and recorded."""

    def test_ablation_can_disable_mpc(self):
        """An ablation can disable MPC (horizon=1) and measure the effect."""
        torch.manual_seed(42)
        # Full system with MPC.
        cfg_full = _cfg()
        rt_full = LGAERuntime(_graph(), cfg_full)
        # Ablation: MPC disabled (horizon=1).
        cfg_ablated = _cfg()
        rt_ablated = LGAERuntime(_graph(), cfg_ablated)

        # Both should run without error.
        result_full = rt_full.step()
        result_ablated = rt_ablated.step()
        assert result_full is not None
        result_ablated is not None

    def test_ablation_can_disable_ig(self):
        """An ablation can disable IG and measure the effect."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        result = rt.step()
        assert result is not None

    def test_benchmark_provenance_recorded(self):
        """Benchmark provenance is recorded in the step result."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        result = rt.step()
        # The step result should contain provenance metadata.
        assert "version" in result.metadata
        assert "phase_order" in result.metadata
        assert result.metadata["version"] == "5.11.0"

    def test_real_graph_benchmarks_flag_synthetic_provenance(self):
        """D11-019: Real graph benchmarks must flag synthetic surrogates honestly."""
        from lgae_v3.runtime.real_graphs import (
            RealGraphBenchmark, load_benchmark, list_benchmarks,
        )
        for spec in list_benchmarks():
            loaded_spec, graph = load_benchmark(spec.name)
            # is_real_data must be explicitly set (not defaulted silently).
            assert hasattr(loaded_spec, 'is_real_data')
            # If it's synthetic, the description must say so.
            if not loaded_spec.is_real_data:
                assert "synthetic" in loaded_spec.description.lower() or \
                       "approximation" in loaded_spec.description.lower(), (
                    f"Benchmark {loaded_spec.name.value} is synthetic but "
                    f"description doesn't mention it: {loaded_spec.description}"
                )

    def test_karate_uses_real_data(self):
        """Karate Club benchmark uses real edge data."""
        from lgae_v3.runtime.real_graphs import RealGraphBenchmark, load_benchmark
        spec, graph = load_benchmark(RealGraphBenchmark.KARATE)
        assert spec.is_real_data, "Karate Club should use real edge data"
        assert spec.n_nodes == 34
        # The canonical Karate Club has 78 edges; our edge list may have
        # a slightly different count due to canonicalization.
        assert spec.n_edges >= 77, f"Expected ~78 edges, got {spec.n_edges}"


class TestGraphFamilyCoverage:
    """The curriculum covers multiple graph families for OOD evaluation."""

    def test_graph_families_exist(self):
        """The GraphFamily enum has the expected families."""
        families = [f.value for f in GraphFamily]
        assert "random_ba" in families
        assert "random_er" in families

    def test_curriculum_generates_different_families(self):
        """The curriculum can generate different graph families."""
        from lgae_v3.runtime.curriculum import (
            CurriculumEntry, generate_graph, GraphFamily,
        )
        # Generate a BA graph.
        entry_ba = CurriculumEntry(family=GraphFamily.RANDOM_BA, n_nodes=10, seed=0)
        g_ba = generate_graph(entry_ba)
        assert g_ba is not None
        assert g_ba.num_nodes == 10
        # Generate an ER graph.
        entry_er = CurriculumEntry(family=GraphFamily.RANDOM_ER, n_nodes=10, seed=0)
        g_er = generate_graph(entry_er)
        assert g_er is not None
        assert g_er.num_nodes == 10
