"""Tests for v7.0-exp3: Task-Conditioned Topology Learning."""
import numpy as np
import pytest

from lgae_v3.experimental.exp7_3 import (
    TaskFeatures, extract_features, features_to_topology_hints,
    TopologyControllerV2, ConformalAdvantageGate,
    compute_shadow_transfer, ShadowTransferResult,
    run_lgae_adaptive_v2, run_exp7_3,
)
from lgae_v3.experimental.exp7_2 import (
    create_default_nodes, create_default_topology, MockModelBackend,
    ObjectiveWeights, generate_benchmark, TASK_CLASSES,
)


class TestTaskFeatures:
    """Test task feature extraction (no labels)."""

    def test_extract_features_simple(self):
        f = extract_features("What is the capital of France?")
        assert f.n_tokens > 0
        assert f.has_question_mark
        assert f.complexity_score >= 0.0

    def test_extract_features_research(self):
        f = extract_features("Synthesize information from multiple sources about topic.")
        assert f.has_research_keywords
        assert f.suggests_research

    def test_extract_features_coding(self):
        f = extract_features("Debug this code snippet for a bug.")
        assert f.has_code_keywords
        assert f.has_debug_keywords
        assert f.suggests_critic

    def test_extract_features_memory(self):
        f = extract_features("Recall context from previous discussion.")
        assert f.has_memory_keywords
        assert f.suggests_memory

    def test_features_to_vector(self):
        f = extract_features("Solve this multi-step reasoning problem.")
        vec = f.to_vector()
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_features_to_topology_hints(self):
        f = extract_features("Research and synthesize information.")
        hints = features_to_topology_hints(f)
        assert "research_weight" in hints
        assert hints["research_weight"] > 1.0  # boosted by research suggestion

    def test_no_label_leakage(self):
        """Features should be derived from text, not from task class labels."""
        f = extract_features("What is 2+2?")
        # Should not have task_class field.
        assert not hasattr(f, "task_class")


class TestConformalAdvantageGate:
    """Test conformal advantage gate."""

    def test_gate_with_no_history(self):
        gate = ConformalAdvantageGate(alpha=0.2, min_history=5)
        # With no history, should use conservative threshold.
        assert gate.threshold() > 0
        # Should apply positive advantage.
        assert gate.should_apply(0.1)

    def test_gate_with_history(self):
        gate = ConformalAdvantageGate(alpha=0.2, min_history=5)
        for adv in [0.01, 0.02, -0.01, 0.03, 0.05]:
            gate.record(adv)
        threshold = gate.threshold()
        # Threshold should be the alpha quantile of history.
        assert threshold >= -0.01


class TestShadowTransfer:
    """Test shadow transfer analysis."""

    def test_compute_shadow_transfer_perfect_correlation(self):
        result = compute_shadow_transfer(
            shadow_advantages=[0.1, 0.2, 0.3, 0.4],
            full_advantages=[0.1, 0.2, 0.3, 0.4],
            shadow_batch_size=10,
        )
        assert result.correlation > 0.99
        assert result.tp == 4
        assert result.fp == 0
        assert result.fn == 0

    def test_compute_shadow_transfer_no_correlation(self):
        result = compute_shadow_transfer(
            shadow_advantages=[0.1, -0.1, 0.2, -0.2],
            full_advantages=[-0.1, 0.1, -0.2, 0.2],
            shadow_batch_size=10,
        )
        assert result.correlation < 0
        # All false positives and false negatives.
        assert result.tp == 0

    def test_compute_shadow_transfer_partial(self):
        result = compute_shadow_transfer(
            shadow_advantages=[0.1, 0.2, -0.1, -0.2],
            full_advantages=[0.1, -0.1, -0.1, -0.2],
            shadow_batch_size=10,
        )
        # 1 TP (0.1, 0.1), 1 FP (0.2, -0.1), 0 FN, 2 TN
        assert result.tp == 1
        assert result.fp == 1
        assert result.tn == 2


class TestTopologyControllerV2:
    """Test improved topology controller."""

    def test_controller_with_task_features(self):
        nodes = create_default_nodes()
        topo = create_default_topology(nodes)
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        controller = TopologyControllerV2(
            topo, backend, weights,
            shadow_batch_size=5,
            use_task_features=True,
        )
        tasks = [{"task_id": "t1", "input": "Research and synthesize info.", "task_class": "research"}] * 5
        records = controller.adapt(tasks)
        assert len(records) > 0

    def test_controller_without_task_features(self):
        nodes = create_default_nodes()
        topo = create_default_topology(nodes)
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        controller = TopologyControllerV2(
            topo, backend, weights,
            shadow_batch_size=5,
            use_task_features=False,
        )
        tasks = [{"task_id": "t1", "input": "Debug code.", "task_class": "coding"}] * 5
        records = controller.adapt(tasks)
        assert len(records) > 0

    def test_online_rollback(self):
        nodes = create_default_nodes()
        topo = create_default_topology(nodes)
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        controller = TopologyControllerV2(
            topo, backend, weights,
            online_rollback_window=5,
            online_rollback_epsilon=0.01,
        )
        # Feed good objectives first to set baseline.
        for i in range(5):
            controller.observe_objective(0.5)
        # Feed degrading objectives.
        for i in range(5):
            controller.observe_objective(0.1 - i * 0.05)
        # Should have triggered rollback.
        assert controller.n_rollbacks > 0


class TestConditions:
    """Test the four conditions."""

    def test_lgae_telemetry_only(self):
        tasks = generate_benchmark(n_per_class=3, seed=42)
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        result = run_lgae_adaptive_v2(
            tasks, backend, weights,
            adaptation_interval=5, shadow_batch_size=5,
            use_task_features=False,
        )
        assert result.condition_name == "C_lgae_telemetry_only"
        assert len(result.records) == len(tasks)

    def test_lgae_task_conditioned(self):
        tasks = generate_benchmark(n_per_class=3, seed=42)
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        result = run_lgae_adaptive_v2(
            tasks, backend, weights,
            adaptation_interval=5, shadow_batch_size=5,
            use_task_features=True,
        )
        assert result.condition_name == "D_lgae_task_conditioned"
        assert len(result.records) == len(tasks)


class TestExperimentRunner:
    """Test the full experiment runner."""

    def test_run_exp7_3_smoke(self):
        """Smoke test with minimal tasks, no shadow sweep."""
        result = run_exp7_3(
            n_tasks_per_class=3,
            backend_type="mock",
            adaptation_interval=5,
            shadow_batch_size=5,
            run_shadow_sweep=False,
        )
        assert result is not None
        assert len(result.condition_results) == 4
        names = [r.condition_name for r in result.condition_results]
        assert "A_fixed" in names
        assert "B_dynamic" in names
        assert "C_lgae_telemetry_only" in names
        assert "D_lgae_task_conditioned" in names
        assert len(result.gates) == 12
