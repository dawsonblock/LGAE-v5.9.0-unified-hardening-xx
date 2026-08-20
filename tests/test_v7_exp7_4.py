"""Tests for v7.0-exp4: Learned Routing Policy."""
import numpy as np
import pytest

from lgae_v3.experimental.exp7_4 import (
    TaskEmbedding, embed_task, embed_batch, cosine_similarity, nearest_neighbors,
    MarginalValueEstimator, MarginalValueSample, OPTIONAL_NODES,
    NodeNecessityRouter, RoutingDecision,
    run_lgae_node_necessity, run_exp7_4,
)
from lgae_v3.experimental.exp7_2 import (
    create_default_nodes, create_default_topology, MockModelBackend,
    ObjectiveWeights, generate_benchmark, TASK_CLASSES,
)


class TestTaskEmbedding:
    """Test task embedding."""

    def test_embed_task(self):
        emb = embed_task("Research and synthesize information.")
        assert isinstance(emb, TaskEmbedding)
        assert emb.dim > 0
        assert len(emb.embedding) == 16

    def test_embed_batch(self):
        embs = embed_batch(["task 1", "task 2", "task 3"])
        assert embs.shape[0] == 3
        assert embs.shape[1] > 0

    def test_cosine_similarity(self):
        e1 = embed_task("research task").vector
        e2 = embed_task("research task").vector
        sim = cosine_similarity(e1, e2)
        assert sim > 0.99  # identical inputs

    def test_nearest_neighbors(self):
        embs = np.array([embed_task(t).vector for t in ["research", "coding", "memory"]])
        query = embed_task("research and synthesize").vector
        neighbors = nearest_neighbors(query, embs, k=2)
        assert len(neighbors) == 2
        # "research" should be the nearest neighbor.
        assert neighbors[0][0] == 0

    def test_different_tasks_different_embeddings(self):
        e1 = embed_task("Research and synthesize information.").vector
        e2 = embed_task("Debug this code snippet for a bug.").vector
        assert not np.allclose(e1, e2)

    def test_no_label_leakage(self):
        emb = embed_task("What is 2+2?")
        assert not hasattr(emb, "task_class")


class TestMarginalValueEstimator:
    """Test per-node marginal value estimation."""

    def test_add_sample_and_predict(self):
        est = MarginalValueEstimator(k=3, min_samples=2)
        # Add samples: researcher helps research tasks.
        for i in range(5):
            est.add_sample(
                f"Research and synthesize topic {i}",
                "researcher",
                j_with=0.5,
                j_without=0.2,
            )
        # Predict for a similar task.
        delta, conf = est.predict_marginal_value("Research and synthesize new topic", "researcher")
        assert delta > 0  # should predict positive marginal value
        assert conf > 0

    def test_predict_with_no_data(self):
        est = MarginalValueEstimator(k=3, min_samples=5)
        delta, conf = est.predict_marginal_value("test task", "researcher")
        assert delta == 0.0
        assert conf == 0.0

    def test_predict_all_nodes(self):
        est = MarginalValueEstimator(k=3, min_samples=2)
        for node in OPTIONAL_NODES:
            for i in range(3):
                est.add_sample(f"task {i}", node, j_with=0.5, j_without=0.3)
        predictions = est.predict_all_nodes("new task")
        assert len(predictions) == len(OPTIONAL_NODES)
        for node in OPTIONAL_NODES:
            assert "delta_j" in predictions[node]
            assert "confidence" in predictions[node]
            assert "include" in predictions[node]

    def test_negative_marginal_value(self):
        est = MarginalValueEstimator(k=3, min_samples=2)
        # Critic hurts simple tasks.
        for i in range(5):
            est.add_sample(f"What is the capital of country {i}?", "critic", j_with=0.3, j_without=0.5)
        delta, conf = est.predict_marginal_value("What is the capital of France?", "critic")
        assert delta < 0  # should predict negative marginal value


class TestNodeNecessityRouter:
    """Test the node-necessity router."""

    def test_route_task_no_data(self):
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        router = NodeNecessityRouter(backend, weights, min_samples=3)
        topo, decision = router.route_task("t1", "Research task")
        # With no data, all nodes should be included (default).
        assert isinstance(decision, RoutingDecision)
        assert len(decision.included_nodes) > 0

    def test_route_task_with_data(self):
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        router = NodeNecessityRouter(backend, weights, k_neighbors=3, min_samples=2)

        # Calibrate with some data.
        tasks = [
            {"task_id": f"t{i}", "input": f"Research and synthesize topic {i}", "task_class": "research"}
            for i in range(5)
        ]
        router.calibrate(tasks)

        # Route a similar task.
        topo, decision = router.route_task("t99", "Research and synthesize new topic")
        # The router should make a decision with marginal values populated.
        assert len(decision.marginal_values) == len(OPTIONAL_NODES)
        # The researcher should have a non-zero prediction (data exists).
        assert decision.marginal_values["researcher"]["confidence"] > 0

    def test_calibrate_adds_samples(self):
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        router = NodeNecessityRouter(backend, weights, shadow_batch_size=3)
        tasks = [
            {"task_id": f"t{i}", "input": f"task {i}", "task_class": "general"}
            for i in range(5)
        ]
        n_samples = router.calibrate(tasks)
        # 3 tasks × 4 nodes = 12 samples.
        assert n_samples == 12

    def test_routing_summary(self):
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        router = NodeNecessityRouter(backend, weights, min_samples=2)
        router.route_task("t1", "task 1")
        router.route_task("t2", "task 2")
        summary = router.get_routing_summary()
        assert summary["n_decisions"] == 2


class TestConditions:
    """Test the conditions."""

    def test_lgae_node_necessity(self):
        tasks = generate_benchmark(n_per_class=3, seed=42)
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        result = run_lgae_node_necessity(
            tasks, backend, weights,
            calibration_interval=5,
            shadow_batch_size=3,
        )
        assert result.condition_name == "D_lgae_node_necessity"
        assert len(result.records) == len(tasks)


class TestExperimentRunner:
    """Test the full experiment runner."""

    def test_run_exp7_4_smoke(self):
        """Smoke test with minimal tasks."""
        result = run_exp7_4(
            n_tasks_per_class=3,
            backend_type="mock",
            calibration_interval=5,
            shadow_batch_size=3,
        )
        assert result is not None
        assert len(result.condition_results) == 4
        names = [r.condition_name for r in result.condition_results]
        assert "A_fixed" in names
        assert "B_dynamic" in names
        assert "C_lgae_telemetry_only" in names or "D_lgae_task_conditioned" in names
        assert "D_lgae_node_necessity" in names
        assert len(result.gates) == 12
