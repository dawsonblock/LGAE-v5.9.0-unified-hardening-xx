"""Tests for v7.0-exp2: Live Model Topology Benchmark."""
import numpy as np
import pytest

from lgae_v3.experimental.exp7_2 import (
    ModelBackend, ModelResponse, Message, MockModelBackend, create_backend,
    AINode, NodeRole, NodeTelemetry, create_default_nodes, SYSTEM_PROMPTS,
    AITopology, TopologyEdge, AIRuntime, StructuralTransitionRecord, create_default_topology,
    TopologyAction, TopologyActionType, generate_candidate_actions,
    TopologyController,
    ObjectiveWeights, compute_objective, compute_objective_from_record,
    compute_quality_per_token, compute_quality_per_cost, compute_pareto_efficiency,
    BenchmarkTask, generate_benchmark, TASK_CLASSES,
    evaluate_quality,
    run_fixed_topology, run_dynamic_router, run_lgae_adaptive,
    run_exp7_2,
)


class TestModelBackend:
    """Test the pluggable backend."""

    def test_mock_backend(self):
        backend = MockModelBackend(seed=42)
        response = backend.generate(
            role="worker",
            system_prompt="You are a worker.",
            messages=[Message(role="user", content="test input")],
        )
        assert isinstance(response, ModelResponse)
        assert response.tokens_in > 0
        assert response.tokens_out > 0
        assert len(response.text) > 0

    def test_create_backend(self):
        backend = create_backend("mock")
        assert isinstance(backend, MockModelBackend)

    def test_topology_sensitive_output(self):
        """Mock backend should produce different output with different context."""
        backend = MockModelBackend(seed=42)
        # Without research context.
        r1 = backend.generate(
            role="worker",
            system_prompt="You are a worker.",
            messages=[Message(role="user", content="solve problem")],
        )
        # With research context.
        r2 = backend.generate(
            role="worker",
            system_prompt="You are a worker.",
            messages=[
                Message(role="user", content="solve problem"),
                Message(role="assistant", content="RESEARCH: 3 findings about problem"),
                Message(role="user", content="Based on above, do work"),
            ],
        )
        # Outputs should differ because context differs.
        assert r1.text != r2.text


class TestAINode:
    """Test 6-node topology."""

    def test_create_default_nodes(self):
        nodes = create_default_nodes()
        assert len(nodes) == 6
        assert "researcher" in nodes
        assert nodes["researcher"].role == NodeRole.RESEARCHER

    def test_node_invoke_with_context(self):
        nodes = create_default_nodes()
        backend = MockModelBackend()
        output, telemetry = nodes["worker"].invoke(
            task_input="test task",
            accumulated_context="[researcher]: RESEARCH: findings here",
            backend=backend,
        )
        assert isinstance(output, str)
        assert telemetry.tokens_in > 0


class TestAITopology:
    """Test topology with 6 nodes."""

    def test_create_default_topology(self):
        nodes = create_default_nodes()
        topo = create_default_topology(nodes)
        assert len(topo.nodes) == 6
        assert topo.has_edge("planner", "researcher")
        assert topo.has_edge("researcher", "worker")

    def test_adjacency_matrix_6x6(self):
        nodes = create_default_nodes()
        topo = create_default_topology(nodes)
        adj = topo.to_adjacency_matrix()
        assert adj.shape == (6, 6)


class TestAIRuntime:
    """Test runtime with context accumulation."""

    def test_execute_task(self):
        nodes = create_default_nodes()
        topo = create_default_topology(nodes)
        backend = MockModelBackend(seed=42)
        runtime = AIRuntime(topo, backend)
        record = runtime.execute_task("test_1", "What is 2+2?", "simple_factual")
        assert isinstance(record, StructuralTransitionRecord)
        assert record.total_tokens > 0
        assert record.total_llm_calls > 0
        assert len(record.nodes_executed) > 0

    def test_bypass_researcher_changes_output(self):
        """Bypassing researcher should change worker context."""
        nodes = create_default_nodes()
        topo = create_default_topology(nodes)
        # Make researcher the dominant path from planner.
        topo.reweight_edge("planner", "researcher", 2.0)
        topo.reweight_edge("planner", "worker", 0.1)
        backend = MockModelBackend(seed=42)

        # With researcher (dominant path).
        runtime1 = AIRuntime(topo, backend, routing_seed=42)
        r1 = runtime1.execute_task("t1", "research task", "research_synthesis")

        # Without researcher.
        topo2 = topo.clone()
        topo2.remove_edge("planner", "researcher")
        topo2.remove_edge("researcher", "worker")
        runtime2 = AIRuntime(topo2, backend, routing_seed=42)
        r2 = runtime2.execute_task("t1", "research task", "research_synthesis")

        # With researcher, the researcher node should be visited.
        assert "researcher" in r1.nodes_executed
        # Without researcher, it should not.
        assert "researcher" not in r2.nodes_executed
        # Tokens should differ because researcher adds context.
        assert r1.total_tokens != r2.total_tokens


class TestTopologyActions:
    """Test topology mutations."""

    def test_all_action_types(self):
        nodes = create_default_nodes()
        topo = create_default_topology(nodes)

        # ADD_ROUTE
        action = TopologyAction(TopologyActionType.ADD_ROUTE, "planner", "verifier", 0.5)
        assert action.apply(topo)
        assert topo.has_edge("planner", "verifier")

        # REMOVE_ROUTE
        action = TopologyAction(TopologyActionType.REMOVE_ROUTE, "planner", "worker")
        assert action.apply(topo)
        assert not topo.has_edge("planner", "worker")

        # REWEIGHT_ROUTE
        action = TopologyAction(TopologyActionType.REWEIGHT_ROUTE, "worker", "critic", 2.0)
        assert action.apply(topo)
        assert topo.get_edge("worker", "critic").weight == 2.0

        # BYPASS_NODE
        action = TopologyAction(TopologyActionType.BYPASS_NODE, node_id="researcher")
        assert action.apply(topo)
        assert not topo.has_edge("planner", "researcher")

    def test_generate_candidates(self):
        nodes = create_default_nodes()
        topo = create_default_topology(nodes)
        candidates = generate_candidate_actions(topo)
        assert len(candidates) > 0


class TestObjective:
    """Test normalized objective."""

    def test_compute_objective_normalized(self):
        weights = ObjectiveWeights()
        j = compute_objective(
            quality=0.8, tokens=1000, latency_ms=2500,
            failures=0, calls=3, weights=weights,
        )
        # J = 1.0*0.8 - 0.3*(1000/2000) - 0.2*(2500/5000) - 0.2*(3/6) - 0.5*0
        # J = 0.8 - 0.15 - 0.1 - 0.1 - 0 = 0.45
        assert abs(j - 0.45) < 0.01

    def test_quality_per_token(self):
        qpt = compute_quality_per_token(0.8, 400)
        assert qpt == 0.002


class TestQualityEvaluators:
    """Test deterministic quality evaluation."""

    def test_evaluate_with_verification_pass(self):
        q = evaluate_quality("simple_factual", "output", "expected", "pass", "")
        assert q >= 0.5

    def test_evaluate_with_verification_fail(self):
        q = evaluate_quality("simple_factual", "output", "expected", "fail", "")
        assert q < 0.5

    def test_evaluate_research_with_research_context(self):
        q = evaluate_quality("research_synthesis", "output", "", None, "RESEARCH: 3 findings")
        assert q > 0.5

    def test_evaluate_memory_with_memory_context(self):
        q = evaluate_quality("memory_dependent", "output", "", None, "MEMORY: retrieved items")
        assert q > 0.5


class TestBenchmark:
    """Test benchmark generation."""

    def test_generate_benchmark(self):
        tasks = generate_benchmark(n_per_class=10, seed=42)
        assert len(tasks) == 10 * len(TASK_CLASSES)
        classes = {t.task_class for t in tasks}
        assert classes == set(TASK_CLASSES)

    def test_task_flags(self):
        tasks = generate_benchmark(n_per_class=5, seed=42)
        research_tasks = [t for t in tasks if t.task_class == "research_synthesis"]
        assert all(t.benefits_from_research for t in research_tasks)


class TestConditions:
    """Test the three conditions."""

    def test_fixed_topology(self):
        tasks = generate_benchmark(n_per_class=3, seed=42)
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        result = run_fixed_topology(tasks, backend, weights)
        assert result.condition_name == "A_fixed"
        assert len(result.records) == len(tasks)

    def test_dynamic_router(self):
        tasks = generate_benchmark(n_per_class=3, seed=42)
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        result = run_dynamic_router(tasks, backend, weights)
        assert result.condition_name == "B_dynamic"
        assert len(result.records) == len(tasks)

    def test_lgae_adaptive(self):
        tasks = generate_benchmark(n_per_class=3, seed=42)
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        result = run_lgae_adaptive(tasks, backend, weights, adaptation_interval=5, shadow_batch_size=3)
        assert result.condition_name == "C_lgae"
        assert len(result.records) == len(tasks)


class TestExperimentRunner:
    """Test the full experiment runner."""

    def test_run_exp7_2_smoke(self):
        """Smoke test with minimal tasks."""
        result = run_exp7_2(
            n_tasks_per_class=3,
            backend_type="mock",
            adaptation_interval=5,
            shadow_batch_size=3,
        )
        assert result is not None
        assert len(result.condition_results) == 3
        assert len(result.pareto_analysis["points"]) == 3
        assert len(result.gates) == 10
