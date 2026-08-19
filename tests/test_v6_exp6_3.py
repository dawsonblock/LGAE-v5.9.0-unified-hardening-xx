"""Tests for v6.0-exp6.3: Long-horizon structural value."""
from __future__ import annotations

import pytest
import torch
import numpy as np

from lgae_v3.experimental.exp6_3 import (
    DelayedValueTask, get_all_delayed_value_tasks,
    make_task_graph, make_task_latent,
    ExactPlan, exact_mpc, greedy_one_step,
    FutureValueModel, V0Zero, V1TypeMean, V3Ridge, V5MLP,
    BeamSearchResult, beam_search,
    TrustBundle, DynamicsTrust, ValueTrust, RiskTrust,
    compute_trust_bundle,
    HorizonPolicy,
    generate_test_e_configs, generate_test_e_graph,
    first_action_agreement, planning_regret, search_savings,
    trajectory_recall, greedy_improvement,
    ValueRecord, generate_value_dataset,
    run_exp6_3,
)
from lgae_v3.runtime.analytical_utility import (
    AnalyticalUtilityOracle, AnalyticalUtilityContract,
)


class TestAnalyticalUtilityOracle:
    """Test the trusted kernel analytical utility oracle."""

    def test_contract_validation(self):
        contract = AnalyticalUtilityContract()
        assert contract.validate() is True
        assert "add_edge" in contract.supported_mutations

    def test_contract_violation_on_nonstatic_latent(self):
        contract = AnalyticalUtilityContract(latent_state_static=False)
        assert contract.validate() is False

    def test_delta_add_edge(self):
        from lgae_v3 import make_graph_buffers
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1)])
        z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        oracle = AnalyticalUtilityOracle()
        delta = oracle.delta_add_edge(graph, z, 0, 2, weight=1.0)
        # ||z0 - z2||^2 = 4.0, delta = -1.0 * 4.0 = -4.0
        assert abs(delta - (-4.0)) < 1e-4

    def test_delta_remove_edge(self):
        from lgae_v3 import make_graph_buffers
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1)])
        z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        oracle = AnalyticalUtilityOracle()
        delta = oracle.delta_remove_edge(graph, z, 0, 1)
        # ||z0 - z1||^2 = 1.0, delta = +1.0 * 1.0 = +1.0
        assert abs(delta - 1.0) < 1e-4

    def test_delta_remove_nonexistent_edge(self):
        from lgae_v3 import make_graph_buffers
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1)])
        z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        oracle = AnalyticalUtilityOracle()
        delta = oracle.delta_remove_edge(graph, z, 2, 3)
        assert delta == 0.0

    def test_delta_reweight(self):
        from lgae_v3 import make_graph_buffers
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1)])
        z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        oracle = AnalyticalUtilityOracle()
        delta = oracle.delta_reweight(graph, z, 0, 1, factor=2.0)
        # w_old=1.0, w_new=2.0, d^2=1.0, delta = -(2.0-1.0)*1.0 = -1.0
        assert abs(delta - (-1.0)) < 1e-4

    def test_rank_candidates(self):
        from lgae_v3 import make_graph_buffers
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1)])
        z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        oracle = AnalyticalUtilityOracle()
        candidates = [
            ("add_edge", 0, 3, {"weight": 1.0}),
            ("add_edge", 0, 2, {"weight": 1.0}),
            ("add_edge", 0, 1, {"weight": 1.0}),
        ]
        deltas = oracle.rank_candidates(graph, z, candidates)
        # 0->3: d^2=9, delta=-9
        # 0->2: d^2=4, delta=-4
        # 0->1: d^2=1, delta=-1
        assert deltas[0] < deltas[1] < deltas[2]

    def test_verify_against_exact(self):
        from lgae_v3 import make_graph_buffers
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1)])
        z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        oracle = AnalyticalUtilityOracle()
        candidates = [("add_edge", 0, 2, {"weight": 1.0})]
        exact_deltas = np.array([-4.0])
        report = oracle.verify_against_exact(graph, z, candidates, exact_deltas)
        assert report["passed"] is True
        assert report["r2"] > 0.9999


class TestDelayedValueTasks:
    """Test the non-greedy benchmark tasks."""

    def test_all_tasks_exist(self):
        tasks = get_all_delayed_value_tasks()
        assert len(tasks) == 5
        names = {t.name for t in tasks}
        assert "delayed_bridge" in names
        assert "staged_community_bridge" in names
        assert "hub_decomposition" in names
        assert "bottleneck_repair" in names
        assert "redundancy_then_shortcut" in names

    def test_tasks_have_actions(self):
        tasks = get_all_delayed_value_tasks()
        for task in tasks:
            assert len(task.available_actions) >= 4
            assert task.n_nodes > 0
            assert len(task.initial_edges) > 0

    def test_task_graph_creation(self):
        tasks = get_all_delayed_value_tasks()
        for task in tasks:
            graph = make_task_graph(task)
            assert int(graph.num_nodes) == task.n_nodes

    def test_task_latent_creation(self):
        tasks = get_all_delayed_value_tasks()
        for task in tasks:
            z = make_task_latent(task)
            assert z.shape == (task.n_nodes, task.latent_dim)

    def test_greedy_suboptimal_at_h2(self):
        """At least 20% of tasks must have greedy suboptimal at H=2."""
        tasks = get_all_delayed_value_tasks()
        n_suboptimal = 0
        for task in tasks:
            graph = make_task_graph(task)
            z = make_task_latent(task)
            utility_fn = task.utility_fn
            actions = task.available_actions
            greedy = greedy_one_step(graph, z, actions, utility_fn)
            exact_h2 = exact_mpc(graph, z, actions, utility_fn, horizon=2, gamma=0.9)
            if greedy.first_action != exact_h2.first_action:
                n_suboptimal += 1
        assert n_suboptimal / len(tasks) >= 0.2, (
            f"Only {n_suboptimal}/{len(tasks)} tasks have greedy suboptimal at H=2"
        )


class TestExactMPC:
    """Test exact multi-step planning."""

    def test_greedy_one_step(self):
        tasks = get_all_delayed_value_tasks()
        task = tasks[0]
        graph = make_task_graph(task)
        z = make_task_latent(task)
        greedy = greedy_one_step(graph, z, task.available_actions, task.utility_fn)
        assert greedy.first_action[0] != ""  # found some action
        assert greedy.nodes_expanded == len(task.available_actions)

    def test_exact_h2(self):
        tasks = get_all_delayed_value_tasks()
        task = tasks[0]
        graph = make_task_graph(task)
        z = make_task_latent(task)
        result = exact_mpc(graph, z, task.available_actions, task.utility_fn, horizon=2)
        assert result.horizon == 2
        assert result.nodes_expanded == len(task.available_actions) ** 2
        assert result.first_action[0] != ""

    def test_exact_h3(self):
        tasks = get_all_delayed_value_tasks()
        task = tasks[0]
        graph = make_task_graph(task)
        z = make_task_latent(task)
        result = exact_mpc(graph, z, task.available_actions, task.utility_fn, horizon=3)
        assert result.horizon == 3
        assert result.nodes_expanded == len(task.available_actions) ** 3

    def test_empty_actions(self):
        tasks = get_all_delayed_value_tasks()
        task = tasks[0]
        graph = make_task_graph(task)
        z = make_task_latent(task)
        result = exact_mpc(graph, z, [], task.utility_fn, horizon=2)
        assert result.total_value == float("-inf")


class TestFutureValueModels:
    """Test the future value model ladder."""

    def test_v0_zero(self):
        model = V0Zero()
        tasks = get_all_delayed_value_tasks()
        task = tasks[0]
        graph = make_task_graph(task)
        z = make_task_latent(task)
        assert model.predict(graph, z) == 0.0

    def test_v1_type_mean(self):
        model = V1TypeMean()
        X = np.random.randn(10, 9)
        y = np.random.randn(10)
        model.fit(X, y)
        assert model._mean != 0.0

    def test_v3_ridge(self):
        model = V3Ridge(alpha=1.0)
        X = np.random.randn(20, 9)
        y = X @ np.ones(9) + 0.1 * np.random.randn(20)
        model.fit(X, y)
        assert model._w is not None

    def test_v5_mlp(self):
        model = V5MLP(hidden_dim=16, n_epochs=50)
        X = np.random.randn(20, 9)
        y = X @ np.ones(9) + 0.1 * np.random.randn(20)
        model.fit(X, y)
        assert model._W1 is not None


class TestBeamSearch:
    """Test beam search planning."""

    def test_beam_search_runs(self):
        tasks = get_all_delayed_value_tasks()
        task = tasks[0]
        graph = make_task_graph(task)
        z = make_task_latent(task)
        model = V0Zero()
        result = beam_search(
            graph, z, task.available_actions, task.utility_fn, model,
            horizon=2, gamma=0.9, beam_width=3,
        )
        assert result.first_action[0] != ""
        assert result.nodes_expanded > 0

    def test_beam_search_savings(self):
        """Beam width < n_actions should give positive savings."""
        tasks = get_all_delayed_value_tasks()
        task = tasks[0]
        graph = make_task_graph(task)
        z = make_task_latent(task)
        model = V0Zero()
        exact = exact_mpc(graph, z, task.available_actions, task.utility_fn, horizon=2)
        bs = beam_search(
            graph, z, task.available_actions, task.utility_fn, model,
            horizon=2, gamma=0.9, beam_width=2,
        )
        savings = 1.0 - bs.nodes_expanded / exact.nodes_expanded
        assert savings > 0.0, f"Beam search should save nodes, got savings={savings}"


class TestTrustBundle:
    """Test separated trust channels."""

    def test_untrusted_bundle(self):
        bundle = compute_trust_bundle()
        assert bundle.dynamics.level.value == "untrusted"
        assert bundle.value.level.value == "untrusted"
        assert bundle.max_horizon == 0

    def test_trusted_bundle(self):
        bundle = compute_trust_bundle(
            dynamics_r2=0.6, dynamics_n_cal=5,
            value_spearman=0.8, value_n_cal=5,
        )
        assert bundle.dynamics.level.value == "high_confidence"
        assert bundle.value.level.value == "high_confidence"
        assert bundle.max_horizon == 3

    def test_trust_separation(self):
        """Strong dynamics trust should NOT imply value trust."""
        bundle = compute_trust_bundle(
            dynamics_r2=0.9, dynamics_n_cal=10,
            value_spearman=0.0, value_n_cal=0,
        )
        assert bundle.dynamics.can_rollout is True
        assert bundle.value.can_plan is False
        assert bundle.max_horizon == 0  # No value trust → no planning

    def test_horizon_policy(self):
        policy = HorizonPolicy()
        bundle = compute_trust_bundle(
            dynamics_r2=0.6, dynamics_n_cal=5,
            value_spearman=0.8, value_n_cal=5,
        )
        h = policy.select(bundle)
        assert h == 3

    def test_trust_log(self):
        bundle = compute_trust_bundle()
        log = bundle.to_log()
        assert "dynamics" in log
        assert "value" in log
        assert "risk" in log
        assert "max_horizon" in log


class TestMetrics:
    """Test exp6.3 metrics."""

    def test_first_action_agreement(self):
        exact = ExactPlan(first_action=("add_edge", 0, 1))
        model = BeamSearchResult(first_action=("add_edge", 0, 1))
        assert first_action_agreement(exact, model) is True

    def test_first_action_disagreement(self):
        exact = ExactPlan(first_action=("add_edge", 0, 1))
        model = BeamSearchResult(first_action=("add_edge", 2, 3))
        assert first_action_agreement(exact, model) is False

    def test_search_savings(self):
        exact = ExactPlan(nodes_expanded=36)
        model = BeamSearchResult(nodes_expanded=18)
        assert abs(search_savings(exact, model) - 0.5) < 1e-6

    def test_search_savings_zero(self):
        exact = ExactPlan(nodes_expanded=36)
        model = BeamSearchResult(nodes_expanded=36)
        assert search_savings(exact, model) == 0.0


class TestValueDataset:
    """Test value dataset generation."""

    def test_dataset_generation(self):
        tasks = get_all_delayed_value_tasks()[:2]
        records = generate_value_dataset(tasks, horizons=[1, 2])
        assert len(records) > 0
        for r in records:
            assert r.label_type == "EXACT_ENUMERATED"
            assert len(r.state_features) > 0

    def test_future_residual(self):
        tasks = get_all_delayed_value_tasks()[:1]
        records = generate_value_dataset(tasks, horizons=[1, 2])
        for r in records:
            # future_residual_h2 = q_h2 - delta_u
            assert abs(r.future_residual_h2 - (r.exact_q_h2 - r.analytical_delta_u)) < 1e-6


class TestExperimentRunner:
    """Test the full experiment."""

    def test_experiment_passes_gates(self):
        result = run_exp6_3()
        assert result.n_tasks == 5
        assert result.n_suboptimal_h2 >= 1
        assert result.all_gates_passed is True
        assert "A_benchmark_validity" in result.gates
        assert result.gates["A_benchmark_validity"]["passed"] is True
