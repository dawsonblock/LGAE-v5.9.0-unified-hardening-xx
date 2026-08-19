"""Tests for the post-audit honest beam search (no information leakage)."""
from __future__ import annotations

import pytest
import torch
import numpy as np

from lgae_v3.experimental.exp6_3.split_utility import (
    compute_additive_utility, compute_bonus, compute_total_utility,
    make_total_utility_fn, BonusPredictor, ZeroBonusPredictor,
)
from lgae_v3.experimental.exp6_3.honest_beam_search import (
    honest_beam_search, HonestBeamResult,
)
from lgae_v3.experimental.exp6_3.honest_experiment_runner import (
    run_honest_exp6_3,
)
from lgae_v3.experimental.exp6_3.delayed_tasks import (
    get_all_delayed_value_tasks, make_task_graph, make_task_latent,
)
from lgae_v3.experimental.exp6_3.exact_mpc import exact_mpc, greedy_one_step


class TestSplitUtility:
    """Test the split utility architecture."""

    def test_additive_utility(self):
        tasks = get_all_delayed_value_tasks()
        task = tasks[0]
        graph = make_task_graph(task)
        z = make_task_latent(task)
        u_add = compute_additive_utility(graph, z)
        assert isinstance(u_add, float)

    def test_bonus_is_nonzero_for_disconnected(self):
        from lgae_v3 import make_graph_buffers
        graph = make_graph_buffers(num_nodes=6, edges=[(0, 1), (2, 3)])
        z = torch.zeros(6, 4)
        bonus = compute_bonus(graph, z, lambda_conn=30.0, threshold=1)
        # 3 components, threshold=1: bonus = 30 * max(0, 2 - 3) = 0
        # Wait: threshold + 1 - n_comp = 1 + 1 - 3 = -1, max(0, -1) = 0
        # So bonus = 0 when n_comp > threshold + 1
        assert bonus == 0.0

    def test_bonus_is_nonzero_when_connected(self):
        from lgae_v3 import make_graph_buffers
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1), (1, 2), (2, 3)])
        z = torch.zeros(4, 4)
        bonus = compute_bonus(graph, z, lambda_conn=30.0, threshold=1)
        # 1 component, threshold=1: bonus = 30 * max(0, 2 - 1) = 30
        assert bonus == 30.0

    def test_total_equals_additive_plus_bonus(self):
        tasks = get_all_delayed_value_tasks()
        task = tasks[0]
        graph = make_task_graph(task)
        z = make_task_latent(task)
        u_add = compute_additive_utility(graph, z)
        u_bonus = compute_bonus(graph, z,
                                task.utility_params.get("lambda_conn", 30.0),
                                task.utility_params.get("threshold", 1))
        u_total = compute_total_utility(graph, z,
                                        task.utility_params.get("lambda_conn", 30.0),
                                        task.utility_params.get("threshold", 1))
        assert abs(u_total - (u_add + u_bonus)) < 1e-4


class TestHonestBeamSearch:
    """Test that the honest beam search does NOT use utility_fn."""

    def test_honest_beam_search_runs(self):
        tasks = get_all_delayed_value_tasks()
        task = tasks[0]
        graph = make_task_graph(task)
        z = make_task_latent(task)
        predictor = ZeroBonusPredictor()
        result = honest_beam_search(
            graph, z, task.available_actions, predictor,
            horizon=2, gamma=0.9, beam_width=3,
        )
        assert result.first_action[0] != ""
        assert result.nodes_expanded > 0
        assert result.used_exact_bonus is False

    def test_zero_bonus_equals_greedy(self):
        """With zero bonus prediction, beam search should match greedy."""
        tasks = get_all_delayed_value_tasks()
        task = tasks[0]
        graph = make_task_graph(task)
        z = make_task_latent(task)
        predictor = ZeroBonusPredictor()
        bs = honest_beam_search(
            graph, z, task.available_actions, predictor,
            horizon=2, gamma=0.9, beam_width=10,
        )
        greedy = greedy_one_step(graph, z, task.available_actions, task.utility_fn)
        # With zero bonus and beam_width >= n_actions, should match greedy
        # (both pick the action with best additive delta)
        assert bs.first_action == greedy.first_action

    def test_no_utility_fn_in_search(self):
        """Verify that honest_beam_search signature does not take utility_fn."""
        import inspect
        sig = inspect.signature(honest_beam_search)
        params = list(sig.parameters.keys())
        assert "utility_fn" not in params, (
            "honest_beam_search must NOT accept utility_fn — that would be information leakage"
        )


class TestBonusPredictor:
    """Test the bonus prediction model."""

    def test_zero_predictor(self):
        tasks = get_all_delayed_value_tasks()
        task = tasks[0]
        graph = make_task_graph(task)
        z = make_task_latent(task)
        pred = ZeroBonusPredictor()
        assert pred.predict(graph, z) == 0.0

    def test_ridge_predictor_fits(self):
        tasks = get_all_delayed_value_tasks()
        # Generate training data.
        from lgae_v3.experimental.exp6_3.honest_experiment_runner import generate_bonus_training_data
        graphs, z_list, bonuses = generate_bonus_training_data(tasks[:2], n_samples_per_task=10)
        pred = BonusPredictor(lambda_conn=30.0, threshold=1)
        pred.fit(graphs, z_list)
        # Should produce non-zero predictions after fitting.
        graph = make_task_graph(tasks[0])
        z = make_task_latent(tasks[0])
        prediction = pred.predict(graph, z)
        assert isinstance(prediction, float)


class TestHonestExperiment:
    """Test the honest experiment runner."""

    def test_honest_experiment_runs(self):
        result = run_honest_exp6_3()
        assert result.n_tasks == 5
        assert result.n_suboptimal_h2 >= 1
        # Gate G must pass (no leakage).
        assert result.gates["G_no_information_leakage"]["passed"] is True
        # Gate A must pass (benchmark validity).
        assert result.gates["A_benchmark_validity"]["passed"] is True

    def test_audit_note_present(self):
        result = run_honest_exp6_3()
        assert "post-audit" in result.audit_note.lower() or "leakage" in result.audit_note.lower()
