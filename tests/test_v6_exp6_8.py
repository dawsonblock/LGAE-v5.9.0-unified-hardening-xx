"""Tests for v6.0-exp6.8: Exact-transition model-based structural planning."""
import numpy as np
import torch
import pytest

from lgae_v3 import make_graph_buffers
from lgae_v3.experimental.exp6_8 import (
    StructuralState, compute_structural_observables,
    STRUCTURAL_OBSERVABLE_DIM, get_observable_value,
    ConsequentialStateModel, exact_transition,
    roll_forward_exact, roll_forward_predicted,
    recursive_causal_mpc, evaluate_objective_on_state,
    RecursivePlanResult,
)


class TestStructuralState:
    """Test structural state representation."""

    def test_compute_observables(self):
        graph = make_graph_buffers(
            num_nodes=6, edges=[(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=20,
        )
        z = compute_structural_observables(graph)
        assert z.shape == (STRUCTURAL_OBSERVABLE_DIM,)
        assert z[0] == 6.0 / 30.0  # n_nodes normalized

    def test_from_graph(self):
        graph = make_graph_buffers(
            num_nodes=5, edges=[(0,1),(1,2),(2,3),(3,4)], capacity=20,
        )
        state = StructuralState.from_graph(graph)
        assert state.is_predicted is False
        assert state.z.shape == (STRUCTURAL_OBSERVABLE_DIM,)

    def test_from_predicted(self):
        graph = make_graph_buffers(num_nodes=5, edges=[(0,1)], capacity=20)
        z_pred = np.zeros(STRUCTURAL_OBSERVABLE_DIM, dtype=np.float32)
        state = StructuralState.from_predicted(graph, z_pred)
        assert state.is_predicted is True
        assert np.array_equal(state.z, z_pred)

    def test_get_observable_value(self):
        z = np.array([0.2, 0.5, 0.3, 0.4, 0.5, 0.2, 0.3, 0.5, 0.7, 0.1])
        # n_components = z[1] * 6 = 3.0
        assert get_observable_value(z, "n_components") == 3.0
        # spectral_gap = z[6] = 0.3
        assert get_observable_value(z, "spectral_gap") == 0.3


class TestExactTransition:
    """Test exact graph transitions."""

    def test_add_edge_valid(self):
        graph = make_graph_buffers(
            num_nodes=6, edges=[(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=20,
        )
        new_graph, status = exact_transition(graph, ("add_edge", 0, 3, {"weight": 1.0}))
        assert status == "VALID"

    def test_remove_edge_valid(self):
        graph = make_graph_buffers(
            num_nodes=6, edges=[(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=20,
        )
        new_graph, status = exact_transition(graph, ("remove_edge", 0, 1, {}))
        assert status == "VALID"

    def test_reweight_edge_valid(self):
        graph = make_graph_buffers(
            num_nodes=6, edges=[(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=20,
        )
        new_graph, status = exact_transition(graph, ("reweight_edge", 0, 1, {"factor": 2.0}))
        assert status == "VALID"

    def test_edge_swap_valid(self):
        graph = make_graph_buffers(
            num_nodes=6, edges=[(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=20,
        )
        new_graph, status = exact_transition(
            graph, ("edge_swap", 0, 1, {"new_target": 3, "weight": 1.0}),
        )
        assert status == "VALID"

    def test_no_op_for_nonexisting_edge(self):
        graph = make_graph_buffers(
            num_nodes=6, edges=[(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=20,
        )
        new_graph, status = exact_transition(graph, ("remove_edge", 0, 5, {}))
        assert status == "NO_OP"


class TestRollForward:
    """Test roll-forward with exact and predicted transitions."""

    def test_roll_forward_exact(self):
        graph = make_graph_buffers(
            num_nodes=6, edges=[(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=20,
        )
        z = torch.randn(6, 4)
        state = StructuralState.from_graph(graph)
        new_state = roll_forward_exact(state, z, ("add_edge", 0, 3, {"weight": 1.0}))
        assert new_state.is_predicted is False
        # n_components should decrease (bridge added).
        assert new_state.z[1] <= state.z[1]

    def test_roll_forward_predicted(self):
        graph = make_graph_buffers(
            num_nodes=6, edges=[(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=20,
        )
        z = torch.randn(6, 4)
        model = ConsequentialStateModel(hidden_dim=16, n_epochs=10)
        # Not fitted — should return identity.
        state = StructuralState.from_graph(graph)
        new_state = roll_forward_predicted(state, z, ("add_edge", 0, 3, {"weight": 1.0}), model)
        assert new_state.is_predicted is True


class TestConsequentialStateModel:
    """Test the learned consequential state model."""

    def test_fit_and_predict(self):
        model = ConsequentialStateModel(hidden_dim=32, n_epochs=50, lr=0.01)
        # Feature dim = OBSERVABLE_FEATURE_DIM (64) + STRUCTURAL_OBSERVABLE_DIM (10) = 74.
        X = np.random.randn(50, 74).astype(np.float32)
        y = np.random.randn(50, STRUCTURAL_OBSERVABLE_DIM).astype(np.float32)
        model.fit(X, y)
        assert model._fitted

        graph = make_graph_buffers(num_nodes=6, edges=[(0,1),(1,2)], capacity=20)
        z = torch.randn(6, 4)
        z_state = compute_structural_observables(graph)
        z_pred = model.predict_z(graph, z, z_state, ("add_edge", 0, 2, {"weight": 1.0}))
        assert z_pred.shape == (STRUCTURAL_OBSERVABLE_DIM,)

    def test_predict_std(self):
        model = ConsequentialStateModel(hidden_dim=16, n_epochs=10)
        X = np.random.randn(20, 74).astype(np.float32)
        y = np.random.randn(20, STRUCTURAL_OBSERVABLE_DIM).astype(np.float32)
        model.fit(X, y)
        graph = make_graph_buffers(num_nodes=6, edges=[(0,1)], capacity=20)
        z = torch.randn(6, 4)
        z_state = compute_structural_observables(graph)
        std = model.predict_z_std(graph, z, z_state, ("add_edge", 0, 2, {"weight": 1.0}))
        assert std >= 0.0


class TestObjectiveEvaluation:
    """Test correct O(S+ΔS) - O(S) evaluation."""

    def test_threshold_reached(self):
        from lgae_v3.experimental.exp6_6.objective_spec import ObjectiveSpec
        spec = ObjectiveSpec(
            name="test", observable="n_components", direction="minimize",
            threshold=1.0, magnitude=30.0, reward_shape="threshold",
        )
        # current=2, after=1: reaches threshold.
        prev = StructuralState(
            graph=make_graph_buffers(num_nodes=4, edges=[(0,1),(2,3)], capacity=10),
            z=np.array([0.13, 2/6, 0.1, 0.1, 0.1, 0.0, 0.0, 0.5, 0.5, 0.0]),
        )
        after = StructuralState(
            graph=make_graph_buffers(num_nodes=4, edges=[(0,1),(1,2),(2,3)], capacity=10),
            z=np.array([0.13, 1/6, 0.15, 0.1, 0.2, 0.0, 0.0, 0.4, 0.6, 0.0]),
        )
        value = evaluate_objective_on_state(after, spec, prev)
        assert value > 0  # bonus for reaching threshold

    def test_threshold_not_reached(self):
        from lgae_v3.experimental.exp6_6.objective_spec import ObjectiveSpec
        spec = ObjectiveSpec(
            name="test", observable="n_components", direction="minimize",
            threshold=1.0, magnitude=30.0, reward_shape="threshold",
        )
        # current=4, after=3: does NOT reach threshold=1.
        prev = StructuralState(
            graph=make_graph_buffers(num_nodes=8, edges=[(0,1),(2,3),(4,5),(6,7)], capacity=20),
            z=np.array([0.27, 4/6, 0.05, 0.0, 0.1, 0.0, 0.0, 1.0, 0.3, 0.0]),
        )
        after = StructuralState(
            graph=make_graph_buffers(num_nodes=8, edges=[(0,1),(1,2),(3,4),(5,6),(6,7)], capacity=20),
            z=np.array([0.27, 3/6, 0.1, 0.05, 0.15, 0.0, 0.0, 0.8, 0.4, 0.0]),
        )
        value = evaluate_objective_on_state(after, spec, prev)
        assert value == 0.0  # no bonus for partial progress


class TestRecursivePlanner:
    """Test the recursive model-based planner."""

    def test_planner_runs(self):
        from lgae_v3.experimental.exp6_5.multi_mechanism_data import (
            generate_mechanism_task_configs, _make_graph_from_config,
        )
        from lgae_v3.experimental.exp6_7.multi_operator_candidates import (
            generate_multi_operator_candidates,
        )
        from lgae_v3.experimental.exp6_6.objective_spec import get_objective_spec

        configs = generate_mechanism_task_configs(
            mechanism="connectivity_threshold", n_tasks=1, seed=42,
        )
        config = configs[0]
        graph, z = _make_graph_from_config(config)
        candidates = generate_multi_operator_candidates(
            graph, z, config, rng=__import__("random").Random(config.seed),
        )
        model = ConsequentialStateModel(hidden_dim=32, n_epochs=20)
        # Quick fit with correct dimension (64 + 10 = 74).
        X = np.random.randn(20, 74).astype(np.float32)
        y = np.random.randn(20, STRUCTURAL_OBSERVABLE_DIM).astype(np.float32)
        model.fit(X, y)

        obj_spec = get_objective_spec("connectivity_threshold")
        result = recursive_causal_mpc(
            graph, z, candidates, model, obj_spec, config,
            horizon=2, gamma=0.9, beam_width=2,
            threshold=int(obj_spec.threshold),
        )
        assert result.horizon == 2
        assert result.nodes_expanded > 0
        assert result.planner_name == "recursive_causal_mpc"
