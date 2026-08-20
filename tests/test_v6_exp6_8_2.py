"""Tests for v6.0-exp6.8.2: Calibrated selective planning."""
import numpy as np
import torch
import pytest

from lgae_v3 import make_graph_buffers
from lgae_v3.experimental.exp6_8_2 import (
    EnsembleLearnedModel,
    lcb_hybrid_plan, LCBPlanResult,
    compute_cvar, compute_extended_risk_metrics,
    compute_uncertainty_error_correlation,
    compute_risk_by_uncertainty_deciles,
)


class TestCVaR:
    """Test Conditional Value at Risk."""

    def test_cvar_basic(self):
        regrets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cvar95 = compute_cvar(regrets, 95)
        # P95 of [1..10] = 9.5, tail = [10], CVaR = 10.
        assert cvar95 >= 9.0  # approximately the max

    def test_cvar_empty(self):
        assert compute_cvar(np.array([]), 95) == 0.0

    def test_cvar_uniform(self):
        regrets = np.array([5.0] * 20)
        cvar95 = compute_cvar(regrets, 95)
        assert abs(cvar95 - 5.0) < 1e-6

    def test_extended_risk_metrics(self):
        regrets = np.array([0, 1, 2, 5, 10, 20, 50, 100])
        m = compute_extended_risk_metrics(regrets)
        assert "cvar95" in m
        assert "cvar99" in m
        assert "median_regret" in m
        assert m["cvar95"] >= m["median_regret"]


class TestUncertaintyCorrelation:
    """Test uncertainty-error correlation."""

    def test_positive_correlation(self):
        # High uncertainty -> high error.
        uncertainties = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
        errors =        [0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
        result = compute_uncertainty_error_correlation(uncertainties, errors)
        assert result["correlation"] > 0.5  # should be strongly positive
        assert result["n_samples"] == 10

    def test_no_correlation(self):
        uncertainties = [1.0, 1.0, 1.0, 1.0, 1.0]
        errors = [0.1, 0.5, 0.3, 0.8, 0.2]
        result = compute_uncertainty_error_correlation(uncertainties, errors)
        # Zero variance in uncertainties -> correlation = 0.
        assert abs(result["correlation"]) < 1e-6

    def test_too_few_samples(self):
        result = compute_uncertainty_error_correlation([1.0], [0.5])
        assert result["n_samples"] == 0


class TestRiskDeciles:
    """Test risk-by-uncertainty decile analysis."""

    def test_monotonic_increasing(self):
        # Regret increases with uncertainty.
        uncertainties = list(np.linspace(0, 10, 100))
        regrets = list(np.linspace(0, 100, 100))  # perfectly correlated
        result = compute_risk_by_uncertainty_deciles(uncertainties, regrets)
        assert result["is_monotonic"] is True
        assert len(result["deciles"]) == 10
        # First decile should have lower mean regret than last.
        assert result["deciles"][0]["mean_regret"] < result["deciles"][-1]["mean_regret"]

    def test_non_monotonic(self):
        # Regret decreases with uncertainty (anti-correlated).
        uncertainties = list(np.linspace(0, 10, 100))
        regrets = list(np.linspace(100, 0, 100))  # anti-correlated
        result = compute_risk_by_uncertainty_deciles(uncertainties, regrets)
        assert result["is_monotonic"] is False

    def test_too_few_samples(self):
        result = compute_risk_by_uncertainty_deciles([1, 2, 3], [0.1, 0.2, 0.3])
        assert result["deciles"] == []


class TestEnsembleModel:
    """Test the ensemble learned state model."""

    def test_fit_and_predict(self):
        model = EnsembleLearnedModel(n_members=3, hidden_dim=16, n_epochs=20)
        X = np.random.randn(30, 76).astype(np.float32)
        y = np.random.randn(30, 3).astype(np.float32)
        model.fit(X, y)
        assert model._fitted
        assert len(model._members) == 3

        graph = make_graph_buffers(num_nodes=6, edges=[(0,1),(1,2)], capacity=20)
        z = torch.randn(6, 4)
        from lgae_v3.experimental.exp6_8_1.split_state import SplitStructuralState
        state = SplitStructuralState.from_graph(graph)
        mean, std = model.predict_learned_ensemble(
            graph, z, state, ("add_edge", 0, 2, {"weight": 1.0}),
        )
        assert mean.shape == (3,)
        assert std.shape == (3,)
        # Std should be non-negative.
        assert np.all(std >= 0)

    def test_predict_uncertainty(self):
        model = EnsembleLearnedModel(n_members=3, hidden_dim=16, n_epochs=20)
        X = np.random.randn(30, 76).astype(np.float32)
        y = np.random.randn(30, 3).astype(np.float32)
        model.fit(X, y)

        graph = make_graph_buffers(num_nodes=6, edges=[(0,1)], capacity=20)
        z = torch.randn(6, 4)
        from lgae_v3.experimental.exp6_8_1.split_state import SplitStructuralState
        state = SplitStructuralState.from_graph(graph)
        std = model.predict_uncertainty(
            graph, z, state, ("add_edge", 0, 2, {"weight": 1.0}),
        )
        assert std >= 0.0

    def test_predict_z_compatibility(self):
        """Test compatibility with exp6_8 recursive_causal_mpc."""
        model = EnsembleLearnedModel(n_members=3, hidden_dim=16, n_epochs=20)
        X = np.random.randn(30, 76).astype(np.float32)
        y = np.random.randn(30, 3).astype(np.float32)
        model.fit(X, y)

        graph = make_graph_buffers(num_nodes=6, edges=[(0,1),(1,2)], capacity=20)
        z = torch.randn(6, 4)
        z_state = np.random.randn(12).astype(np.float32)  # 6+3+3
        pred = model.predict_z(graph, z, z_state, ("add_edge", 0, 3, {"weight": 1.0}))
        assert pred.shape == z_state.shape


class TestLCBPlanner:
    """Test the LCB-margin hybrid planner."""

    def test_planner_runs(self):
        from lgae_v3.experimental.exp6_5.multi_mechanism_data import (
            generate_mechanism_task_configs, _make_graph_from_config,
        )
        from lgae_v3.experimental.exp6_7.multi_operator_candidates import (
            generate_multi_operator_candidates,
        )
        from lgae_v3.experimental.exp6_6.objective_spec import get_objective_spec
        from lgae_v3.experimental.exp6_4.test_f import make_test_f_utility

        configs = generate_mechanism_task_configs(
            mechanism="connectivity_threshold", n_tasks=1, seed=42,
        )
        config = configs[0]
        graph, z = _make_graph_from_config(config)
        candidates = generate_multi_operator_candidates(
            graph, z, config, rng=__import__("random").Random(config.seed),
        )

        model = EnsembleLearnedModel(n_members=3, hidden_dim=16, n_epochs=20)
        X = np.random.randn(20, 76).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y)

        obj_spec = get_objective_spec("connectivity_threshold")
        utility_fn = make_test_f_utility(
            "connectivity_threshold", config.lambda_bonus, int(obj_spec.threshold),
        )

        result = lcb_hybrid_plan(
            graph, z, candidates, model, obj_spec, config, utility_fn,
            horizon=2, gamma=0.9, beam_width=2,
            threshold=int(obj_spec.threshold), kappa=1.0,
        )
        assert result.planner_name == "lcb_hybrid"
        assert result.source in ["learned", "greedy_fallback"]
        assert result.kappa == 1.0
        assert hasattr(result, "lcb_margin")

    def test_high_kappa_falls_back(self):
        """With very high kappa, LCB is very negative, so falls back to greedy."""
        from lgae_v3.experimental.exp6_5.multi_mechanism_data import (
            generate_mechanism_task_configs, _make_graph_from_config,
        )
        from lgae_v3.experimental.exp6_7.multi_operator_candidates import (
            generate_multi_operator_candidates,
        )
        from lgae_v3.experimental.exp6_6.objective_spec import get_objective_spec
        from lgae_v3.experimental.exp6_4.test_f import make_test_f_utility

        configs = generate_mechanism_task_configs(
            mechanism="connectivity_threshold", n_tasks=1, seed=42,
        )
        config = configs[0]
        graph, z = _make_graph_from_config(config)
        candidates = generate_multi_operator_candidates(
            graph, z, config, rng=__import__("random").Random(config.seed),
        )

        model = EnsembleLearnedModel(n_members=3, hidden_dim=16, n_epochs=20)
        X = np.random.randn(20, 76).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y)

        obj_spec = get_objective_spec("connectivity_threshold")
        utility_fn = make_test_f_utility(
            "connectivity_threshold", config.lambda_bonus, int(obj_spec.threshold),
        )

        # Very high kappa: LCB very negative, should fall back.
        result = lcb_hybrid_plan(
            graph, z, candidates, model, obj_spec, config, utility_fn,
            horizon=2, gamma=0.9, beam_width=2,
            threshold=int(obj_spec.threshold), kappa=1000.0,
        )
        assert result.used_learned is False
        assert result.source == "greedy_fallback"

    def test_zero_kappa_uses_learned_if_margin_positive(self):
        """With kappa=0, uses learned if margin > 0."""
        from lgae_v3.experimental.exp6_5.multi_mechanism_data import (
            generate_mechanism_task_configs, _make_graph_from_config,
        )
        from lgae_v3.experimental.exp6_7.multi_operator_candidates import (
            generate_multi_operator_candidates,
        )
        from lgae_v3.experimental.exp6_6.objective_spec import get_objective_spec
        from lgae_v3.experimental.exp6_4.test_f import make_test_f_utility

        configs = generate_mechanism_task_configs(
            mechanism="connectivity_threshold", n_tasks=1, seed=42,
        )
        config = configs[0]
        graph, z = _make_graph_from_config(config)
        candidates = generate_multi_operator_candidates(
            graph, z, config, rng=__import__("random").Random(config.seed),
        )

        model = EnsembleLearnedModel(n_members=3, hidden_dim=16, n_epochs=20)
        X = np.random.randn(20, 76).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y)

        obj_spec = get_objective_spec("connectivity_threshold")
        utility_fn = make_test_f_utility(
            "connectivity_threshold", config.lambda_bonus, int(obj_spec.threshold),
        )

        # kappa=0: LCB = margin. Uses learned if margin > 0.
        result = lcb_hybrid_plan(
            graph, z, candidates, model, obj_spec, config, utility_fn,
            horizon=2, gamma=0.9, beam_width=2,
            threshold=int(obj_spec.threshold), kappa=0.0,
        )
        # Should use learned if margin is positive.
        if result.margin_mean > 0:
            assert result.used_learned is True
            assert result.source == "learned"
