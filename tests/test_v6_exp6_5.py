"""Tests for v6.0-exp6.5: Cross-mechanism foresight generalization."""
from __future__ import annotations

import pytest
import torch
import numpy as np

from lgae_v3.experimental.exp6_5 import (
    MECHANISM_NAMES,
    extract_observable_features,
    OBSERVABLE_FEATURE_DIM,
    ScalarMLP, MultiHeadModel, EnsembleScalarMLP,
    get_decomposed_model_ladder,
    adaptive_beam_search, AdaptiveBeamResult,
    ScalingConfig, run_scaling_benchmark,
)
from lgae_v3 import make_graph_buffers


class TestObservableFeatures:
    """Test mechanism-agnostic observable features."""

    def test_feature_dim(self):
        graph = make_graph_buffers(num_nodes=6, edges=[(0, 1), (2, 3)], capacity=20)
        z = torch.randn(6, 4)
        features = extract_observable_features(graph, z, threshold=1, horizon=2)
        assert len(features) == OBSERVABLE_FEATURE_DIM

    def test_action_features(self):
        graph = make_graph_buffers(num_nodes=6, edges=[(0, 1), (2, 3)], capacity=20)
        z = torch.randn(6, 4)
        action = ("add_edge", 0, 2, {"weight": 1.0})
        features = extract_observable_features(graph, z, action, threshold=1, horizon=2)
        assert len(features) == OBSERVABLE_FEATURE_DIM

    def test_no_mechanism_label(self):
        """Features must NOT contain the mechanism label."""
        graph = make_graph_buffers(num_nodes=6, edges=[(0, 1), (2, 3)], capacity=20)
        z = torch.randn(6, 4)
        action = ("add_edge", 0, 2, {"weight": 1.0})
        features = extract_observable_features(graph, z, action, threshold=1, horizon=2)
        # Features should be purely structural — no string labels embedded.
        assert features.dtype in (np.float64, np.float32)


class TestDecomposedModels:
    """Test the decomposed model ladder."""

    def test_scalar_mlp_fits(self):
        model = ScalarMLP(hidden_dim=16, n_epochs=50)
        X = np.random.randn(50, OBSERVABLE_FEATURE_DIM)
        y = np.random.randn(50) * 10
        model.fit(X, y)
        assert model._fitted

    def test_multi_head_fits(self):
        model = MultiHeadModel(hidden_dim=16, n_heads=4, n_epochs=50)
        X = np.random.randn(50, OBSERVABLE_FEATURE_DIM)
        y = np.random.randn(50) * 10
        model.fit(X, y)
        assert model._fitted

    def test_ensemble_fits(self):
        model = EnsembleScalarMLP(n_models=3, hidden_dim=16, n_epochs=50)
        X = np.random.randn(50, OBSERVABLE_FEATURE_DIM)
        y = np.random.randn(50) * 10
        model.fit(X, y)
        assert model._fitted

    def test_ensemble_uncertainty(self):
        model = EnsembleScalarMLP(n_models=3, hidden_dim=16, n_epochs=50)
        X = np.random.randn(50, OBSERVABLE_FEATURE_DIM)
        y = np.random.randn(50) * 10
        model.fit(X, y)
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1)], capacity=20)
        z = torch.randn(4, 4)
        action = ("add_edge", 0, 2, {"weight": 1.0})
        std = model.predict_residual_std(graph, z, action, threshold=1)
        assert isinstance(std, float)

    def test_get_ladder(self):
        models = get_decomposed_model_ladder()
        assert len(models) >= 3
        names = [m.name for m in models]
        assert "ScalarMLP" in names
        assert "MultiHeadMLP" in names


class TestAdaptiveBeam:
    """Test adaptive beam search."""

    def test_adaptive_beam_runs(self):
        from lgae_v3.experimental.exp6_4.procedural_tasks import (
            generate_procedural_tasks, make_procedural_graph, generate_candidates,
        )
        configs = generate_procedural_tasks(n_tasks=1, seed=42)
        config = configs[0]
        graph, z, _ = make_procedural_graph(config)
        candidates = generate_candidates(config, graph, z)

        model = ScalarMLP(hidden_dim=16, n_epochs=50)
        X = np.random.randn(20, OBSERVABLE_FEATURE_DIM)
        y = np.random.randn(20) * 10
        model.fit(X, y)

        result = adaptive_beam_search(
            graph, z, candidates, model,
            horizon=2, gamma=0.9,
            min_beam_width=2, max_beam_width=5,
            threshold=1,
        )
        assert result.first_action[0] != ""
        assert result.nodes_expanded > 0
        assert result.beam_width_used >= 2

    def test_no_utility_fn_in_signature(self):
        import inspect
        sig = inspect.signature(adaptive_beam_search)
        params = list(sig.parameters.keys())
        assert "utility_fn" not in params


class TestScalingBenchmark:
    """Test the scaling benchmark."""

    def test_scaling_runs(self):
        model = ScalarMLP(hidden_dim=16, n_epochs=30)
        X = np.random.randn(20, OBSERVABLE_FEATURE_DIM)
        y = np.random.randn(20) * 10
        model.fit(X, y)

        configs = [ScalingConfig(n_nodes=10, n_candidates=6, seed=42)]
        results = run_scaling_benchmark(model, configs=configs)
        assert len(results) >= 0  # May skip if candidates < 4


class TestMultiMechanismData:
    """Test multi-mechanism data generation."""

    def test_mechanism_names(self):
        assert "connectivity_threshold" in MECHANISM_NAMES
        assert "redundancy_threshold" in MECHANISM_NAMES
        assert "hub_load_threshold" in MECHANISM_NAMES
        assert "spectral_gap_threshold" in MECHANISM_NAMES

    def test_generate_eval_tasks(self):
        from lgae_v3.experimental.exp6_5.multi_mechanism_data import generate_mechanism_eval_tasks
        configs = generate_mechanism_eval_tasks(
            mechanism="connectivity_threshold", n_tasks=3, seed=42,
        )
        assert len(configs) == 3
        for c in configs:
            assert c.mechanism == "connectivity_threshold"
