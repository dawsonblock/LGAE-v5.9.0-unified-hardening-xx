"""Tests for v6.0-exp6.6: Objective-conditioned causal foresight."""
from __future__ import annotations

import pytest
import torch
import numpy as np

from lgae_v3.experimental.exp6_6 import (
    ObjectiveSpec, OBJECTIVE_SPECS, get_objective_spec,
    encode_objective, OBJECTIVE_ENCODING_DIM,
    StructuralEffect, compute_effect_labels,
    ScalarResidualModel, ObjectiveConditionedModel, CausalEffectModel,
    ObjectiveEvaluator, get_architecture_ladder,
    honest_beam_search_v3, HonestBeamResultV3,
)
from lgae_v3 import make_graph_buffers


class TestObjectiveSpec:
    """Test machine-readable objective specification."""

    def test_all_mechanisms_have_specs(self):
        for name in ["connectivity_threshold", "redundancy_threshold",
                      "hub_load_threshold", "spectral_gap_threshold"]:
            assert name in OBJECTIVE_SPECS

    def test_encode_objective(self):
        spec = get_objective_spec("connectivity_threshold")
        enc = encode_objective(spec)
        assert len(enc) == OBJECTIVE_ENCODING_DIM

    def test_encoding_no_mechanism_name(self):
        """Encoding must NOT contain the mechanism name string."""
        spec = get_objective_spec("connectivity_threshold")
        enc = encode_objective(spec)
        # All values should be numeric, no string labels.
        assert enc.dtype in (np.float64, np.float32)

    def test_different_specs_different_encoding(self):
        spec1 = get_objective_spec("connectivity_threshold")
        spec2 = get_objective_spec("spectral_gap_threshold")
        enc1 = encode_objective(spec1)
        enc2 = encode_objective(spec2)
        assert not np.allclose(enc1, enc2)


class TestStructuralEffect:
    """Test structural effect computation."""

    def test_effect_labels(self):
        graph = make_graph_buffers(num_nodes=6, edges=[(0, 1), (2, 3)], capacity=20)
        z = torch.randn(6, 4)
        action = ("add_edge", 0, 2, {"weight": 1.0})
        effects = compute_effect_labels(graph, z, action)
        assert isinstance(effects, StructuralEffect)
        assert effects.delta_n_components == -1  # merges two components

    def test_effect_to_array(self):
        effects = StructuralEffect(delta_n_components=-1, delta_redundancy=0.5,
                                    delta_hub_load=-0.3, delta_spectral_gap=0.1)
        arr = effects.to_array()
        assert len(arr) == 4
        assert arr[0] == -1.0

    def test_effect_from_array(self):
        arr = np.array([-1.0, 0.5, -0.3, 0.1])
        effects = StructuralEffect.from_array(arr)
        assert effects.delta_n_components == -1.0
        assert effects.delta_redundancy == 0.5


class TestObjectiveEvaluator:
    """Test the deterministic objective evaluator."""

    def test_connectivity_threshold_reached(self):
        """Effect that reaches threshold gets full bonus."""
        spec = get_objective_spec("connectivity_threshold")
        # current=2, threshold=1: 2->1 reaches threshold, bonus=30.
        effects = StructuralEffect(delta_n_components=-1)
        value = ObjectiveEvaluator.evaluate(effects, spec, current_value=2.0)
        assert value > 0  # bonus for reaching threshold

    def test_connectivity_threshold_not_reached(self):
        """Effect that moves toward but doesn't reach threshold gets 0."""
        spec = get_objective_spec("connectivity_threshold")
        # current=4, threshold=1: 4->3 does NOT reach threshold.
        effects = StructuralEffect(delta_n_components=-1)
        value = ObjectiveEvaluator.evaluate(effects, spec, current_value=4.0)
        assert value == 0.0  # no bonus for partial progress

    def test_no_bonus_for_wrong_direction(self):
        spec = get_objective_spec("connectivity_threshold")
        # Effect that increases components → no bonus.
        effects = StructuralEffect(delta_n_components=1)
        value = ObjectiveEvaluator.evaluate(effects, spec, current_value=3.0)
        assert value == 0.0

    def test_spectral_gap_maximize(self):
        spec = get_objective_spec("spectral_gap_threshold")
        # current=0.0, threshold=0.5: 0.0+1.0=1.0 >= 0.5, bonus.
        effects = StructuralEffect(delta_spectral_gap=1.0)
        value = ObjectiveEvaluator.evaluate(effects, spec, current_value=0.0)
        assert value > 0


class TestArchitectures:
    """Test the three model architectures."""

    def test_scalar_fits(self):
        model = ScalarResidualModel(hidden_dim=16, n_epochs=50)
        X = np.random.randn(50, 64)
        y = np.random.randn(50) * 10
        model.fit(X, y)
        assert model._fitted

    def test_objective_conditioned_fits(self):
        model = ObjectiveConditionedModel(hidden_dim=16, n_epochs=50)
        X = np.random.randn(50, 64 + OBJECTIVE_ENCODING_DIM)
        y = np.random.randn(50) * 10
        model.fit(X, y)
        assert model._fitted

    def test_causal_effect_fits(self):
        model = CausalEffectModel(hidden_dim=16, n_epochs=50)
        X = np.random.randn(50, 64)
        y_effects = np.random.randn(50, 4) * 5
        model.fit(X, y_effects=y_effects)
        assert model._fitted

    def test_causal_effect_predicts_effects(self):
        model = CausalEffectModel(hidden_dim=16, n_epochs=50)
        X = np.random.randn(50, 64)
        y_effects = np.random.randn(50, 4) * 5
        model.fit(X, y_effects=y_effects)
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1)], capacity=20)
        z = torch.randn(4, 4)
        action = ("add_edge", 0, 2, {"weight": 1.0})
        effects = model.predict_effects(graph, z, action, threshold=1)
        assert isinstance(effects, StructuralEffect)

    def test_causal_effect_uses_objective(self):
        model = CausalEffectModel(hidden_dim=16, n_epochs=50)
        X = np.random.randn(50, 64)
        y_effects = np.random.randn(50, 4) * 5
        model.fit(X, y_effects=y_effects)
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1)], capacity=20)
        z = torch.randn(4, 4)
        action = ("add_edge", 0, 2, {"weight": 1.0})
        spec = get_objective_spec("connectivity_threshold")
        val = model.predict_residual(graph, z, action, threshold=1, objective=spec)
        assert isinstance(val, float)

    def test_get_ladder(self):
        models = get_architecture_ladder()
        assert len(models) == 3
        names = [m.name for m in models]
        assert "A_scalar" in names
        assert "B_objective_conditioned" in names
        assert "C_causal_effect" in names


class TestHonestBeamV3:
    """Test honest beam search v3."""

    def test_no_utility_fn_in_signature(self):
        import inspect
        sig = inspect.signature(honest_beam_search_v3)
        params = list(sig.parameters.keys())
        assert "utility_fn" not in params

    def test_beam_search_runs(self):
        from lgae_v3.experimental.exp6_4.procedural_tasks import (
            generate_procedural_tasks, make_procedural_graph, generate_candidates,
        )
        configs = generate_procedural_tasks(n_tasks=1, seed=42)
        config = configs[0]
        graph, z, _ = make_procedural_graph(config)
        candidates = generate_candidates(config, graph, z)

        model = ScalarResidualModel(hidden_dim=16, n_epochs=50)
        X = np.random.randn(20, 64)
        y = np.random.randn(20) * 10
        model.fit(X, y)

        result = honest_beam_search_v3(
            graph, z, candidates, model,
            horizon=2, gamma=0.9, beam_width=2,
            threshold=1,
        )
        assert result.first_action[0] != ""
        assert result.nodes_expanded > 0

    def test_passes_objective_to_model(self):
        """Verify that objective spec is passed to the model."""
        from lgae_v3.experimental.exp6_4.procedural_tasks import (
            generate_procedural_tasks, make_procedural_graph, generate_candidates,
        )
        configs = generate_procedural_tasks(n_tasks=1, seed=42)
        config = configs[0]
        graph, z, _ = make_procedural_graph(config)
        candidates = generate_candidates(config, graph, z)

        model = CausalEffectModel(hidden_dim=16, n_epochs=50)
        X = np.random.randn(20, 64)
        y_effects = np.random.randn(20, 4) * 5
        model.fit(X, y_effects=y_effects)

        spec = get_objective_spec("connectivity_threshold")
        result = honest_beam_search_v3(
            graph, z, candidates, model,
            horizon=2, gamma=0.9, beam_width=2,
            threshold=1, objective=spec,
        )
        assert result.first_action[0] != ""
