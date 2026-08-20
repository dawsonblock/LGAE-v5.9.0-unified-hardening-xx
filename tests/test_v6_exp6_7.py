"""Tests for v6.0-exp6.7: Multi-operator causal structural model."""
from __future__ import annotations

import pytest
import torch
import numpy as np

from lgae_v3.experimental.exp6_7 import (
    MUTATION_TYPES, generate_multi_operator_candidates,
    ExtendedEffect, compute_extended_effect_labels, EXTENDED_EFFECT_DIM,
    CausalEffectModelV2, ObjectiveEvaluatorV2,
    ScalarResidualModelV2, get_architecture_ladder_v2,
    RewardVariant, make_reward_variant_utility, REWARD_VARIANTS,
)
from lgae_v3 import make_graph_buffers
from lgae_v3.experimental.exp6_6.objective_spec import ObjectiveSpec


class TestMultiOperatorCandidates:
    """Test multi-operator candidate generation."""

    def test_mutation_types(self):
        assert "add_edge" in MUTATION_TYPES
        assert "remove_edge" in MUTATION_TYPES
        assert "reweight_edge" in MUTATION_TYPES
        assert "edge_swap" in MUTATION_TYPES

    def test_generate_candidates(self):
        from lgae_v3.experimental.exp6_5.multi_mechanism_data import (
            generate_mechanism_task_configs, _make_graph_from_config,
        )
        configs = generate_mechanism_task_configs(
            mechanism="connectivity_threshold", n_tasks=1, seed=42,
        )
        config = configs[0]
        graph, z = _make_graph_from_config(config)
        candidates = generate_multi_operator_candidates(graph, z, config)
        assert len(candidates) >= 4
        types = {c[0] for c in candidates}
        # Should have at least 2 different mutation types.
        assert len(types) >= 2


class TestExtendedEffects:
    """Test 7-head structural effect computation."""

    def test_effect_dim(self):
        assert EXTENDED_EFFECT_DIM == 7

    def test_effect_labels(self):
        graph = make_graph_buffers(num_nodes=6, edges=[(0, 1), (2, 3)], capacity=20)
        z = torch.randn(6, 4)
        action = ("add_edge", 0, 2, {"weight": 1.0})
        effects = compute_extended_effect_labels(graph, z, action)
        assert isinstance(effects, ExtendedEffect)
        assert effects.delta_n_components == -1  # merges two components

    def test_effect_to_array(self):
        effects = ExtendedEffect(
            delta_n_components=-1, delta_redundancy=0.5,
            delta_hub_load=-0.3, delta_spectral_gap=0.1,
            delta_path_length=-0.2, delta_efficiency=0.05,
            delta_curvature=0.3,
        )
        arr = effects.to_array()
        assert len(arr) == 7

    def test_effect_from_array(self):
        arr = np.array([-1.0, 0.5, -0.3, 0.1, -0.2, 0.05, 0.3])
        effects = ExtendedEffect.from_array(arr)
        assert effects.delta_n_components == -1.0
        assert effects.delta_path_length == -0.2

    def test_effect_remove_edge(self):
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1), (1, 2), (2, 3)], capacity=20)
        z = torch.randn(4, 4)
        action = ("remove_edge", 1, 2, {})
        effects = compute_extended_effect_labels(graph, z, action)
        # Removing an edge should increase components or decrease none.
        assert isinstance(effects, ExtendedEffect)


class TestCausalEffectModelV2:
    """Test the 7-head causal effect model."""

    def test_scalar_fits(self):
        model = ScalarResidualModelV2(hidden_dim=16, n_epochs=50)
        X = np.random.randn(50, 64)
        y = np.random.randn(50) * 10
        model.fit(X, y)
        assert model._fitted

    def test_causal_fits(self):
        model = CausalEffectModelV2(hidden_dim=16, n_epochs=50)
        X = np.random.randn(50, 64)
        y_effects = np.random.randn(50, 7) * 5
        model.fit(X, y_effects=y_effects)
        assert model._fitted

    def test_causal_predicts_7_effects(self):
        model = CausalEffectModelV2(hidden_dim=16, n_epochs=50)
        X = np.random.randn(50, 64)
        y_effects = np.random.randn(50, 7) * 5
        model.fit(X, y_effects=y_effects)
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1)], capacity=20)
        z = torch.randn(4, 4)
        action = ("add_edge", 0, 2, {"weight": 1.0})
        effects = model.predict_effects(graph, z, action, threshold=1)
        assert isinstance(effects, ExtendedEffect)

    def test_causal_uses_objective(self):
        model = CausalEffectModelV2(hidden_dim=16, n_epochs=50)
        X = np.random.randn(50, 64)
        y_effects = np.random.randn(50, 7) * 5
        model.fit(X, y_effects=y_effects)
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1)], capacity=20)
        z = torch.randn(4, 4)
        action = ("add_edge", 0, 2, {"weight": 1.0})
        spec = ObjectiveSpec(name="test", observable="n_components",
                             direction="minimize", threshold=1.0,
                             magnitude=25.0, reward_shape="threshold")
        val = model.predict_residual(graph, z, action, threshold=1, objective=spec)
        assert isinstance(val, float)

    def test_get_ladder_v2(self):
        models = get_architecture_ladder_v2()
        assert len(models) == 2


class TestObjectiveEvaluatorV2:
    """Test the objective evaluator with 7 effects."""

    def test_threshold_connectivity(self):
        spec = ObjectiveSpec(name="test", observable="n_components",
                             direction="minimize", threshold=1.0,
                             magnitude=25.0, reward_shape="threshold")
        effects = ExtendedEffect(delta_n_components=-1)
        # current_value=2, threshold=1: 2->1 reaches threshold, bonus=25.
        # At current=2, threshold not reached, bonus=0.
        # So delta = 25 - 0 = 25.
        value = ObjectiveEvaluatorV2.evaluate(effects, spec, current_value=2.0)
        assert value > 0

    def test_threshold_not_reached(self):
        """Effect that moves toward but doesn't reach threshold gets 0 bonus."""
        spec = ObjectiveSpec(name="test", observable="n_components",
                             direction="minimize", threshold=1.0,
                             magnitude=25.0, reward_shape="threshold")
        effects = ExtendedEffect(delta_n_components=-1)
        # current_value=4, threshold=1: 4->3 does NOT reach threshold.
        # bonus_after=0, bonus_current=0, delta=0.
        value = ObjectiveEvaluatorV2.evaluate(effects, spec, current_value=4.0)
        assert value == 0.0

    def test_linear_reward(self):
        spec = ObjectiveSpec(name="test", observable="spectral_gap",
                             direction="maximize", threshold=0.5,
                             magnitude=20.0, reward_shape="linear")
        effects = ExtendedEffect(delta_spectral_gap=0.1)
        value = ObjectiveEvaluatorV2.evaluate(effects, spec)
        assert value > 0

    def test_composite(self):
        effects = ExtendedEffect(delta_spectral_gap=0.1, delta_redundancy=0.5)
        value = ObjectiveEvaluatorV2.evaluate_composite(
            effects, {"spectral_gap": 2.0, "redundancy": 0.5}
        )
        assert value > 0


class TestRewardVariants:
    """Test reward-formulation variants."""

    def test_all_mechanisms_have_variants(self):
        for mech in ["connectivity_threshold", "spectral_gap_threshold",
                      "redundancy_threshold", "hub_load_threshold"]:
            assert mech in REWARD_VARIANTS
            assert "linear" in REWARD_VARIANTS[mech]
            assert "composite" in REWARD_VARIANTS[mech]

    def test_threshold_variant(self):
        fn = make_reward_variant_utility("connectivity_threshold", "threshold", 30.0, 1)
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1)], capacity=20)
        z = torch.randn(4, 4)
        val = fn(graph, z)
        assert isinstance(val, float)

    def test_linear_variant(self):
        fn = make_reward_variant_utility("connectivity_threshold", "linear", 30.0, 1)
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1)], capacity=20)
        z = torch.randn(4, 4)
        val = fn(graph, z)
        assert isinstance(val, float)

    def test_composite_variant(self):
        fn = make_reward_variant_utility("spectral_gap_threshold", "composite", 20.0, 0.5)
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1), (1, 2), (2, 3)], capacity=20)
        z = torch.randn(4, 4)
        val = fn(graph, z)
        assert isinstance(val, float)


class TestEdgeSwap:
    """Test edge_swap mutation support."""

    def test_edge_swap_in_apply_action(self):
        from lgae_v3.experimental.exp6_3.exact_mpc import apply_action
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1), (1, 2)], capacity=20)
        z = torch.randn(4, 4)
        action = ("edge_swap", 0, 1, {"new_target": 3, "weight": 1.0})
        new_graph = apply_action(graph, action)
        assert new_graph is not None

    def test_edge_swap_analytical(self):
        from lgae_v3.runtime.analytical_utility import AnalyticalUtilityOracle
        oracle = AnalyticalUtilityOracle()
        graph = make_graph_buffers(num_nodes=4, edges=[(0, 1), (1, 2)], capacity=20)
        z = torch.randn(4, 4)
        delta = oracle.delta_for_mutation(graph, z, "edge_swap", 0, 1, {"new_target": 3})
        assert isinstance(delta, float)
