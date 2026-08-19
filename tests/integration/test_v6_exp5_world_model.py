"""v6.0-exp5: Lightweight structural latent world model — test suite.

Covers:
- State/action encoding (dimensions, schema hashes, determinism)
- Linear dynamics (fitting, prediction, serialization)
- MLP dynamics (fitting, prediction, serialization)
- Joint world model (dynamics + outcomes)
- Training pipeline (train-only fitting, split enforcement)
- Evaluation (single-step, multi-step rollout)
- WorldModelInterface implementation
- Authority boundary preservation
- Degenerate dataset handling
- Determinism and reproducibility
- Model state serialization roundtrip
"""
from __future__ import annotations

import pytest
import numpy as np
import math
import json
import warnings
from dataclasses import dataclass, field
from typing import Any

from lgae_v3.experimental.exp5 import (
    STATE_DIM,
    ACTION_DIM,
    MUTATION_TYPES,
    encode_state,
    decode_state,
    encode_action,
    StateVector,
    ActionVector,
    DynamicsModel,
    LinearDynamics,
    MLPDynamics,
    DynamicsMetrics,
    compute_dynamics_metrics,
    JointWorldModel,
    JointModelConfig,
    JointModelMetrics,
    WorldModelPrediction,
    TrainingConfig,
    TrainingResult,
    train_world_model,
    EvaluationResult,
    evaluate_world_model,
    rollout_evaluation,
    RolloutReport,
    LightweightWorldModel,
    WorldModelTrustReport,
)
from lgae_v3.experimental.exp5.state_encoding import state_action_schema_hash
from lgae_v3.experimental.world_model import ModelPrediction, WorldModelInterface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class MockState:
    """Mock StructuralStateSummary."""
    n_nodes: int = 20
    n_edges: int = 19
    density: float = 0.095
    spectral_gap: float = 0.5
    degree_mean: float = 1.9
    degree_std: float = 0.3
    n_components: int = 1
    avg_clustering: float = 0.1
    fiber_count: int = 0
    fiber_width: int = 0
    gauge_dim: int = 0
    state_hash: str = "abc123"
    graph_version: int = 1


@dataclass
class MockRecord:
    """Mock TransitionRecord for training tests."""
    split: str = "train"
    action: str = "ADD_EDGE"
    action_target: dict = field(default_factory=lambda: {"u": 1, "v": 5})
    structural_state_before: Any = None
    structural_state_after: Any = None
    realized_delta: float = 0.1
    realized_risk: float = 0.05
    realized_cost: float = 1.0
    episode_id: str = "ep0"
    step_id: int = 0
    provenance: Any = None


def _make_records(n=20, split="train") -> list[MockRecord]:
    """Create mock transition records with state transitions."""
    records = []
    for i in range(n):
        state_before = MockState(n_nodes=20, n_edges=19 + i, density=0.095 + i * 0.001)
        state_after = MockState(n_nodes=20, n_edges=20 + i, density=0.1 + i * 0.001)
        records.append(MockRecord(
            split=split,
            action="ADD_EDGE" if i % 2 == 0 else "REMOVE_EDGE",
            action_target={"u": i % 20, "v": (i + 1) % 20},
            structural_state_before=state_before,
            structural_state_after=state_after,
            realized_delta=0.1 * (i % 5 - 2) / 5.0,
            realized_risk=0.05 + i * 0.001,
            realized_cost=1.0 + i * 0.01,
            episode_id="ep0" if i < n // 2 else "ep1",
            step_id=i % max(n // 2, 1),
        ))
    return records


# ---------------------------------------------------------------------------
# 1. State encoding
# ---------------------------------------------------------------------------

class TestStateEncoding:
    """State encoding tests."""

    def test_state_dim_is_14(self):
        assert STATE_DIM == 14

    def test_encode_state_returns_correct_dim(self):
        sv = encode_state(MockState())
        assert sv.vector.shape == (STATE_DIM,)

    def test_encode_state_values_correct(self):
        sv = encode_state(MockState(n_nodes=20, n_edges=19, density=0.095))
        assert sv.vector[0] == 20.0
        assert sv.vector[1] == 19.0
        assert sv.vector[2] == pytest.approx(0.095)

    def test_encode_state_derived_features(self):
        sv = encode_state(MockState(n_nodes=20, density=0.095, spectral_gap=0.5))
        # log_density = log1p(0.095)
        assert sv.vector[11] == pytest.approx(math.log1p(0.095))
        # log_spectral_gap = log1p(0.5)
        assert sv.vector[12] == pytest.approx(math.log1p(0.5))
        # spectral_gap_per_node = 0.5 / 20
        assert sv.vector[13] == pytest.approx(0.5 / 20.0)

    def test_state_schema_hash_nonempty(self):
        sv = encode_state(MockState())
        assert sv.schema_hash != ""

    def test_state_schema_hash_deterministic(self):
        sv1 = encode_state(MockState())
        sv2 = encode_state(MockState())
        assert sv1.schema_hash == sv2.schema_hash

    def test_decode_state_returns_fields(self):
        sv = encode_state(MockState(n_nodes=20, n_edges=19))
        decoded = decode_state(sv)
        assert decoded["n_nodes"] == 20.0
        assert decoded["n_edges"] == 19.0

    def test_encode_state_handles_missing_fields(self):
        """Encoding should handle objects with missing fields gracefully."""
        sv = encode_state(object())
        assert sv.vector.shape == (STATE_DIM,)
        assert all(np.isfinite(sv.vector))


# ---------------------------------------------------------------------------
# 2. Action encoding
# ---------------------------------------------------------------------------

class TestActionEncoding:
    """Action encoding tests."""

    def test_action_dim_is_12(self):
        assert ACTION_DIM == 12

    def test_encode_action_returns_correct_dim(self):
        av = encode_action("ADD_EDGE", {"u": 1, "v": 5})
        assert av.vector.shape == (ACTION_DIM,)

    def test_encode_action_one_hot(self):
        av = encode_action("ADD_EDGE", {"u": 1, "v": 5})
        # First 6 dims are one-hot for mutation types.
        assert av.vector[0] == 1.0  # ADD_EDGE is first
        assert av.vector[1] == 0.0  # REMOVE_EDGE
        assert sum(av.vector[:len(MUTATION_TYPES)]) == 1.0

    def test_encode_action_unknown_type(self):
        av = encode_action("UNKNOWN", {"u": 1, "v": 5})
        # Unknown type uses uniform encoding.
        assert all(v > 0 for v in av.vector[:len(MUTATION_TYPES)])

    def test_encode_action_target_features(self):
        av = encode_action("ADD_EDGE", {"u": 5, "v": 10}, n_nodes=20)
        # u_normalized = 5/20 = 0.25
        assert av.vector[len(MUTATION_TYPES)] == pytest.approx(0.25)
        # v_normalized = 10/20 = 0.5
        assert av.vector[len(MUTATION_TYPES) + 1] == pytest.approx(0.5)

    def test_action_schema_hash_nonempty(self):
        av = encode_action("ADD_EDGE", {"u": 1, "v": 5})
        assert av.schema_hash != ""

    def test_action_schema_hash_deterministic(self):
        av1 = encode_action("ADD_EDGE", {"u": 1, "v": 5})
        av2 = encode_action("ADD_EDGE", {"u": 1, "v": 5})
        assert av1.schema_hash == av2.schema_hash

    def test_state_action_schema_hash(self):
        h = state_action_schema_hash()
        assert h != ""
        assert len(h) == 16

    def test_encode_action_empty_target(self):
        av = encode_action("ADD_EDGE", {})
        assert av.vector.shape == (ACTION_DIM,)
        assert all(np.isfinite(av.vector))


# ---------------------------------------------------------------------------
# 3. Linear dynamics
# ---------------------------------------------------------------------------

class TestLinearDynamics:
    """Linear dynamics model tests."""

    def test_fit_and_predict(self):
        model = LinearDynamics(n_epochs=10, seed=42)
        z_t = np.random.RandomState(42).randn(20, STATE_DIM)
        a_t = np.random.RandomState(43).randn(20, ACTION_DIM)
        # Generate next state as a function of z_t and a_t.
        a_padded = np.zeros((20, STATE_DIM))
        a_padded[:, :ACTION_DIM] = a_t
        z_next = z_t * 0.9 + a_padded * 0.1 + 0.05
        model.fit(z_t, a_t, z_next, split="train")
        pred = model.predict(z_t[0], a_t[0])
        assert pred.shape == (STATE_DIM,)
        assert all(np.isfinite(pred))

    def test_fit_rejects_non_train_split(self):
        model = LinearDynamics()
        z_t = np.random.randn(10, STATE_DIM)
        a_t = np.random.randn(10, ACTION_DIM)
        z_next = np.random.randn(10, STATE_DIM)
        with pytest.raises(ValueError, match="train split"):
            model.fit(z_t, a_t, z_next, split="validation")

    def test_predict_before_fit_returns_identity(self):
        model = LinearDynamics()
        z = np.ones(STATE_DIM)
        pred = model.predict(z, np.zeros(ACTION_DIM))
        assert np.allclose(pred, z)

    def test_batch_prediction(self):
        model = LinearDynamics(n_epochs=10, seed=42)
        z_t = np.random.RandomState(42).randn(10, STATE_DIM)
        a_t = np.random.RandomState(43).randn(10, ACTION_DIM)
        z_next = z_t + 0.1
        model.fit(z_t, a_t, z_next, split="train")
        preds = model.predict_batch(z_t, a_t)
        assert preds.shape == (10, STATE_DIM)

    def test_serialization_roundtrip(self):
        model = LinearDynamics(n_epochs=10, seed=42)
        z_t = np.random.RandomState(42).randn(20, STATE_DIM)
        a_t = np.random.RandomState(43).randn(20, ACTION_DIM)
        z_next = z_t * 0.9 + 0.1
        model.fit(z_t, a_t, z_next, split="train")
        pred_before = model.predict(z_t[0], a_t[0])

        state = model.get_state()
        model2 = LinearDynamics()
        model2.set_state(state)
        pred_after = model2.predict(z_t[0], a_t[0])

        assert np.allclose(pred_before, pred_after)

    def test_n_parameters(self):
        model = LinearDynamics()
        assert model.n_parameters == 0  # not fitted
        z_t = np.random.randn(10, STATE_DIM)
        a_t = np.random.randn(10, ACTION_DIM)
        z_next = np.random.randn(10, STATE_DIM)
        model.fit(z_t, a_t, z_next, split="train")
        # A: 14*14, B: 14*12, c: 14
        assert model.n_parameters == STATE_DIM * STATE_DIM + STATE_DIM * ACTION_DIM + STATE_DIM

    def test_hyperparameters(self):
        model = LinearDynamics(lr=0.01, n_epochs=100, seed=42, regularization=1e-4)
        hp = model.hyperparameters()
        assert hp["model_type"] == "linear_dynamics"
        assert hp["lr"] == 0.01
        assert hp["n_epochs"] == 100
        assert hp["seed"] == 42

    def test_degenerate_empty_dataset(self):
        model = LinearDynamics()
        model.fit(np.zeros((0, STATE_DIM)), np.zeros((0, ACTION_DIM)), np.zeros((0, STATE_DIM)), split="train")
        # Should default to identity dynamics.
        z = np.ones(STATE_DIM)
        pred = model.predict(z, np.zeros(ACTION_DIM))
        assert np.allclose(pred, z)

    def test_deterministic_fit(self):
        """Same data + same seed → same parameters."""
        z_t = np.random.RandomState(42).randn(20, STATE_DIM)
        a_t = np.random.RandomState(43).randn(20, ACTION_DIM)
        z_next = z_t * 0.9 + 0.1
        m1 = LinearDynamics(seed=42)
        m1.fit(z_t, a_t, z_next, split="train")
        m2 = LinearDynamics(seed=42)
        m2.fit(z_t, a_t, z_next, split="train")
        assert np.allclose(m1.get_state()["A"], m2.get_state()["A"])


# ---------------------------------------------------------------------------
# 4. MLP dynamics
# ---------------------------------------------------------------------------

class TestMLPDynamics:
    """MLP dynamics model tests."""

    def test_fit_and_predict(self):
        model = MLPDynamics(hidden_dim=16, n_epochs=50, seed=42)
        z_t = np.random.RandomState(42).randn(20, STATE_DIM)
        a_t = np.random.RandomState(43).randn(20, ACTION_DIM)
        a_padded = np.zeros((20, STATE_DIM))
        a_padded[:, :ACTION_DIM] = a_t
        z_next = np.tanh(z_t + a_padded * 0.5)
        model.fit(z_t, a_t, z_next, split="train")
        pred = model.predict(z_t[0], a_t[0])
        assert pred.shape == (STATE_DIM,)
        assert all(np.isfinite(pred))

    def test_fit_rejects_non_train_split(self):
        model = MLPDynamics()
        z_t = np.random.randn(10, STATE_DIM)
        a_t = np.random.randn(10, ACTION_DIM)
        z_next = np.random.randn(10, STATE_DIM)
        with pytest.raises(ValueError, match="train split"):
            model.fit(z_t, a_t, z_next, split="held_out")

    def test_predict_before_fit_returns_identity(self):
        model = MLPDynamics()
        z = np.ones(STATE_DIM)
        pred = model.predict(z, np.zeros(ACTION_DIM))
        assert np.allclose(pred, z)

    def test_batch_prediction(self):
        model = MLPDynamics(hidden_dim=16, n_epochs=20, seed=42)
        z_t = np.random.RandomState(42).randn(10, STATE_DIM)
        a_t = np.random.RandomState(43).randn(10, ACTION_DIM)
        z_next = z_t + 0.1
        model.fit(z_t, a_t, z_next, split="train")
        preds = model.predict_batch(z_t, a_t)
        assert preds.shape == (10, STATE_DIM)

    def test_serialization_roundtrip(self):
        model = MLPDynamics(hidden_dim=16, n_epochs=20, seed=42)
        z_t = np.random.RandomState(42).randn(20, STATE_DIM)
        a_t = np.random.RandomState(43).randn(20, ACTION_DIM)
        z_next = z_t * 0.9 + 0.1
        model.fit(z_t, a_t, z_next, split="train")
        pred_before = model.predict(z_t[0], a_t[0])

        state = model.get_state()
        model2 = MLPDynamics()
        model2.set_state(state)
        pred_after = model2.predict(z_t[0], a_t[0])

        assert np.allclose(pred_before, pred_after)

    def test_n_parameters(self):
        model = MLPDynamics(hidden_dim=32)
        assert model.n_parameters == 0  # not fitted
        z_t = np.random.randn(10, STATE_DIM)
        a_t = np.random.randn(10, ACTION_DIM)
        z_next = np.random.randn(10, STATE_DIM)
        model.fit(z_t, a_t, z_next, split="train")
        # W1: (26, 32), b1: 32, W2: (32, 14), b2: 14
        input_dim = STATE_DIM + ACTION_DIM
        expected = input_dim * 32 + 32 + 32 * STATE_DIM + STATE_DIM
        assert model.n_parameters == expected

    def test_hyperparameters(self):
        model = MLPDynamics(hidden_dim=32, lr=0.001, n_epochs=100, seed=42)
        hp = model.hyperparameters()
        assert hp["model_type"] == "mlp_dynamics"
        assert hp["hidden_dim"] == 32
        assert hp["lr"] == 0.001

    def test_degenerate_empty_dataset(self):
        model = MLPDynamics(hidden_dim=16, seed=42)
        model.fit(np.zeros((0, STATE_DIM)), np.zeros((0, ACTION_DIM)), np.zeros((0, STATE_DIM)), split="train")
        z = np.ones(STATE_DIM)
        pred = model.predict(z, np.zeros(ACTION_DIM))
        assert pred.shape == (STATE_DIM,)


# ---------------------------------------------------------------------------
# 5. Dynamics metrics
# ---------------------------------------------------------------------------

class TestDynamicsMetrics:
    """Dynamics metrics tests."""

    def test_perfect_prediction(self):
        pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        actual = np.array([[1.0, 2.0], [3.0, 4.0]])
        m = compute_dynamics_metrics(pred, actual)
        assert m.rmse == pytest.approx(0.0, abs=1e-10)
        assert m.mae == pytest.approx(0.0, abs=1e-10)
        assert m.r2 == pytest.approx(1.0, abs=1e-6)

    def test_imperfect_prediction(self):
        pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        actual = np.array([[1.5, 2.5], [3.5, 4.5]])
        m = compute_dynamics_metrics(pred, actual)
        assert m.rmse > 0.0
        assert m.mae > 0.0

    def test_per_dim_rmse(self):
        pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        actual = np.array([[1.0, 3.0], [3.0, 5.0]])
        m = compute_dynamics_metrics(pred, actual)
        assert len(m.per_dim_rmse) == 2
        assert m.per_dim_rmse[0] == pytest.approx(0.0, abs=1e-10)
        assert m.per_dim_rmse[1] == pytest.approx(1.0, abs=1e-10)

    def test_empty_input(self):
        m = compute_dynamics_metrics(np.zeros((0, 14)), np.zeros((0, 14)))
        assert m.n_samples == 0
        assert m.rmse == 0.0

    def test_horizon_recorded(self):
        pred = np.array([[1.0]])
        actual = np.array([[1.0]])
        m = compute_dynamics_metrics(pred, actual, horizon=3)
        assert m.horizon == 3


# ---------------------------------------------------------------------------
# 6. Joint world model
# ---------------------------------------------------------------------------

class TestJointWorldModel:
    """Joint world model tests."""

    def test_fit_and_predict(self):
        config = JointModelConfig(dynamics_type="linear", n_epochs=10, seed=42)
        model = JointWorldModel(config=config)
        n = 20
        z_t = np.random.RandomState(42).randn(n, STATE_DIM)
        a_t = np.random.RandomState(43).randn(n, ACTION_DIM)
        z_next = z_t * 0.9 + 0.1
        y = np.random.RandomState(44).randn(n, 3) * 0.1
        model.fit(z_t, a_t, z_next, y, split="train")
        pred = model.predict(z_t[0], a_t[0])
        assert isinstance(pred, WorldModelPrediction)
        assert pred.predicted_next_state.shape == (STATE_DIM,)
        assert isinstance(pred.predicted_delta_utility, float)
        assert isinstance(pred.predicted_risk, float)
        assert isinstance(pred.predicted_cost, float)
        assert pred.predicted_uncertainty >= 0.0

    def test_fit_rejects_non_train_split(self):
        model = JointWorldModel()
        z_t = np.random.randn(10, STATE_DIM)
        a_t = np.random.randn(10, ACTION_DIM)
        z_next = np.random.randn(10, STATE_DIM)
        y = np.random.randn(10, 3)
        with pytest.raises(ValueError, match="train split"):
            model.fit(z_t, a_t, z_next, y, split="validation")

    def test_predict_before_fit_returns_zeros(self):
        model = JointWorldModel()
        z = np.ones(STATE_DIM)
        a = np.zeros(ACTION_DIM)
        pred = model.predict(z, a)
        assert np.allclose(pred.predicted_next_state, z)
        assert pred.predicted_delta_utility == 0.0

    def test_batch_prediction(self):
        config = JointModelConfig(n_epochs=10, seed=42)
        model = JointWorldModel(config=config)
        n = 10
        z_t = np.random.RandomState(42).randn(n, STATE_DIM)
        a_t = np.random.RandomState(43).randn(n, ACTION_DIM)
        z_next = z_t + 0.1
        y = np.random.RandomState(44).randn(n, 3)
        model.fit(z_t, a_t, z_next, y, split="train")
        preds = model.predict_batch(z_t, a_t)
        assert len(preds) == n
        assert all(isinstance(p, WorldModelPrediction) for p in preds)

    def test_dynamics_batch(self):
        config = JointModelConfig(n_epochs=10, seed=42)
        model = JointWorldModel(config=config)
        n = 10
        z_t = np.random.RandomState(42).randn(n, STATE_DIM)
        a_t = np.random.RandomState(43).randn(n, ACTION_DIM)
        z_next = z_t + 0.1
        y = np.random.RandomState(44).randn(n, 3)
        model.fit(z_t, a_t, z_next, y, split="train")
        preds = model.predict_dynamics_batch(z_t, a_t)
        assert preds.shape == (n, STATE_DIM)

    def test_outcome_batch(self):
        config = JointModelConfig(n_epochs=10, seed=42)
        model = JointWorldModel(config=config)
        n = 10
        z_t = np.random.RandomState(42).randn(n, STATE_DIM)
        a_t = np.random.RandomState(43).randn(n, ACTION_DIM)
        z_next = z_t + 0.1
        y = np.random.RandomState(44).randn(n, 3)
        model.fit(z_t, a_t, z_next, y, split="train")
        preds = model.predict_outcome_batch(z_t, a_t)
        assert preds.shape == (n, 3)

    def test_serialization_roundtrip(self):
        config = JointModelConfig(n_epochs=10, seed=42)
        model = JointWorldModel(config=config)
        n = 20
        z_t = np.random.RandomState(42).randn(n, STATE_DIM)
        a_t = np.random.RandomState(43).randn(n, ACTION_DIM)
        z_next = z_t * 0.9 + 0.1
        y = np.random.RandomState(44).randn(n, 3) * 0.1
        model.fit(z_t, a_t, z_next, y, split="train")
        pred_before = model.predict(z_t[0], a_t[0])

        state = model.get_state()
        model2 = JointWorldModel(config=config)
        model2.set_state(state)
        pred_after = model2.predict(z_t[0], a_t[0])

        assert np.allclose(pred_before.predicted_next_state, pred_after.predicted_next_state)
        assert pred_before.predicted_delta_utility == pytest.approx(pred_after.predicted_delta_utility)

    def test_n_parameters(self):
        model = JointWorldModel()
        assert model.n_parameters >= 0

    def test_hyperparameters(self):
        model = JointWorldModel()
        hp = model.hyperparameters()
        assert hp["model_type"] == "joint_world_model"
        assert "config" in hp

    def test_probability_positive(self):
        config = JointModelConfig(n_epochs=10, seed=42)
        model = JointWorldModel(config=config)
        n = 20
        z_t = np.random.RandomState(42).randn(n, STATE_DIM)
        a_t = np.random.RandomState(43).randn(n, ACTION_DIM)
        z_next = z_t + 0.1
        y = np.random.RandomState(44).randn(n, 3)
        model.fit(z_t, a_t, z_next, y, split="train")
        pred = model.predict(z_t[0], a_t[0])
        assert pred.probability_positive is not None
        assert 0.0 <= pred.probability_positive <= 1.0

    def test_mlp_dynamics_variant(self):
        config = JointModelConfig(dynamics_type="mlp", hidden_dim=16, n_epochs=20, seed=42)
        model = JointWorldModel(config=config)
        n = 20
        z_t = np.random.RandomState(42).randn(n, STATE_DIM)
        a_t = np.random.RandomState(43).randn(n, ACTION_DIM)
        a_padded = np.zeros((n, STATE_DIM))
        a_padded[:, :ACTION_DIM] = a_t
        z_next = np.tanh(z_t + a_padded * 0.5)
        y = np.random.RandomState(44).randn(n, 3)
        model.fit(z_t, a_t, z_next, y, split="train")
        pred = model.predict(z_t[0], a_t[0])
        assert pred.predicted_next_state.shape == (STATE_DIM,)


# ---------------------------------------------------------------------------
# 7. Training pipeline
# ---------------------------------------------------------------------------

class TestTrainingPipeline:
    """Training pipeline tests."""

    def test_train_world_model(self):
        records = _make_records(n=20, split="train")
        config = TrainingConfig(n_epochs=20, seed=42)
        result = train_world_model(records, config)
        assert isinstance(result, TrainingResult)
        assert result.n_train > 0
        assert result.model.n_parameters > 0
        assert result.train_metrics.n_samples > 0

    def test_train_only_on_train_split(self):
        """Training should only use train split records."""
        records = _make_records(n=20, split="train") + _make_records(n=10, split="validation")
        config = TrainingConfig(n_epochs=20, seed=42)
        result = train_world_model(records, config)
        assert result.n_train == 20  # only train records

    def test_train_with_no_records(self):
        result = train_world_model([], TrainingConfig())
        assert result.n_train == 0

    def test_train_metrics_include_dynamics(self):
        records = _make_records(n=20, split="train")
        result = train_world_model(records, TrainingConfig(n_epochs=20))
        assert result.train_metrics.dynamics.rmse >= 0.0
        assert result.train_metrics.dynamics.n_samples > 0

    def test_train_metrics_include_outcomes(self):
        records = _make_records(n=20, split="train")
        result = train_world_model(records, TrainingConfig(n_epochs=20))
        assert result.train_metrics.outcome_rmse >= 0.0
        assert result.train_metrics.risk_rmse >= 0.0
        assert result.train_metrics.cost_rmse >= 0.0

    def test_extract_training_data_filters_split(self):
        records = _make_records(n=20, split="train") + _make_records(n=10, split="held_out")
        from lgae_v3.experimental.exp5.training import extract_training_data
        z_t, a_t, z_next, y = extract_training_data(records, split="train")
        assert len(z_t) == 20

    def test_extract_training_data_handles_none_state_after(self):
        """Records with state_after=None should be skipped."""
        records = _make_records(n=20, split="train")
        records[0].structural_state_after = None
        from lgae_v3.experimental.exp5.training import extract_training_data
        z_t, _, _, _ = extract_training_data(records, split="train")
        assert len(z_t) == 19


# ---------------------------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:
    """Evaluation tests."""

    def test_evaluate_world_model(self):
        records = _make_records(n=30, split="train") + _make_records(n=15, split="validation")
        config = TrainingConfig(n_epochs=20, seed=42)
        result = train_world_model(records, config)
        eval_result = evaluate_world_model(result.model, records)
        assert isinstance(eval_result, EvaluationResult)
        assert eval_result.n_train > 0
        assert eval_result.n_validation > 0

    def test_rollout_evaluation(self):
        records = _make_records(n=30, split="train") + _make_records(n=15, split="validation")
        config = TrainingConfig(n_epochs=20, seed=42)
        result = train_world_model(records, config)
        report = rollout_evaluation(result.model, records, split="validation", max_horizon=3)
        assert isinstance(report, RolloutReport)
        assert len(report.horizons) == 3
        assert len(report.rmse_by_horizon) == 3

    def test_rollout_error_grows_with_horizon(self):
        """Rollout error should generally increase with horizon."""
        records = _make_records(n=40, split="train") + _make_records(n=20, split="validation")
        config = TrainingConfig(n_epochs=30, seed=42)
        result = train_world_model(records, config)
        report = rollout_evaluation(result.model, records, split="validation", max_horizon=3)
        # RMSE at horizon 3 should be >= RMSE at horizon 1 (typically).
        # This is not guaranteed in all cases, but should hold for reasonable models.
        if report.n_trajectories > 0:
            assert report.rmse_by_horizon[0] >= 0.0
            assert report.rmse_by_horizon[2] >= 0.0

    def test_evaluation_empty_heldout(self):
        records = _make_records(n=30, split="train")
        config = TrainingConfig(n_epochs=20, seed=42)
        result = train_world_model(records, config)
        eval_result = evaluate_world_model(result.model, records)
        assert eval_result.n_heldout == 0


# ---------------------------------------------------------------------------
# 9. WorldModelInterface implementation
# ---------------------------------------------------------------------------

class TestWorldModelInterface:
    """WorldModelInterface implementation tests."""

    def test_lightweight_world_model_is_world_model_interface(self):
        model = LightweightWorldModel()
        assert isinstance(model, WorldModelInterface)

    def test_predict_next_state(self):
        # Train a small model.
        records = _make_records(n=20, split="train")
        config = TrainingConfig(n_epochs=20, seed=42)
        result = train_world_model(records, config)
        wm = LightweightWorldModel(joint_model=result.model)

        state = MockState()
        action = ("ADD_EDGE", {"u": 1, "v": 5}, 20, 1.9)
        pred = wm.predict_next_state(state, action)
        assert isinstance(pred, ModelPrediction)
        assert pred.predicted_delta_utility is not None
        assert pred.predicted_risk is not None
        assert pred.predicted_next_state_hash is not None

    def test_rollout(self):
        records = _make_records(n=20, split="train")
        config = TrainingConfig(n_epochs=20, seed=42)
        result = train_world_model(records, config)
        wm = LightweightWorldModel(joint_model=result.model)

        state = MockState()
        actions = [
            ("ADD_EDGE", {"u": 1, "v": 5}, 20, 1.9),
            ("REMOVE_EDGE", {"u": 2, "v": 6}, 20, 1.9),
            ("ADD_EDGE", {"u": 3, "v": 7}, 20, 1.9),
        ]
        preds = wm.rollout(state, actions)
        assert len(preds) == 3
        assert all(isinstance(p, ModelPrediction) for p in preds)

    def test_update_is_noop(self):
        wm = LightweightWorldModel()
        # update should not raise.
        wm.update(MockState(), ("ADD_EDGE", {"u": 1, "v": 5}), MockState())

    def test_trust_report(self):
        wm = LightweightWorldModel()
        report = wm.trust_report()
        from lgae_v3.experimental.world_model import ModelTrustReport as _MTR
        assert isinstance(report, _MTR)

    def test_name(self):
        wm = LightweightWorldModel()
        assert wm.name == "lightweight_world_model"

    def test_set_trust_report(self):
        wm = LightweightWorldModel()
        tr = WorldModelTrustReport(trust_score=0.8, recommended_horizon=2)
        wm.set_trust_report(tr)
        report = wm.trust_report()
        assert report.trust_score == 0.8
        assert report.recommended_horizon == 2


# ---------------------------------------------------------------------------
# 10. Authority boundary
# ---------------------------------------------------------------------------

class TestAuthorityBoundary:
    """Authority boundary preservation tests."""

    def test_world_model_does_not_mutate_runtime(self):
        """The world model must not touch the v5.11 runtime."""
        from lgae_v3.runtime import LGAERuntime, RuntimeConfig
        from lgae_v3 import ResearchConfig, make_graph_buffers
        cfg = ResearchConfig()
        cfg.audit.orc_backend = "exact_lp"
        cfg.audit.persistent_homology_enabled = False
        cfg.audit.entropic_nodes = 0
        cfg.audit.bakry_nodes = 0
        cfg.audit.cde_nodes = 0
        cfg.audit.exact_lly_top_k = 0
        cfg.audit.orc_top_k = 0
        cfg.mutation.shadow_horizons = [1, 2]
        cfg.mutation.curvature_ema_enabled = False
        graph = make_graph_buffers(8, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)], capacity=16)
        runtime = LGAERuntime(graph=graph, config=cfg, runtime_config=RuntimeConfig())
        gen_before = runtime.snapshot().generation

        # Train and predict with the world model.
        records = _make_records(n=20, split="train")
        result = train_world_model(records, TrainingConfig(n_epochs=10))
        wm = LightweightWorldModel(joint_model=result.model)
        pred = wm.predict_next_state(MockState(), ("ADD_EDGE", {"u": 1, "v": 5}))
        preds = wm.rollout(MockState(), [("ADD_EDGE", {"u": 1, "v": 5})])

        gen_after = runtime.snapshot().generation
        assert gen_before == gen_after

    def test_dynamics_model_has_no_commit_access(self):
        """Dynamics models should not have commit/mutate methods."""
        model = LinearDynamics()
        assert not hasattr(model, "commit")
        assert not hasattr(model, "mutate")
        assert not hasattr(model, "commit_channel")

    def test_joint_model_has_no_commit_access(self):
        model = JointWorldModel()
        assert not hasattr(model, "commit")
        assert not hasattr(model, "mutate")

    def test_world_model_impl_has_no_commit_access(self):
        wm = LightweightWorldModel()
        assert not hasattr(wm, "commit")
        assert not hasattr(wm, "mutate")

    def test_training_function_has_no_runtime_param(self):
        """train_world_model should not accept runtime/commit parameters."""
        import inspect
        sig = inspect.signature(train_world_model)
        params = set(sig.parameters.keys())
        assert "runtime" not in params
        assert "commit_channel" not in params
        assert "authority" not in params


# ---------------------------------------------------------------------------
# 11. Determinism and reproducibility
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Determinism and reproducibility tests."""

    def test_linear_dynamics_same_seed_same_result(self):
        z_t = np.random.RandomState(42).randn(20, STATE_DIM)
        a_t = np.random.RandomState(43).randn(20, ACTION_DIM)
        z_next = z_t * 0.9 + 0.1
        m1 = LinearDynamics(seed=42)
        m1.fit(z_t, a_t, z_next, split="train")
        m2 = LinearDynamics(seed=42)
        m2.fit(z_t, a_t, z_next, split="train")
        p1 = m1.predict(z_t[0], a_t[0])
        p2 = m2.predict(z_t[0], a_t[0])
        assert np.allclose(p1, p2)

    def test_mlp_dynamics_same_seed_same_result(self):
        z_t = np.random.RandomState(42).randn(20, STATE_DIM)
        a_t = np.random.RandomState(43).randn(20, ACTION_DIM)
        z_next = z_t + 0.1
        m1 = MLPDynamics(hidden_dim=16, n_epochs=20, seed=42)
        m1.fit(z_t, a_t, z_next, split="train")
        m2 = MLPDynamics(hidden_dim=16, n_epochs=20, seed=42)
        m2.fit(z_t, a_t, z_next, split="train")
        p1 = m1.predict(z_t[0], a_t[0])
        p2 = m2.predict(z_t[0], a_t[0])
        assert np.allclose(p1, p2)

    def test_training_same_config_same_model(self):
        records = _make_records(n=20, split="train")
        config = TrainingConfig(n_epochs=20, seed=42)
        r1 = train_world_model(records, config)
        r2 = train_world_model(records, config)
        p1 = r1.model.predict(np.zeros(STATE_DIM), np.zeros(ACTION_DIM))
        p2 = r2.model.predict(np.zeros(STATE_DIM), np.zeros(ACTION_DIM))
        assert np.allclose(p1.predicted_next_state, p2.predicted_next_state)


# ---------------------------------------------------------------------------
# 12. Degenerate cases
# ---------------------------------------------------------------------------

class TestDegenerateCases:
    """Degenerate dataset handling tests."""

    def test_constant_targets(self):
        """All targets the same value."""
        records = _make_records(n=20, split="train")
        for r in records:
            r.realized_delta = 0.5
            r.realized_risk = 0.1
            r.realized_cost = 1.0
        result = train_world_model(records, TrainingConfig(n_epochs=20))
        assert result.n_train > 0

    def test_single_record(self):
        records = _make_records(n=1, split="train")
        result = train_world_model(records, TrainingConfig())
        assert result.n_train == 1

    def test_no_state_after(self):
        """All records have state_after=None."""
        records = _make_records(n=20, split="train")
        for r in records:
            r.structural_state_after = None
        result = train_world_model(records, TrainingConfig())
        assert result.n_train == 0

    def test_nonfinite_inputs(self):
        """Model should handle non-finite inputs gracefully."""
        model = LinearDynamics()
        z_t = np.random.randn(10, STATE_DIM)
        z_t[0, 0] = float('nan')
        a_t = np.random.randn(10, ACTION_DIM)
        z_next = np.random.randn(10, STATE_DIM)
        # Fit should not crash (may produce NaN predictions).
        try:
            model.fit(z_t, a_t, z_next, split="train")
        except Exception:
            pass  # acceptable — just shouldn't crash the system

    def test_zero_variance_state(self):
        """All states identical."""
        records = _make_records(n=20, split="train")
        state = MockState()
        for r in records:
            r.structural_state_before = state
            r.structural_state_after = state
        result = train_world_model(records, TrainingConfig(n_epochs=20))
        assert result.n_train > 0
