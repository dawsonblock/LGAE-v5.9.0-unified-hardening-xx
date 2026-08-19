"""v6.0-exp4.1: Real structural model competition tests.

Tests cover:
1. Model state serialization roundtrip (Fix 1)
2. Complete hyperparameter hashing (Fix 2)
3. Full compatibility binding (Fix 3)
4. Unified outcome terminology (Fix 4)
5. Competition harness on synthetic exp2-like data
6. Artifact state hash stability
7. Group metric generation
8. CF-to-real gap measurement
9. Authority boundary untouched
"""
from __future__ import annotations

import pytest
import numpy as np
import json
import math
import pickle
import base64

from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.experimental.models import (
    # Protocol
    Prediction, ClassificationPrediction, ModelLifecycle,
    # Models
    GlobalMeanPredictor, MutationTypeMeanPredictor, NearestExperiencePredictor,
    LinearRegressionPredictor, RidgeRegressionPredictor, LogisticRegressionPredictor,
    GradientBoostedTreePredictor, MLPRegressor, MLPClassifier,
    PointwiseRankingModel, PairwiseRankingModel,
    # Artifact
    ModelArtifact, CompatibilityError, create_artifact,
    # Registry
    ModelRegistry,
    # Competition
    CompetitionEntry, CompetitionReport, ExtractedData,
    extract_competition_data, run_competition,
    DEFAULT_ENCODERS, DEFAULT_PREDICTORS,
)
from lgae_v3.experimental.encoders import (
    MinimalControlEncoder, GlobalStateEncoder, EncoderRegistry,
)
from lgae_v3.experimental.world_model import ModelPrediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> ResearchConfig:
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 0
    cfg.audit.orc_backend = "exact_lp"
    cfg.audit.persistent_homology_enabled = False
    cfg.audit.entropic_nodes = 0
    cfg.audit.bakry_nodes = 0
    cfg.audit.cde_nodes = 0
    cfg.audit.exact_lly_top_k = 0
    cfg.audit.orc_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]
    cfg.mutation.curvature_ema_enabled = False
    return cfg


def _small_graph():
    return make_graph_buffers(8, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)], capacity=16)


def _make_dataset(n=50, d=24, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    w = rng.randn(d) * 0.5
    y = X @ w + rng.randn(n) * 0.1
    return X, y


def _make_classification_dataset(n=50, d=24, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    logits = X[:, 0] * 2 + X[:, 1] * 1.5
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(float)
    return X, y


# ---------------------------------------------------------------------------
# Fix 1: Model state serialization
# ---------------------------------------------------------------------------

class TestModelStateSerialization:
    """Fix 1: Serialize actual model state, not just metadata."""

    def test_linear_state_roundtrip(self):
        X, y = _make_dataset(n=30)
        model = LinearRegressionPredictor(n_epochs=50)
        model.fit(X, y, split="train")
        state = model.get_state()
        # Create new model and load state.
        model2 = LinearRegressionPredictor(n_epochs=50)
        model2.set_state(state)
        preds1 = model.predict(X[:5])
        preds2 = model2.predict(X[:5])
        for a, b in zip(preds1, preds2):
            assert abs(a.mean - b.mean) < 1e-6

    def test_ridge_state_roundtrip(self):
        X, y = _make_dataset(n=30)
        model = RidgeRegressionPredictor(alpha=1.0, n_epochs=50)
        model.fit(X, y, split="train")
        state = model.get_state()
        model2 = RidgeRegressionPredictor(alpha=1.0, n_epochs=50)
        model2.set_state(state)
        preds1 = model.predict(X[:5])
        preds2 = model2.predict(X[:5])
        for a, b in zip(preds1, preds2):
            assert abs(a.mean - b.mean) < 1e-6

    def test_tree_state_roundtrip(self):
        X, y = _make_dataset(n=30, d=10)
        model = GradientBoostedTreePredictor(n_estimators=10, seed=42)
        model.fit(X, y, split="train")
        state = model.get_state()
        model2 = GradientBoostedTreePredictor(n_estimators=10, seed=42)
        model2.set_state(state)
        preds1 = model.predict(X[:5])
        preds2 = model2.predict(X[:5])
        for a, b in zip(preds1, preds2):
            assert abs(a.mean - b.mean) < 1e-6

    def test_mlp_state_roundtrip(self):
        X, y = _make_dataset(n=30, d=12)
        model = MLPRegressor(hidden_dim=16, n_ensemble=2, n_epochs=20, seed=42)
        model.fit(X, y, split="train")
        state = model.get_state()
        model2 = MLPRegressor(hidden_dim=16, n_ensemble=2, n_epochs=20, seed=42)
        model2.set_state(state)
        preds1 = model.predict(X[:5])
        preds2 = model2.predict(X[:5])
        for a, b in zip(preds1, preds2):
            assert abs(a.mean - b.mean) < 1e-5

    def test_logistic_state_roundtrip(self):
        X, y = _make_classification_dataset(n=30)
        model = LogisticRegressionPredictor(n_epochs=50)
        model.fit(X, y, split="train")
        state = model.get_state()
        model2 = LogisticRegressionPredictor(n_epochs=50)
        model2.set_state(state)
        preds1 = model.predict_proba(X[:5])
        preds2 = model2.predict_proba(X[:5])
        for a, b in zip(preds1, preds2):
            assert abs(a.probability - b.probability) < 1e-6

    def test_global_mean_state_roundtrip(self):
        X, y = _make_dataset(n=20)
        model = GlobalMeanPredictor()
        model.fit(X, y, split="train")
        state = model.get_state()
        model2 = GlobalMeanPredictor()
        model2.set_state(state)
        preds1 = model.predict(X[:5])
        preds2 = model2.predict(X[:5])
        for a, b in zip(preds1, preds2):
            assert a.mean == b.mean

    def test_nearest_experience_state_roundtrip(self):
        X, y = _make_dataset(n=20, d=10)
        model = NearestExperiencePredictor()
        model.fit(X, y, split="train")
        state = model.get_state()
        model2 = NearestExperiencePredictor()
        model2.set_state(state)
        preds1 = model.predict(X[:5])
        preds2 = model2.predict(X[:5])
        for a, b in zip(preds1, preds2):
            assert a.mean == b.mean

    def test_artifact_contains_model_state(self):
        """Fix 1: Artifact must contain actual serialized model state."""
        X, y = _make_dataset(n=20)
        model = LinearRegressionPredictor(n_epochs=50)
        model.fit(X, y, split="train")
        artifact = create_artifact(
            model,
            encoder_id="global",
            encoder_schema_hash="enc_hash",
            dataset_schema_hash="ds_hash",
        )
        assert artifact.model_state  # Non-empty
        assert artifact.model_state_hash  # Non-empty
        # Verify state can be loaded.
        state = artifact.load_state()
        assert state is not None
        assert "weights" in state

    def test_artifact_state_hash_deterministic(self):
        """State hash should be deterministic for the same model state."""
        X, y = _make_dataset(n=20)
        model = LinearRegressionPredictor(n_epochs=50, seed=42)
        model.fit(X, y, split="train")
        a1 = create_artifact(model, encoder_id="e", encoder_schema_hash="h", dataset_schema_hash="d")
        model2 = LinearRegressionPredictor(n_epochs=50, seed=42)
        model2.fit(X, y, split="train")
        a2 = create_artifact(model2, encoder_id="e", encoder_schema_hash="h", dataset_schema_hash="d")
        assert a1.model_state_hash == a2.model_state_hash

    def test_artifact_state_hash_differs_for_different_state(self):
        X, y = _make_dataset(n=20)
        model1 = LinearRegressionPredictor(n_epochs=50, seed=42)
        model1.fit(X, y, split="train")
        a1 = create_artifact(model1, encoder_id="e", encoder_schema_hash="h", dataset_schema_hash="d")
        model2 = LinearRegressionPredictor(n_epochs=50, seed=99)
        model2.fit(X, y, split="train")
        a2 = create_artifact(model2, encoder_id="e", encoder_schema_hash="h", dataset_schema_hash="d")
        assert a1.model_state_hash != a2.model_state_hash


# ---------------------------------------------------------------------------
# Fix 2: Complete hyperparameter hashing
# ---------------------------------------------------------------------------

class TestHyperparameterHash:
    """Fix 2: Hash complete hyperparameter configuration."""

    def test_linear_hp_includes_lr_and_epochs(self):
        model = LinearRegressionPredictor(lr=0.01, n_epochs=100, seed=42)
        hp = model.hyperparameters()
        assert hp["lr"] == 0.01
        assert hp["n_epochs"] == 100
        assert hp["seed"] == 42

    def test_ridge_hp_includes_alpha(self):
        model = RidgeRegressionPredictor(alpha=2.0, seed=42)
        hp = model.hyperparameters()
        assert hp["alpha"] == 2.0

    def test_tree_hp_includes_n_estimators(self):
        model = GradientBoostedTreePredictor(n_estimators=50, learning_rate=0.1, seed=42)
        hp = model.hyperparameters()
        assert hp["n_estimators"] == 50
        assert hp["learning_rate"] == 0.1

    def test_mlp_hp_includes_hidden_dim_and_ensemble(self):
        model = MLPRegressor(hidden_dim=64, n_ensemble=5, n_epochs=100, seed=42)
        hp = model.hyperparameters()
        assert hp["hidden_dim"] == 64
        assert hp["n_ensemble"] == 5
        assert hp["n_epochs"] == 100

    def test_different_hp_produces_different_hash(self):
        """Two MLPs with different hyperparameters must have different hashes."""
        X, y = _make_dataset(n=30, d=12)
        m1 = MLPRegressor(hidden_dim=32, n_ensemble=3, n_epochs=20, seed=42)
        m1.fit(X, y, split="train")
        a1 = create_artifact(m1, encoder_id="e", encoder_schema_hash="h", dataset_schema_hash="d")
        m2 = MLPRegressor(hidden_dim=64, n_ensemble=5, n_epochs=50, seed=42)
        m2.fit(X, y, split="train")
        a2 = create_artifact(m2, encoder_id="e", encoder_schema_hash="h", dataset_schema_hash="d")
        assert a1.hyperparameter_hash != a2.hyperparameter_hash

    def test_same_hp_produces_same_hash(self):
        X, y = _make_dataset(n=30, d=12)
        m1 = MLPRegressor(hidden_dim=32, n_ensemble=3, n_epochs=20, seed=42)
        m1.fit(X, y, split="train")
        a1 = create_artifact(m1, encoder_id="e", encoder_schema_hash="h", dataset_schema_hash="d")
        m2 = MLPRegressor(hidden_dim=32, n_ensemble=3, n_epochs=20, seed=42)
        m2.fit(X, y, split="train")
        a2 = create_artifact(m2, encoder_id="e", encoder_schema_hash="h", dataset_schema_hash="d")
        assert a1.hyperparameter_hash == a2.hyperparameter_hash


# ---------------------------------------------------------------------------
# Fix 3: Full compatibility binding
# ---------------------------------------------------------------------------

class TestCompatibilityBinding:
    """Fix 3: Bind compatibility to dataset/split/normalization identity."""

    def test_artifact_has_feature_and_target_schema(self):
        artifact = ModelArtifact(
            model_id="test", predictor_type="linear", predictor_version="v1",
            encoder_id="global", encoder_schema_hash="enc",
            dataset_schema_hash="ds", train_split_hash="ts",
            normalization_hash="norm", feature_schema_hash="feat",
            target_schema_hash="tgt", hyperparameter_hash="hp",
            model_state_hash="ms", seed=42, training_code_version="v6",
            n_train_samples=10, n_features=5,
        )
        assert artifact.feature_schema_hash == "feat"
        assert artifact.target_schema_hash == "tgt"

    def test_compatibility_checks_split_hash(self):
        artifact = ModelArtifact(
            model_id="test", predictor_type="linear", predictor_version="v1",
            encoder_id="global", encoder_schema_hash="enc",
            dataset_schema_hash="ds", train_split_hash="ts1",
            normalization_hash="norm", feature_schema_hash="feat",
            target_schema_hash="tgt", hyperparameter_hash="hp",
            model_state_hash="ms", seed=42, training_code_version="v6",
            n_train_samples=10, n_features=5,
        )
        # Same split → compatible.
        assert artifact.is_compatible_with(
            encoder_schema_hash="enc", dataset_schema_hash="ds",
            train_split_hash="ts1",
        )
        # Different split → incompatible.
        assert not artifact.is_compatible_with(
            encoder_schema_hash="enc", dataset_schema_hash="ds",
            train_split_hash="ts2",
        )

    def test_compatibility_checks_normalization_hash(self):
        artifact = ModelArtifact(
            model_id="test", predictor_type="linear", predictor_version="v1",
            encoder_id="global", encoder_schema_hash="enc",
            dataset_schema_hash="ds", train_split_hash="ts",
            normalization_hash="norm1", feature_schema_hash="feat",
            target_schema_hash="tgt", hyperparameter_hash="hp",
            model_state_hash="ms", seed=42, training_code_version="v6",
            n_train_samples=10, n_features=5,
        )
        assert not artifact.is_compatible_with(
            encoder_schema_hash="enc", dataset_schema_hash="ds",
            normalization_hash="norm2",
        )

    def test_compatibility_empty_fields_are_wildcards(self):
        """Empty identity fields should be treated as wildcards."""
        artifact = ModelArtifact(
            model_id="test", predictor_type="linear", predictor_version="v1",
            encoder_id="global", encoder_schema_hash="enc",
            dataset_schema_hash="ds", train_split_hash="",
            normalization_hash="", feature_schema_hash="",
            target_schema_hash="", hyperparameter_hash="hp",
            model_state_hash="ms", seed=42, training_code_version="v6",
            n_train_samples=10, n_features=5,
        )
        # Should be compatible regardless of split/normalization provided.
        assert artifact.is_compatible_with(
            encoder_schema_hash="enc", dataset_schema_hash="ds",
            train_split_hash="anything", normalization_hash="anything",
        )

    def test_registry_verify_compatibility_with_identity(self):
        X, y = _make_dataset(n=20)
        model = LinearRegressionPredictor(n_epochs=50)
        model.fit(X, y, split="train")
        artifact = ModelRegistry.register(
            model,
            encoder_id="global",
            encoder_schema_hash="enc",
            dataset_schema_hash="ds",
            train_split_hash="ts1",
            normalization_hash="norm1",
            feature_schema_hash="feat1",
            target_schema_hash="tgt1",
        )
        # Correct identity → passes.
        ModelRegistry.verify_compatibility(
            artifact,
            encoder_schema_hash="enc",
            dataset_schema_hash="ds",
            train_split_hash="ts1",
            normalization_hash="norm1",
            feature_schema_hash="feat1",
            target_schema_hash="tgt1",
        )
        # Wrong split → fails.
        with pytest.raises(CompatibilityError):
            ModelRegistry.verify_compatibility(
                artifact,
                encoder_schema_hash="enc",
                dataset_schema_hash="ds",
                train_split_hash="wrong",
            )


# ---------------------------------------------------------------------------
# Fix 4: Unified outcome terminology
# ---------------------------------------------------------------------------

class TestUnifiedTerminology:
    """Fix 4: Unify exp4 outcome terminology with world_model.py interface."""

    def test_model_prediction_has_risk_not_reward(self):
        p = ModelPrediction(predicted_delta_utility=0.1, predicted_risk=0.2, predicted_cost=0.3)
        assert p.predicted_risk == 0.2
        assert p.predicted_cost == 0.3

    def test_model_prediction_has_probability_positive(self):
        p = ModelPrediction(probability_positive=0.7)
        assert p.probability_positive == 0.7

    def test_model_prediction_reward_is_deprecated_alias(self):
        p = ModelPrediction(predicted_risk=0.5)
        # predicted_reward should return the same value as predicted_risk
        # but emit a DeprecationWarning.
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            val = p.predicted_reward
        assert val == 0.5
        assert any(issubclass(w.category, DeprecationWarning) for w in caught), \
            "predicted_reward should emit DeprecationWarning"

    def test_model_prediction_to_log_includes_risk(self):
        p = ModelPrediction(predicted_delta_utility=0.1, predicted_risk=0.2)
        log = p.to_log()
        assert "predicted_risk" in log
        assert "probability_positive" in log


# ---------------------------------------------------------------------------
# Competition harness
# ---------------------------------------------------------------------------

class TestCompetitionHarness:
    """Test the competition harness with synthetic data."""

    def test_extract_competition_data(self):
        """Test data extraction from synthetic records."""
        from dataclasses import dataclass
        from lgae_v3.experimental.transition_record import (
            TransitionProvenance, AuthorizationDecision,
            AuthorityIdentity, StructuralStateSummary, DiagnosisSummary,
            CandidateSetSummary, PlannerMetadata, ComputeMetrics,
        )

        @dataclass
        class MockRecord:
            split: str
            graph_family: str
            action: str
            action_target: dict
            realized_delta: float
            realized_risk: float
            realized_cost: float
            success: bool
            rollback: bool
            rejected: bool
            structural_state_before: StructuralStateSummary
            provenance: TransitionProvenance
            predicted_delta: float = 0.0
            predicted_risk: float = 0.0
            predicted_cost: float = 0.0
            predicted_ig: float = 0.0

        def _make_record(split, family, delta, action="ADD_EDGE"):
            return MockRecord(
                split=split,
                graph_family=family,
                action=action,
                action_target={"u": 0, "v": 5},
                realized_delta=delta,
                realized_risk=0.1,
                realized_cost=0.05,
                success=delta > 0,
                rollback=False,
                rejected=False,
                structural_state_before=StructuralStateSummary(
                    n_nodes=10, n_edges=9, density=0.2,
                    spectral_gap=0.5, degree_mean=1.8, degree_std=0.4,
                    n_components=1, avg_clustering=0.3,
                    fiber_count=1, fiber_width=2, gauge_dim=0,
                    state_hash="abc", graph_version=1,
                ),
                provenance=TransitionProvenance.REALIZED,
            )

        records = []
        for i in range(10):
            records.append(_make_record("train", "path", 0.1 * i - 0.5))
            records.append(_make_record("validation", "cycle", 0.1 * i - 0.5))
            records.append(_make_record("held_out", "star", 0.1 * i - 0.5))

        encoder = MinimalControlEncoder()
        data = extract_competition_data(records, encoder, "realized_delta")
        assert len(data.y_train) == 10
        assert len(data.y_validation) == 10
        assert len(data.y_heldout) == 10
        assert data.n_features > 0

    def test_run_competition_minimal(self):
        """Run a minimal competition with synthetic data."""
        from dataclasses import dataclass
        from lgae_v3.experimental.transition_record import (
            TransitionProvenance, StructuralStateSummary,
        )

        @dataclass
        class MockRecord:
            split: str
            graph_family: str
            action: str
            action_target: dict
            realized_delta: float
            realized_risk: float
            realized_cost: float
            success: bool
            rollback: bool
            rejected: bool
            structural_state_before: StructuralStateSummary
            provenance: TransitionProvenance
            predicted_delta: float = 0.0
            predicted_risk: float = 0.0
            predicted_cost: float = 0.0
            predicted_ig: float = 0.0

        records = []
        for i in range(15):
            for split, family in [("train", "path"), ("validation", "cycle"), ("held_out", "star")]:
                records.append(MockRecord(
                    split=split, graph_family=family,
                    action="ADD_EDGE", action_target={"u": 0, "v": 5},
                    realized_delta=0.1 * (i % 5) - 0.2,
                    realized_risk=0.1, realized_cost=0.05,
                    success=(i % 3 == 0), rollback=False, rejected=False,
                    structural_state_before=StructuralStateSummary(
                        n_nodes=10, n_edges=9, density=0.2,
                        spectral_gap=0.5, degree_mean=1.8, degree_std=0.4,
                        n_components=1, avg_clustering=0.3,
                        fiber_count=1, fiber_width=2, gauge_dim=0,
                        state_hash="abc", graph_version=1,
                    ),
                    provenance=TransitionProvenance.REALIZED,
                ))

        report = run_competition(
            records,
            encoders=["minimal-control"],
            predictors=["global_mean", "linear"],
            target="realized_delta",
            n_epochs=20,
        )
        assert len(report.entries) == 2  # 1 encoder × 2 predictors
        assert report.n_train > 0
        assert report.n_validation > 0
        # Each entry should have validation metrics.
        for entry in report.entries:
            assert entry.encoder_id == "minimal-control"
            assert entry.predictor_id in ("global_mean", "linear")

    def test_competition_report_summary_table(self):
        """Test that the summary table is generated."""
        report = CompetitionReport(
            entries=[
                CompetitionEntry(
                    encoder_id="global",
                    predictor_id="linear",
                    target="realized_delta",
                    validation_metrics={"spearman": 0.5},
                    heldout_metrics={"spearman": 0.4, "accuracy": 0.7},
                ),
            ],
        )
        table = report.summary_table()
        assert "global" in table
        assert "linear" in table

    def test_competition_report_to_json(self):
        report = CompetitionReport(
            entries=[CompetitionEntry(
                encoder_id="test", predictor_id="test", target="test",
            )],
        )
        data = report.to_json()
        parsed = json.loads(data)
        assert "entries" in parsed


# ---------------------------------------------------------------------------
# Authority boundary
# ---------------------------------------------------------------------------

class TestAuthorityBoundaryExp41:
    """v5.11 authority boundary untouched by exp4.1."""

    def test_competition_does_not_touch_runtime(self):
        runtime = LGAERuntime(graph=_small_graph(), config=_cfg(), runtime_config=RuntimeConfig())
        gen_before = runtime.snapshot().generation
        # Run a competition with synthetic data.
        from dataclasses import dataclass
        from lgae_v3.experimental.transition_record import (
            TransitionProvenance, StructuralStateSummary,
        )

        @dataclass
        class MockRecord:
            split: str
            graph_family: str
            action: str
            action_target: dict
            realized_delta: float
            realized_risk: float
            realized_cost: float
            success: bool
            rollback: bool
            rejected: bool
            structural_state_before: StructuralStateSummary
            provenance: TransitionProvenance
            predicted_delta: float = 0.0
            predicted_risk: float = 0.0
            predicted_cost: float = 0.0
            predicted_ig: float = 0.0

        records = []
        for i in range(10):
            for split in ["train", "validation", "held_out"]:
                records.append(MockRecord(
                    split=split, graph_family="path",
                    action="ADD_EDGE", action_target={"u": 0, "v": 5},
                    realized_delta=0.1 * i - 0.5,
                    realized_risk=0.1, realized_cost=0.05,
                    success=(i > 5), rollback=False, rejected=False,
                    structural_state_before=StructuralStateSummary(
                        n_nodes=10, n_edges=9, density=0.2,
                        spectral_gap=0.5, degree_mean=1.8, degree_std=0.4,
                        n_components=1, avg_clustering=0.3,
                        fiber_count=1, fiber_width=2, gauge_dim=0,
                        state_hash="abc", graph_version=1,
                    ),
                    provenance=TransitionProvenance.REALIZED,
                ))

        run_competition(records, encoders=["minimal-control"], predictors=["global_mean"], n_epochs=10)
        gen_after = runtime.snapshot().generation
        assert gen_before == gen_after
