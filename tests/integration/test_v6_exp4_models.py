"""v6.0-exp4: Outcome, risk, and cost models tests.

Tests cover:
1. Model determinism
2. Dataset/encoder compatibility
3. Train-only fitting
4. Held-out protection
5. Serialization roundtrip
6. Prediction finite
7. Uncertainty finite
8. Calibration reproducibility
9. Pairwise ranking correctness
10. Group metric generation
11. Counterfactual-vs-real provenance
12. OOD evaluation
13. Artifact hash stability
14. Model registry
15. Authority boundary untouched
16. Degenerate dataset handling
"""
from __future__ import annotations

import pytest
import numpy as np
import json
import math

from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.experimental.models import (
    # Protocol
    Prediction, ClassificationPrediction, RankingPrediction,
    ModelLifecycle, config_hash,
    # Targets
    TargetType, TargetDefinition, TargetSet, DEFAULT_TARGETS,
    compute_sign_delta, compute_normalized_delta, compute_utility_bucket,
    compute_candidate_ranks, compute_pairwise_labels,
    aggregate_risk, aggregate_cost,
    # Baselines
    GlobalMeanPredictor, MutationTypeMeanPredictor, NearestExperiencePredictor,
    # Linear
    LinearRegressionPredictor, RidgeRegressionPredictor, LogisticRegressionPredictor,
    # Tree
    GradientBoostedTreePredictor,
    # MLP
    MLPRegressor, MLPClassifier,
    # Ranking
    PointwiseRankingModel, PairwiseRankingModel,
    # Uncertainty
    UncertaintyReport, BootstrapEnsemble, analyze_uncertainty,
    # Calibration
    CalibrationReport, ReliabilityCurve,
    expected_calibration_error, brier_score, reliability_curve,
    prediction_interval_coverage, standardized_residual_calibration,
    calibration_drift,
    # Evaluator
    RegressionMetrics, ClassificationMetrics, RankingMetrics,
    GroupMetrics, CFToRealGap,
    compute_regression_metrics, compute_classification_metrics,
    compute_ranking_metrics, compute_group_metrics,
    compute_cf_to_real_gap, compute_ood_degradation,
    # Artifact
    ModelArtifact, CompatibilityError, create_artifact,
    # Registry
    ModelRegistry,
)
from lgae_v3.experimental.encoders import (
    MinimalControlEncoder, GlobalStateEncoder,
    EncoderRegistry, EncoderProvenance,
)
from lgae_v3.experimental import (
    extract_global_features, extract_local_action_features,
)


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
    """Generate a simple synthetic dataset for model testing."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    # Target: linear + noise.
    w = rng.randn(d) * 0.5
    y = X @ w + rng.randn(n) * 0.1
    return X, y


def _make_classification_dataset(n=50, d=24, seed=42):
    """Generate a binary classification dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    logits = X[:, 0] * 2 + X[:, 1] * 1.5
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(float)
    return X, y


# ---------------------------------------------------------------------------
# 1. Protocol and prediction contract
# ---------------------------------------------------------------------------

class TestProtocol:
    """Protocol and prediction contract tests."""

    def test_prediction_creation(self):
        p = Prediction(mean=1.0, uncertainty=0.5, model_id="test")
        assert p.mean == 1.0
        assert p.uncertainty == 0.5
        assert p.model_id == "test"

    def test_prediction_with_interval(self):
        p = Prediction(mean=1.0, uncertainty=0.5, model_id="test", lower=0.0, upper=2.0)
        assert p.lower == 0.0
        assert p.upper == 2.0

    def test_prediction_to_log(self):
        p = Prediction(mean=1.0, uncertainty=0.5, model_id="test")
        log = p.to_log()
        assert log["mean"] == 1.0
        assert log["model_id"] == "test"

    def test_classification_prediction(self):
        p = ClassificationPrediction(probability=0.7, predicted_class=1, uncertainty=0.4, model_id="test")
        assert p.probability == 0.7
        assert p.predicted_class == 1

    def test_ranking_prediction(self):
        p = RankingPrediction(scores=(0.5, 0.9, 0.1), ranked_indices=(1, 0, 2), model_id="test")
        assert p.ranked_indices[0] == 1  # highest score first

    def test_config_hash_deterministic(self):
        h1 = config_hash({"a": 1, "b": 2})
        h2 = config_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_config_hash_differs(self):
        h1 = config_hash({"a": 1})
        h2 = config_hash({"a": 2})
        assert h1 != h2


# ---------------------------------------------------------------------------
# 2. Target transforms
# ---------------------------------------------------------------------------

class TestTargets:
    """Target definition and transform tests."""

    def test_compute_sign_delta(self):
        assert compute_sign_delta(0.1) == 1
        assert compute_sign_delta(-0.1) == 0
        assert compute_sign_delta(0.0) == 0

    def test_compute_normalized_delta(self):
        result = compute_normalized_delta(0.5, 2.0)
        assert abs(result - 0.25) < 1e-6

    def test_compute_normalized_delta_zero_utility(self):
        result = compute_normalized_delta(0.5, 0.0)
        assert abs(result - 0.5 / 1e-6) < 1.0  # should be large but finite

    def test_compute_utility_bucket(self):
        assert compute_utility_bucket(-0.5) == 0  # strongly negative
        assert compute_utility_bucket(0.0) == 2   # neutral
        assert compute_utility_bucket(0.5) == 4   # strongly positive

    def test_compute_candidate_ranks(self):
        deltas = [0.1, 0.5, 0.3]
        ranks = compute_candidate_ranks(deltas)
        assert ranks[1] == 0  # 0.5 is highest → rank 0
        assert ranks[2] == 1  # 0.3 is second → rank 1
        assert ranks[0] == 2  # 0.1 is lowest → rank 2

    def test_compute_pairwise_labels(self):
        deltas = [0.1, 0.5, 0.3]
        pairs, labels = compute_pairwise_labels(deltas)
        assert len(pairs) == 3  # 3 pairs from 3 candidates
        assert len(labels) == 3
        # Pair (0, 1): 0.1 vs 0.5 → 0 (0.1 < 0.5)
        assert labels[0] == 0

    def test_aggregate_risk(self):
        components = {"instability": 0.3, "ood": 0.2}
        result = aggregate_risk(components)
        assert 0.0 < result < 1.0

    def test_aggregate_cost(self):
        components = {"wall_clock": 1.0, "shadow_executions": 2.0}
        result = aggregate_cost(components)
        assert result > 0.0

    def test_target_set(self):
        ts = TargetSet(
            regression_targets=("realized_delta",),
            classification_targets=("sign_delta",),
        )
        assert "realized_delta" in ts.all_targets
        assert "sign_delta" in ts.all_targets


# ---------------------------------------------------------------------------
# 3. Baseline predictors
# ---------------------------------------------------------------------------

class TestBaselines:
    """Baseline predictor tests."""

    def test_global_mean_fit_predict(self):
        X, y = _make_dataset(n=20)
        model = GlobalMeanPredictor()
        model.fit(X, y, split="train")
        preds = model.predict(X)
        assert len(preds) == 20
        assert all(p.mean == pytest.approx(np.mean(y)) for p in preds)

    def test_global_mean_freeze(self):
        X, y = _make_dataset(n=20)
        model = GlobalMeanPredictor()
        model.fit(X, y, split="train")
        model.freeze()
        assert model.lifecycle == "frozen"

    def test_global_mean_rejects_heldout(self):
        X, y = _make_dataset(n=20)
        model = GlobalMeanPredictor()
        with pytest.raises(ValueError):
            model.fit(X, y, split="held_out")

    def test_global_mean_rejects_refit_after_freeze(self):
        X, y = _make_dataset(n=20)
        model = GlobalMeanPredictor()
        model.fit(X, y, split="train")
        model.freeze()
        with pytest.raises(RuntimeError):
            model.fit(X, y, split="train")

    def test_mutation_type_mean_fit_predict(self):
        X, y = _make_dataset(n=30, d=32)
        # Add action one-hot at the end.
        action_types = np.zeros((30, 8))
        for i in range(30):
            action_types[i, i % 8] = 1.0
        X = np.hstack([X, action_types])
        model = MutationTypeMeanPredictor(n_action_types=8, action_offset=24)
        model.fit(X, y, split="train")
        preds = model.predict(X)
        assert len(preds) == 30

    def test_nearest_experience_fit_predict(self):
        X, y = _make_dataset(n=20)
        model = NearestExperiencePredictor()
        model.fit(X, y, split="train")
        preds = model.predict(X[:5])
        assert len(preds) == 5
        # Nearest neighbor of itself should return exact target.
        assert preds[0].mean == pytest.approx(y[0])


# ---------------------------------------------------------------------------
# 4. Linear models
# ---------------------------------------------------------------------------

class TestLinearModels:
    """Linear and ridge regression tests."""

    def test_linear_fit_predict(self):
        X, y = _make_dataset(n=30)
        model = LinearRegressionPredictor(n_epochs=100)
        model.fit(X, y, split="train")
        preds = model.predict(X)
        assert len(preds) == 30
        assert all(math.isfinite(p.mean) for p in preds)
        assert all(math.isfinite(p.uncertainty) for p in preds)

    def test_linear_deterministic(self):
        X, y = _make_dataset(n=30)
        m1 = LinearRegressionPredictor(seed=42, n_epochs=50)
        m2 = LinearRegressionPredictor(seed=42, n_epochs=50)
        m1.fit(X, y, split="train")
        m2.fit(X, y, split="train")
        p1 = m1.predict(X[:5])
        p2 = m2.predict(X[:5])
        for a, b in zip(p1, p2):
            assert a.mean == pytest.approx(b.mean, abs=1e-6)

    def test_linear_rejects_heldout(self):
        X, y = _make_dataset(n=20)
        model = LinearRegressionPredictor()
        with pytest.raises(ValueError):
            model.fit(X, y, split="held_out")

    def test_linear_freeze(self):
        X, y = _make_dataset(n=20)
        model = LinearRegressionPredictor()
        model.fit(X, y, split="train")
        model.freeze()
        assert model.lifecycle == "frozen"
        with pytest.raises(RuntimeError):
            model.fit(X, y, split="train")

    def test_ridge_fit_predict(self):
        X, y = _make_dataset(n=30)
        model = RidgeRegressionPredictor(alpha=1.0, n_epochs=100)
        model.fit(X, y, split="train")
        preds = model.predict(X)
        assert len(preds) == 30
        assert all(math.isfinite(p.mean) for p in preds)

    def test_logistic_fit_predict(self):
        X, y = _make_classification_dataset(n=30)
        model = LogisticRegressionPredictor(n_epochs=100)
        model.fit(X, y, split="train")
        preds = model.predict_proba(X)
        assert len(preds) == 30
        assert all(0 <= p.probability <= 1 for p in preds)


# ---------------------------------------------------------------------------
# 5. Tree model
# ---------------------------------------------------------------------------

class TestTreeModel:
    """Gradient-boosted tree tests."""

    def test_tree_fit_predict(self):
        X, y = _make_dataset(n=30, d=10)
        model = GradientBoostedTreePredictor(n_estimators=20, seed=42)
        model.fit(X, y, split="train")
        preds = model.predict(X)
        assert len(preds) == 30
        assert all(math.isfinite(p.mean) for p in preds)
        assert all(p.lower is not None for p in preds)
        assert all(p.upper is not None for p in preds)

    def test_tree_deterministic(self):
        X, y = _make_dataset(n=30, d=10)
        m1 = GradientBoostedTreePredictor(seed=42, n_estimators=10)
        m2 = GradientBoostedTreePredictor(seed=42, n_estimators=10)
        m1.fit(X, y, split="train")
        m2.fit(X, y, split="train")
        p1 = m1.predict(X[:5])
        p2 = m2.predict(X[:5])
        for a, b in zip(p1, p2):
            assert a.mean == pytest.approx(b.mean, abs=1e-6)

    def test_tree_rejects_heldout(self):
        X, y = _make_dataset(n=20)
        model = GradientBoostedTreePredictor()
        with pytest.raises(ValueError):
            model.fit(X, y, split="held_out")


# ---------------------------------------------------------------------------
# 6. MLP models
# ---------------------------------------------------------------------------

class TestMLPModels:
    """MLP regressor and classifier tests."""

    def test_mlp_regressor_fit_predict(self):
        X, y = _make_dataset(n=30, d=12)
        model = MLPRegressor(hidden_dim=16, n_ensemble=3, n_epochs=30, seed=42)
        model.fit(X, y, split="train")
        preds = model.predict(X)
        assert len(preds) == 30
        assert all(math.isfinite(p.mean) for p in preds)
        assert all(math.isfinite(p.uncertainty) for p in preds)

    def test_mlp_regressor_ensemble_uncertainty(self):
        X, y = _make_dataset(n=30, d=12)
        model = MLPRegressor(n_ensemble=5, n_epochs=30, seed=42)
        model.fit(X, y, split="train")
        preds = model.predict(X)
        # Ensemble should produce non-zero uncertainty.
        assert any(p.uncertainty > 0 for p in preds)

    def test_mlp_regressor_freeze(self):
        X, y = _make_dataset(n=30, d=12)
        model = MLPRegressor(n_ensemble=2, n_epochs=10, seed=42)
        model.fit(X, y, split="train")
        model.freeze()
        assert model.lifecycle == "frozen"

    def test_mlp_classifier_fit_predict(self):
        X, y = _make_classification_dataset(n=30, d=12)
        model = MLPClassifier(hidden_dim=16, n_ensemble=3, n_epochs=30, seed=42)
        model.fit(X, y, split="train")
        preds = model.predict_proba(X)
        assert len(preds) == 30
        assert all(0 <= p.probability <= 1 for p in preds)

    def test_mlp_rejects_heldout(self):
        X, y = _make_dataset(n=30, d=12)
        model = MLPRegressor(n_ensemble=2, n_epochs=10, seed=42)
        with pytest.raises(ValueError):
            model.fit(X, y, split="held_out")


# ---------------------------------------------------------------------------
# 7. Ranking models
# ---------------------------------------------------------------------------

class TestRankingModels:
    """Ranking model tests."""

    def test_pointwise_rank_fit_rank(self):
        X, y = _make_dataset(n=20, d=12)
        model = PointwiseRankingModel(n_epochs=100)
        model.fit(X, y, split="train")
        result = model.rank(X[:5])
        assert len(result.scores) == 5
        assert len(result.ranked_indices) == 5
        # Ranked indices should be in descending score order.
        for i in range(len(result.ranked_indices) - 1):
            assert result.scores[result.ranked_indices[i]] >= result.scores[result.ranked_indices[i + 1]]

    def test_pairwise_rank_fit_rank(self):
        X, y = _make_dataset(n=20, d=12)
        model = PairwiseRankingModel(n_epochs=50)
        model.fit(X, y, split="train")
        result = model.rank(X[:5])
        assert len(result.scores) == 5
        assert len(result.ranked_indices) == 5

    def test_pairwise_ranking_correctness(self):
        """Pairwise ranking should correctly order candidates."""
        # Create a dataset where higher feature 0 → higher target.
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5)
        y = X[:, 0] * 2  # clear signal
        model = PairwiseRankingModel(n_epochs=100, lr=0.1)
        model.fit(X, y, split="train")
        # Test on 5 candidates.
        test_X = np.array([[2.0, 0, 0, 0, 0], [-2.0, 0, 0, 0, 0],
                           [1.0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0],
                           [0.5, 0, 0, 0, 0]])
        result = model.rank(test_X)
        # Highest feature 0 (index 0) should be ranked first.
        assert result.ranked_indices[0] == 0  # feature 0 = 2.0
        assert result.ranked_indices[1] == 2  # feature 0 = 1.0


# ---------------------------------------------------------------------------
# 8. Uncertainty
# ---------------------------------------------------------------------------

class TestUncertainty:
    """Uncertainty estimation tests."""

    def test_analyze_uncertainty(self):
        preds = [
            Prediction(mean=1.0, uncertainty=0.5, model_id="test"),
            Prediction(mean=2.0, uncertainty=1.0, model_id="test"),
        ]
        report = analyze_uncertainty(preds, method="ensemble")
        assert report.n_predictions == 2
        assert report.mean_uncertainty == 0.75

    def test_bootstrap_ensemble(self):
        X, y = _make_dataset(n=30, d=10)
        from lgae_v3.experimental.models import LinearRegressionPredictor
        ensemble = BootstrapEnsemble(
            base_model_factory=lambda: LinearRegressionPredictor(n_epochs=50),
            n_bootstrap=5,
        )
        ensemble.fit(X, y, split="train")
        preds = ensemble.predict(X[:5])
        assert len(preds) == 5
        assert all(math.isfinite(p.uncertainty) for p in preds)


# ---------------------------------------------------------------------------
# 9. Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    """Calibration measurement tests."""

    def test_ece_perfect_calibration(self):
        # Perfect calibration: probabilities match labels.
        probs = [0.0, 0.0, 1.0, 1.0]
        labels = [0, 0, 1, 1]
        report = expected_calibration_error(probs, labels, n_bins=5)
        assert report.value < 0.01  # near zero

    def test_ece_worst_calibration(self):
        # Worst calibration: all predict 1.0 but half are 0.
        probs = [1.0, 1.0, 1.0, 1.0]
        labels = [0, 0, 1, 1]
        report = expected_calibration_error(probs, labels, n_bins=5)
        assert report.value > 0.4  # high ECE

    def test_brier_score(self):
        probs = [0.5, 0.5, 0.5, 0.5]
        labels = [0, 0, 1, 1]
        report = brier_score(probs, labels)
        # Brier = mean((0.5-0)^2 + (0.5-0)^2 + (0.5-1)^2 + (0.5-1)^2) = 0.25
        assert abs(report.value - 0.25) < 1e-6

    def test_reliability_curve(self):
        probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [0, 0, 1, 1, 1]
        curve = reliability_curve(probs, labels, n_bins=5)
        assert len(curve.bin_centers) == 5

    def test_prediction_interval_coverage(self):
        means = [0.0, 0.0, 0.0, 0.0]
        uncertainties = [1.0, 1.0, 1.0, 1.0]
        targets = [0.5, -0.5, 2.0, -2.0]  # 2 out of 4 within ±1.96
        report = prediction_interval_coverage(means, uncertainties, targets, z=1.96)
        assert 0.0 < report.value <= 1.0

    def test_standardized_residual_calibration(self):
        # Well-calibrated: residuals ~ N(0, 1).
        means = [0.0] * 100
        uncertainties = [1.0] * 100
        rng = np.random.RandomState(42)
        targets = rng.randn(100).tolist()  # N(0, 1)
        report = standardized_residual_calibration(means, uncertainties, targets)
        assert report.value < 0.3  # close to N(0, 1)

    def test_calibration_drift(self):
        drift = calibration_drift(0.05, 0.15)
        assert abs(drift - 0.1) < 1e-6  # degradation of 0.1


# ---------------------------------------------------------------------------
# 10. Evaluator
# ---------------------------------------------------------------------------

class TestEvaluator:
    """Evaluator metric tests."""

    def test_regression_metrics(self):
        preds = [Prediction(mean=1.0, uncertainty=0.5, model_id="test") for _ in range(10)]
        targets = [1.0 + 0.1 * i for i in range(10)]
        metrics = compute_regression_metrics(preds, targets)
        assert metrics.n_samples == 10
        assert metrics.rmse > 0
        assert metrics.mae > 0

    def test_classification_metrics(self):
        preds = [ClassificationPrediction(probability=0.7, predicted_class=1, uncertainty=0.4, model_id="test") for _ in range(10)]
        labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        metrics = compute_classification_metrics(preds, labels)
        assert metrics.n_samples == 10
        assert 0 <= metrics.accuracy <= 1

    def test_ranking_metrics(self):
        predicted = [0, 1, 2, 3]  # predicted order
        true = [0, 1, 2, 3]       # perfect order
        metrics = compute_ranking_metrics(predicted, true, k=3)
        assert metrics.top1_agreement == 1.0
        assert metrics.pairwise_accuracy == 1.0

    def test_ranking_metrics_imperfect(self):
        predicted = [2, 0, 1, 3]
        true = [0, 1, 2, 3]
        metrics = compute_ranking_metrics(predicted, true, k=3)
        assert metrics.top1_agreement == 0.0  # predicted top-1 is index 1, true is index 0

    def test_group_metrics(self):
        preds = [Prediction(mean=1.0, uncertainty=0.5, model_id="test") for _ in range(10)]
        targets = [1.0 + 0.1 * i for i in range(10)]
        groups = ["path"] * 5 + ["cycle"] * 5
        gm = compute_group_metrics(preds, targets, groups, "graph_family")
        assert len(gm.group_values) == 2
        assert len(gm.metrics) == 2

    def test_cf_to_real_gap(self):
        real = RegressionMetrics(rmse=0.1, mae=0.05, r2=0.8, spearman=0.9, kendall_tau=0.8, n_samples=50)
        cf = RegressionMetrics(rmse=0.2, mae=0.1, r2=0.6, spearman=0.7, kendall_tau=0.6, n_samples=50)
        gap = compute_cf_to_real_gap(real, cf, metric="spearman")
        assert abs(gap.gap - 0.2) < 1e-6  # 0.9 - 0.7

    def test_ood_degradation(self):
        id_metrics = RegressionMetrics(rmse=0.1, mae=0.05, r2=0.8, spearman=0.9, kendall_tau=0.8, n_samples=50)
        ood_metrics = RegressionMetrics(rmse=0.3, mae=0.15, r2=0.4, spearman=0.5, kendall_tau=0.4, n_samples=50)
        deg = compute_ood_degradation(id_metrics, ood_metrics, metric="spearman")
        assert deg == 0.4  # 0.9 - 0.5


# ---------------------------------------------------------------------------
# 11. Model artifact
# ---------------------------------------------------------------------------

class TestModelArtifact:
    """Model artifact and provenance tests."""

    def test_artifact_creation(self):
        artifact = ModelArtifact(
            model_id="test-001",
            predictor_type="linear",
            predictor_version="v1",
            encoder_id="global",
            encoder_schema_hash="enc_hash",
            dataset_schema_hash="ds_hash",
            train_split_hash="train_hash",
            normalization_hash="norm_hash",
            hyperparameter_hash="hp_hash",
            seed=42,
            training_code_version="v6.0-exp4",
            n_train_samples=100,
            n_features=24,
        )
        assert artifact.model_id == "test-001"
        assert artifact.artifact_hash

    def test_artifact_hash_deterministic(self):
        a1 = ModelArtifact(
            model_id="test", predictor_type="linear", predictor_version="v1",
            encoder_id="global", encoder_schema_hash="h1", dataset_schema_hash="h2",
            train_split_hash="h3", normalization_hash="h4", hyperparameter_hash="h5",
            seed=42, training_code_version="v6", n_train_samples=10, n_features=5,
        )
        a2 = ModelArtifact(
            model_id="test", predictor_type="linear", predictor_version="v1",
            encoder_id="global", encoder_schema_hash="h1", dataset_schema_hash="h2",
            train_split_hash="h3", normalization_hash="h4", hyperparameter_hash="h5",
            seed=42, training_code_version="v6", n_train_samples=10, n_features=5,
        )
        assert a1.artifact_hash == a2.artifact_hash

    def test_artifact_compatibility(self):
        artifact = ModelArtifact(
            model_id="test", predictor_type="linear", predictor_version="v1",
            encoder_id="global", encoder_schema_hash="enc_hash",
            dataset_schema_hash="ds_hash",
            train_split_hash="", normalization_hash="", hyperparameter_hash="",
            seed=42, training_code_version="v6", n_train_samples=10, n_features=5,
        )
        assert artifact.is_compatible_with("enc_hash", "ds_hash")
        assert not artifact.is_compatible_with("wrong", "ds_hash")

    def test_artifact_to_log(self):
        artifact = ModelArtifact(
            model_id="test", predictor_type="linear", predictor_version="v1",
            encoder_id="global", encoder_schema_hash="h1", dataset_schema_hash="h2",
            train_split_hash="h3", normalization_hash="h4", hyperparameter_hash="h5",
            seed=42, training_code_version="v6", n_train_samples=10, n_features=5,
        )
        log = artifact.to_log()
        assert log["model_id"] == "test"
        assert "artifact_hash" in log

    def test_create_artifact_from_model(self):
        X, y = _make_dataset(n=20)
        model = LinearRegressionPredictor(n_epochs=50)
        model.fit(X, y, split="train")
        artifact = create_artifact(
            model,
            encoder_id="global",
            encoder_schema_hash="enc_hash",
            dataset_schema_hash="ds_hash",
        )
        assert artifact.predictor_type == "linear"
        assert artifact.n_features == 24


# ---------------------------------------------------------------------------
# 12. Model registry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    """Model registry tests."""

    def test_create_linear(self):
        model = ModelRegistry.create("linear")
        assert model.model_type == "linear"

    def test_create_tree(self):
        model = ModelRegistry.create("tree")
        assert model.model_type == "tree"

    def test_create_mlp(self):
        model = ModelRegistry.create("mlp")
        assert model.model_type == "mlp"

    def test_create_unknown_raises(self):
        with pytest.raises(KeyError):
            ModelRegistry.create("nonexistent")

    def test_available_models(self):
        names = ModelRegistry.available_models()
        assert "linear" in names
        assert "tree" in names
        assert "mlp" in names
        assert "global_mean" in names

    def test_all_model_info(self):
        all_info = ModelRegistry.all_model_info()
        assert len(all_info) >= 10

    def test_register_and_verify(self):
        X, y = _make_dataset(n=20)
        model = LinearRegressionPredictor(n_epochs=50)
        model.fit(X, y, split="train")
        artifact = ModelRegistry.register(
            model,
            encoder_id="global",
            encoder_schema_hash="enc_hash",
            dataset_schema_hash="ds_hash",
        )
        assert artifact.artifact_hash
        # Verify compatibility.
        ModelRegistry.verify_compatibility(
            artifact,
            encoder_schema_hash="enc_hash",
            dataset_schema_hash="ds_hash",
        )

    def test_verify_compatibility_fails(self):
        artifact = ModelArtifact(
            model_id="test", predictor_type="linear", predictor_version="v1",
            encoder_id="global", encoder_schema_hash="enc_hash",
            dataset_schema_hash="ds_hash",
            train_split_hash="", normalization_hash="", hyperparameter_hash="",
            seed=42, training_code_version="v6", n_train_samples=10, n_features=5,
        )
        with pytest.raises(CompatibilityError):
            ModelRegistry.verify_compatibility(
                artifact,
                encoder_schema_hash="wrong",
                dataset_schema_hash="ds_hash",
            )


# ---------------------------------------------------------------------------
# 13. Degenerate dataset handling
# ---------------------------------------------------------------------------

class TestDegenerateDatasets:
    """Models should fail explicitly or degrade gracefully on degenerate data."""

    def test_all_positive_labels(self):
        X, y = _make_dataset(n=20)
        y = np.ones(20)
        model = LinearRegressionPredictor(n_epochs=20)
        model.fit(X, y, split="train")
        preds = model.predict(X[:5])
        assert all(math.isfinite(p.mean) for p in preds)

    def test_constant_target(self):
        X, y = _make_dataset(n=20)
        y = np.full(20, 0.5)
        model = LinearRegressionPredictor(n_epochs=20)
        model.fit(X, y, split="train")
        preds = model.predict(X[:5])
        # Should predict near 0.5.
        assert all(abs(p.mean - 0.5) < 0.5 for p in preds)

    def test_tiny_dataset(self):
        X, y = _make_dataset(n=3)
        model = LinearRegressionPredictor(n_epochs=20)
        model.fit(X, y, split="train")
        preds = model.predict(X)
        assert len(preds) == 3
        assert all(math.isfinite(p.mean) for p in preds)

    def test_zero_variance_feature(self):
        X, y = _make_dataset(n=20, d=5)
        X[:, 2] = 0.0  # zero variance feature
        model = LinearRegressionPredictor(n_epochs=20)
        model.fit(X, y, split="train")
        preds = model.predict(X[:5])
        assert all(math.isfinite(p.mean) for p in preds)

    def test_empty_candidate_set_ranking(self):
        model = PointwiseRankingModel(n_epochs=20)
        X = np.zeros((0, 5))
        y = np.zeros(0)
        model.fit(X, y, split="train")
        result = model.rank(X)
        assert len(result.scores) == 0

    def test_nonfinite_input(self):
        X = np.array([[1.0, float("nan"), 3.0], [float("inf"), 2.0, 3.0]])
        y = np.array([1.0, 2.0])
        model = LinearRegressionPredictor(n_epochs=20)
        model.fit(X, y, split="train")
        preds = model.predict(X)
        assert all(math.isfinite(p.mean) for p in preds)


# ---------------------------------------------------------------------------
# 14. Authority boundary
# ---------------------------------------------------------------------------

class TestAuthorityBoundaryExp4:
    """v5.11 authority boundary untouched by models."""

    def test_models_do_not_touch_runtime(self):
        runtime = LGAERuntime(graph=_small_graph(), config=_cfg(), runtime_config=RuntimeConfig())
        gen_before = runtime.snapshot().generation
        # Train and predict with a model.
        X, y = _make_dataset(n=20)
        model = LinearRegressionPredictor(n_epochs=20)
        model.fit(X, y, split="train")
        preds = model.predict(X[:5])
        assert len(preds) == 5
        # Runtime state should be unchanged.
        gen_after = runtime.snapshot().generation
        assert gen_before == gen_after

    def test_models_are_advisory_only(self):
        """Models produce predictions but have no mutation authority."""
        X, y = _make_dataset(n=20)
        model = MLPRegressor(n_ensemble=2, n_epochs=10, seed=42)
        model.fit(X, y, split="train")
        preds = model.predict(X[:5])
        # Predictions are just data — no side effects.
        for p in preds:
            assert isinstance(p, Prediction)
            assert isinstance(p.mean, float)


# ---------------------------------------------------------------------------
# 15. Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    """Serialization roundtrip tests."""

    def test_prediction_to_log_json(self):
        p = Prediction(mean=1.0, uncertainty=0.5, model_id="test", lower=0.5, upper=1.5)
        log = p.to_log()
        data = json.dumps(log, sort_keys=True)
        parsed = json.loads(data)
        assert parsed["mean"] == 1.0
        assert parsed["lower"] == 0.5

    def test_artifact_to_json(self):
        artifact = ModelArtifact(
            model_id="test", predictor_type="linear", predictor_version="v1",
            encoder_id="global", encoder_schema_hash="h1", dataset_schema_hash="h2",
            train_split_hash="h3", normalization_hash="h4", hyperparameter_hash="h5",
            seed=42, training_code_version="v6", n_train_samples=10, n_features=5,
        )
        data = artifact.to_json()
        parsed = json.loads(data)
        assert parsed["model_id"] == "test"
        assert "artifact_hash" in parsed

    def test_metrics_to_log(self):
        metrics = RegressionMetrics(rmse=0.1, mae=0.05, r2=0.8, spearman=0.9, kendall_tau=0.8, n_samples=50)
        log = metrics.to_log()
        data = json.dumps(log, sort_keys=True)
        parsed = json.loads(data)
        assert parsed["rmse"] == 0.1
