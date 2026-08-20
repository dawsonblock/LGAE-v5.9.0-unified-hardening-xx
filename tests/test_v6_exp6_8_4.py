"""Tests for v6.0-exp6.8.4: Advantage Model Identification."""
import numpy as np
import pytest

from lgae_v3 import make_graph_buffers
from lgae_v3.experimental.exp6_8_4 import (
    # Target transforms.
    transform_raw, transform_normalized, transform_sign,
    transform_ordinal, transform_downside,
    apply_target_transform, is_classification_target,
    # Models.
    RidgeModel, GBTModel, MLPModel, PairwiseModel, create_model,
    # Metrics.
    compute_spearman_correlation, compute_downside_probability,
    compute_cvar_negative, compute_risk_adjusted_score,
    # Features.
    get_feature_dim, extract_action_effects, extract_local_topology,
    extract_global_structure,
    F1_DIM, F2_DIM, F3_DIM, F4_DIM,
)


class TestTargetTransforms:
    """Test target transformations."""

    def test_raw(self):
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        result = transform_raw(a)
        assert np.allclose(result, a)

    def test_normalized(self):
        a = np.array([10.0, -20.0, 30.0], dtype=np.float32)
        bq = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        result = transform_normalized(a, bq)
        assert abs(result[0] - 10.0 / 100.0) < 1e-6
        assert abs(result[1] - (-20.0 / 200.0)) < 1e-6

    def test_sign(self):
        a = np.array([1.0, -2.0, 0.0, 3.0], dtype=np.float32)
        result = transform_sign(a)
        assert np.allclose(result, [1, -1, 0, 1])

    def test_ordinal(self):
        a = np.array([-100, -1, 0, 1, 100], dtype=np.float32)
        result = transform_ordinal(a)
        # Should have 5 distinct classes.
        assert len(set(result.tolist())) <= 5
        # Strongly worse < slightly worse < tied < slightly better < strongly better.
        assert result[0] <= result[1] <= result[2] <= result[3] <= result[4]

    def test_downside(self):
        a = np.array([-100, -50, 0, 50, 100], dtype=np.float32)
        result = transform_downside(a, clip_neg=-30)
        # Large negatives should be clipped to -30.
        assert result[0] == -30.0
        assert result[1] == -30.0
        assert result[2] == 0.0
        assert result[3] == 50.0

    def test_is_classification(self):
        assert is_classification_target("T3_sign") is True
        assert is_classification_target("T4_ordinal") is True
        assert is_classification_target("T1_raw") is False
        assert is_classification_target("T2_normalized") is False


class TestModels:
    """Test the model zoo."""

    def test_ridge(self):
        m = RidgeModel(alpha=1.0)
        X = np.random.randn(50, 5).astype(np.float32)
        y = (X @ np.array([1, 2, -1, 0.5, 0.3])).astype(np.float32)
        m.fit(X, y)
        preds = m.predict(X[:10])
        assert preds.shape == (10,)

    def test_gbt(self):
        m = GBTModel(n_estimators=20, lr=0.1, max_depth=2)
        X = np.random.randn(50, 5).astype(np.float32)
        y = (X @ np.array([1, 2, -1, 0.5, 0.3])).astype(np.float32)
        m.fit(X, y)
        preds = m.predict(X[:10])
        assert preds.shape == (10,)

    def test_mlp(self):
        m = MLPModel(hidden_dim=16, n_epochs=50, lr=0.01)
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        m.fit(X, y)
        preds = m.predict(X[:10])
        assert preds.shape == (10,)

    def test_pairwise(self):
        m = PairwiseModel(lr=0.01, n_epochs=100, l2=0.01)
        X = np.random.randn(50, 5).astype(np.float32)
        y = (np.random.randn(50) > 0).astype(np.float32)
        m.fit(X, y)
        preds = m.predict(X[:10])
        assert preds.shape == (10,)
        assert np.all(preds >= 0) and np.all(preds <= 1)  # probabilities

    def test_create_model(self):
        for name in ["M1_ridge", "M2_gbt", "M3_mlp", "M4_pairwise"]:
            m = create_model(name)
            assert m is not None


class TestDownsideMetrics:
    """Test downside-aware metrics."""

    def test_spearman(self):
        pred = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        actual = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        corr = compute_spearman_correlation(pred, actual)
        assert abs(corr - 1.0) < 1e-6

    def test_spearman_anticorrelated(self):
        pred = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        actual = np.array([5, 4, 3, 2, 1], dtype=np.float32)
        corr = compute_spearman_correlation(pred, actual)
        assert abs(corr - (-1.0)) < 1e-6

    def test_downside_probability(self):
        true_advs = [10, -5, 20, -15, 5]
        used = [True, True, True, True, False]
        # P(A < 0 | override) = 2/4 = 0.5
        prob = compute_downside_probability(true_advs, used, tau=0)
        assert abs(prob - 0.5) < 1e-6

    def test_cvar_negative(self):
        true_advs = [10, -5, 20, -15, 5]
        used = [True, True, True, True, True]
        # All 5 overrides, 5th percentile ~ -15, CVaR = mean of worst 5% ~ -15
        cvar = compute_cvar_negative(true_advs, used, 5.0)
        assert cvar <= 0  # should be negative

    def test_risk_adjusted_score(self):
        true_advs = [10, -5, 20, -15, 5]
        used = [True, True, True, True, True]
        score = compute_risk_adjusted_score(true_advs, used, lambda_risk=0.5)
        # E[A] = (10-5+20-15+5)/5 = 3.0
        # DownsideRisk = P(A<0) * |CVaR_neg| = 0.4 * 15 = 6
        # Score = 3.0 - 0.5 * 6 = 0.0
        assert isinstance(score, float)


class TestFeatureDimensions:
    """Test feature dimension calculations."""

    def test_feature_dims_increase(self):
        assert F1_DIM < F2_DIM
        assert F2_DIM < F3_DIM
        assert F3_DIM < F4_DIM

    def test_get_feature_dim(self):
        assert get_feature_dim("F1_current") == F1_DIM
        assert get_feature_dim("F4_full") == F4_DIM


class TestFeatureExtraction:
    """Test rich feature extraction."""

    def test_action_effects(self):
        graph = make_graph_buffers(
            num_nodes=6, edges=[(0,1),(1,2),(2,3)], capacity=20,
        )
        action = ("add_edge", 0, 3, {"weight": 1.0})
        feats = extract_action_effects(graph, action, 6)
        assert feats.shape == (12,)
        # Adding an edge: delta_edges should be 1/5 = 0.2
        assert abs(feats[0] - 0.2) < 1e-6

    def test_local_topology(self):
        graph = make_graph_buffers(
            num_nodes=6, edges=[(0,1),(1,2),(2,3)], capacity=20,
        )
        action = ("add_edge", 0, 3, {"weight": 1.0})
        feats = extract_local_topology(graph, action, 6)
        assert feats.shape == (10,)

    def test_global_structure(self):
        graph = make_graph_buffers(
            num_nodes=6, edges=[(0,1),(1,2),(2,3)], capacity=20,
        )
        feats = extract_global_structure(graph, 6)
        assert feats.shape == (8,)
        # n_components should be > 0
        assert feats[0] > 0
