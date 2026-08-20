"""Tests for v6.0-exp6.8.3: Conformal Structural Advantage."""
import numpy as np
import pytest

from lgae_v3 import make_graph_buffers
from lgae_v3.experimental.exp6_8_3 import (
    # Advantage models.
    ZeroAdvantageModel, LinearRegressionModel, RidgeRegressionModel,
    MLPModel, BootstrapMLPEnsemble, QuantileMLPModel,
    # Conformal calibration.
    compute_conformal_quantile, calibrate_conformal, compute_lcb_advantage,
    select_operating_alpha,
    # Arbitrator.
    conformal_arbitrate, ConformalArbitrationResult,
    # Risk metrics.
    compute_override_precision, compute_false_override_rate,
    compute_override_coverage, compute_mean_override_advantage,
    compute_regret_metrics, compute_cvar, compute_normalized_regret,
    compute_bootstrap_ci, compute_confidence_decile_analysis,
    # Coverage analysis.
    compute_coverage_safety_curve, select_operating_point,
    # No-leakage.
    assert_no_future_oracle_leakage, assert_train_calibration_test_isolation,
    # OOD.
    compute_ood_scores, compute_ood_coverage_analysis,
    # Features.
    encode_action, extract_pairwise_features, FULL_FEATURE_DIM,
    ACTION_FEATURE_DIM, PAIRWISE_FEATURE_DIM,
)
from lgae_v3.experimental.exp6_3.exact_mpc import ActionIdentity


class TestAdvantageModels:
    """Test the advantage model ladder."""

    def test_zero_model(self):
        m = ZeroAdvantageModel()
        m.fit(np.random.randn(10, 5), np.random.randn(10))
        preds = m.predict(np.random.randn(5, 5))
        assert np.all(preds == 0)

    def test_linear_model(self):
        m = LinearRegressionModel()
        X = np.random.randn(50, 5).astype(np.float32)
        y = (X @ np.array([1, 2, -1, 0.5, 0.3]) + 0.5).astype(np.float32)
        m.fit(X, y)
        preds = m.predict(X[:10])
        assert preds.shape == (10,)

    def test_ridge_model(self):
        m = RidgeRegressionModel(alpha=1.0)
        X = np.random.randn(50, 5).astype(np.float32)
        y = (X @ np.array([1, 2, -1, 0.5, 0.3])).astype(np.float32)
        m.fit(X, y)
        preds = m.predict(X[:10])
        assert preds.shape == (10,)

    def test_mlp_model(self):
        m = MLPModel(hidden_dim=16, n_epochs=50, lr=0.01)
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        m.fit(X, y)
        preds = m.predict(X[:10])
        assert preds.shape == (10,)

    def test_bootstrap_ensemble(self):
        m = BootstrapMLPEnsemble(n_members=3, hidden_dim=16, n_epochs=50, lr=0.01)
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        m.fit(X, y)
        preds = m.predict(X[:10])
        stds = m.predict_std(X[:10])
        assert preds.shape == (10,)
        assert stds.shape == (10,)
        assert np.all(stds >= 0)

    def test_quantile_mlp(self):
        m = QuantileMLPModel(hidden_dim=16, n_epochs=50, lr=0.01)
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        m.fit(X, y)
        q05, q50, q95 = m.predict_quantiles(X[:10])
        assert q05.shape == (10,)
        assert q50.shape == (10,)
        assert q95.shape == (10,)


class TestConformalCalibration:
    """Test split-conformal calibration."""

    def test_conformal_quantile(self):
        residuals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        q = compute_conformal_quantile(residuals, alpha=0.10)
        # 90% quantile with finite-sample guarantee.
        assert q >= 8  # should be around 9

    def test_conformal_quantile_empty(self):
        q = compute_conformal_quantile(np.array([]), alpha=0.05)
        assert q == float("inf")

    def test_calibrate_conformal(self):
        y_cal = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        y_hat_cal = np.array([1.1, 1.9, 3.1, 3.9, 5.1], dtype=np.float32)
        quantiles = calibrate_conformal(y_cal, y_hat_cal, alphas=[0.20, 0.10])
        assert 0.20 in quantiles
        assert 0.10 in quantiles
        # Higher confidence (lower alpha) -> larger quantile.
        assert quantiles[0.10] >= quantiles[0.20]

    def test_lcb_advantage(self):
        lcb = compute_lcb_advantage(y_hat=5.0, conformal_quantile=3.0)
        assert lcb == 2.0

    def test_select_operating_alpha(self):
        cal_results = {
            0.20: {"override_precision": 0.80, "coverage": 0.30, "p95_regret": 10, "cvar95": 15},
            0.10: {"override_precision": 0.95, "coverage": 0.20, "p95_regret": 8, "cvar95": 12},
            0.05: {"override_precision": 1.0, "coverage": 0.10, "p95_regret": 5, "cvar95": 8},
            0.01: {"override_precision": 1.0, "coverage": 0.05, "p95_regret": 3, "cvar95": 5},
        }
        alpha, metrics = select_operating_alpha(
            cal_results, min_precision=0.95, min_coverage=0.10,
        )
        # Should select alpha=0.10 (precision=0.95, coverage=0.20) — highest
        # coverage alpha that satisfies precision >= 0.95 and coverage >= 0.10.
        assert alpha == 0.10


class TestConformalArbitrator:
    """Test the conformal advantage arbitrator."""

    def test_override_when_lcb_positive(self):
        baseline_action = ("add_edge", 0, 1, {"weight": 1.0})
        learned_action = ("add_edge", 0, 2, {"weight": 1.0})
        baseline_id = ActionIdentity.from_action(baseline_action)
        learned_id = ActionIdentity.from_action(learned_action)

        result = conformal_arbitrate(
            baseline_action, learned_action,
            baseline_id, learned_id,
            advantage_hat=5.0, conformal_quantile=3.0,
            alpha=0.05,
        )
        assert result.used_learned is True
        assert result.source == "learned"
        assert result.lcb_advantage == 2.0

    def test_fallback_when_lcb_negative(self):
        baseline_action = ("add_edge", 0, 1, {"weight": 1.0})
        learned_action = ("add_edge", 0, 2, {"weight": 1.0})
        baseline_id = ActionIdentity.from_action(baseline_action)
        learned_id = ActionIdentity.from_action(learned_action)

        result = conformal_arbitrate(
            baseline_action, learned_action,
            baseline_id, learned_id,
            advantage_hat=1.0, conformal_quantile=3.0,
            alpha=0.05,
        )
        assert result.used_learned is False
        assert result.source == "baseline"
        assert result.lcb_advantage == -2.0

    def test_fallback_when_lcb_zero(self):
        baseline_action = ("add_edge", 0, 1, {"weight": 1.0})
        learned_action = ("add_edge", 0, 2, {"weight": 1.0})
        baseline_id = ActionIdentity.from_action(baseline_action)
        learned_id = ActionIdentity.from_action(learned_action)

        result = conformal_arbitrate(
            baseline_action, learned_action,
            baseline_id, learned_id,
            advantage_hat=3.0, conformal_quantile=3.0,
            alpha=0.05,
        )
        # LCB = 0, not > 0, so fallback.
        assert result.used_learned is False
        assert result.source == "baseline"


class TestRiskMetrics:
    """Test risk metrics."""

    def test_override_precision(self):
        true_advs = [1.0, -0.5, 2.0, 0.5, -1.0]
        used = [True, True, True, False, False]
        # 2 of 3 overrides were beneficial.
        precision = compute_override_precision(true_advs, used)
        assert abs(precision - 2/3) < 1e-6

    def test_false_override_rate(self):
        true_advs = [1.0, -0.5, 2.0]
        used = [True, True, True]
        rate = compute_false_override_rate(true_advs, used)
        assert abs(rate - 1/3) < 1e-6

    def test_override_coverage(self):
        used = [True, False, True, False, True]
        coverage = compute_override_coverage(used)
        assert abs(coverage - 0.6) < 1e-6

    def test_mean_override_advantage(self):
        true_advs = [1.0, -0.5, 2.0, 0.5]
        used = [True, True, False, False]
        mean_adv = compute_mean_override_advantage(true_advs, used)
        assert abs(mean_adv - 0.25) < 1e-6  # (1.0 + -0.5) / 2

    def test_cvar(self):
        regrets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cvar95 = compute_cvar(regrets, 95)
        assert cvar95 >= 9.0

    def test_regret_metrics(self):
        regrets = np.array([0, 1, 2, 5, 10, 20])
        m = compute_regret_metrics(regrets)
        assert "mean" in m
        assert "median" in m
        assert "p95" in m
        assert "cvar95" in m
        assert m["cvar95"] >= m["median"]

    def test_normalized_regret(self):
        exact_vals = [10.0, 20.0]
        selected_vals = [5.0, 20.0]
        norm = compute_normalized_regret(exact_vals, selected_vals)
        assert abs(norm[0] - 0.5) < 1e-6  # |10-5|/10
        assert norm[1] == 0.0

    def test_bootstrap_ci(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, lo, hi = compute_bootstrap_ci(values)
        assert abs(mean - 3.0) < 1e-6
        assert lo <= mean <= hi

    def test_confidence_decile_analysis(self):
        # Higher LCB -> higher precision.
        lcbs = list(np.linspace(-10, 10, 100))
        true_advs = list(np.linspace(-5, 5, 100))  # correlated with lcb
        used = [lcb > 0 for lcb in lcbs]
        result = compute_confidence_decile_analysis(lcbs, true_advs, used)
        assert "deciles" in result
        assert "is_monotonic" in result
        assert len(result["deciles"]) <= 10


class TestCoverageAnalysis:
    """Test coverage-vs-safety analysis."""

    def test_coverage_safety_curve(self):
        true_advs = [1.0, -0.5, 2.0, 0.5, -1.0, 3.0]
        regrets = [0.0, 0.5, 0.0, 0.0, 1.0, 0.0]
        used_by_alpha = {
            0.20: [True, True, True, True, True, True],
            0.05: [True, False, True, True, False, True],
            0.01: [False, False, True, False, False, True],
        }
        curve = compute_coverage_safety_curve(true_advs, regrets, used_by_alpha)
        assert len(curve) == 3
        # Lower alpha (more conservative) -> lower coverage.
        assert curve[0.01]["coverage"] <= curve[0.20]["coverage"]

    def test_select_operating_point(self):
        curve = {
            0.20: {"override_precision": 0.80, "coverage": 0.30, "p95_regret": 10, "cvar95": 15},
            0.10: {"override_precision": 0.95, "coverage": 0.20, "p95_regret": 8, "cvar95": 12},
            0.05: {"override_precision": 1.0, "coverage": 0.10, "p95_regret": 5, "cvar95": 8},
        }
        alpha, metrics = select_operating_point(
            curve, min_precision=0.95, min_coverage=0.10,
        )
        # alpha=0.10 has precision=0.95 and coverage=0.20 — highest coverage
        # that satisfies precision >= 0.95 and coverage >= 0.10.
        assert alpha == 0.10


class TestNoLeakage:
    """Test no-leakage assertions."""

    def test_no_future_oracle_leakage(self):
        features = np.random.randn(10, 5)
        names = ["state_feat_1", "state_feat_2", "action_type", "u", "v"]
        assert_no_future_oracle_leakage(features, names)  # should not raise

    def test_future_oracle_leakage_detected(self):
        features = np.random.randn(10, 5)
        names = ["state_feat_1", "exact_q", "action_type", "u", "v"]
        with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
            assert_no_future_oracle_leakage(features, names)

    def test_train_calibration_test_isolation(self):
        train = np.array([0, 1, 2, 3])
        cal = np.array([4, 5, 6, 7])
        test = np.array([8, 9, 10, 11])
        assert_train_calibration_test_isolation(train, cal, test)  # should not raise

    def test_isolation_violation_detected(self):
        train = np.array([0, 1, 2, 3])
        cal = np.array([3, 4, 5, 6])  # overlaps with train
        test = np.array([7, 8, 9, 10])
        with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
            assert_train_calibration_test_isolation(train, cal, test)


class TestOODDiagnostics:
    """Test OOD diagnostics."""

    def test_ood_scores(self):
        train = np.random.randn(50, 5)
        test = np.random.randn(10, 5)
        scores = compute_ood_scores(train, test)
        assert scores.shape == (10,)
        assert np.all(scores >= 0)

    def test_ood_coverage_analysis(self):
        ood_scores = np.linspace(0, 10, 100)
        used = [True] * 50 + [False] * 50  # in-dist: override, OOD: don't
        result = compute_ood_coverage_analysis(ood_scores, used)
        assert "deciles" in result
        assert "is_monotonic" in result
        # Coverage should decrease with OOD score.
        assert result["is_monotonic"] is True


class TestAdvantageFeatures:
    """Test advantage feature extraction."""

    def test_encode_action(self):
        action = ("add_edge", 0, 2, {"weight": 1.0, "factor": 2.0})
        action_id = ActionIdentity.from_action(action)
        feat = encode_action(action, action_id)
        assert feat.shape == (9,)  # 4 type + 5 params
        assert feat[0] == 1.0  # add_edge one-hot

    def test_pairwise_features(self):
        baseline = ("add_edge", 0, 1, {"weight": 1.0})
        learned = ("remove_edge", 1, 2, {"weight": 1.0})
        b_id = ActionIdentity.from_action(baseline)
        l_id = ActionIdentity.from_action(learned)
        feat = extract_pairwise_features(baseline, learned, b_id, l_id)
        assert feat.shape == (27,)  # 3 * 9

    def test_full_feature_dim(self):
        assert FULL_FEATURE_DIM > 0
        assert PAIRWISE_FEATURE_DIM == 3 * ACTION_FEATURE_DIM
