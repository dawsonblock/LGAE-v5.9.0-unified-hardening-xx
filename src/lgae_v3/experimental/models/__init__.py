"""v6.0-exp4: Outcome, risk, and cost models.

Determines which encoder/predictor combinations can reliably predict
the consequences of structural interventions.

Targets:
    ΔU = f_θ(z_{S,a})
    R  = g_φ(z_{S,a})
    C  = h_ψ(z_{S,a})
    P(ΔU > 0 | S, a)  for sign/success classification

Predictor ladder:
    Global mean → Mutation-type mean → Nearest experience →
    Linear → Ridge → Tree → MLP → Graph-encoder head → Hybrid

Hard scientific stop condition:
    If no model beats simple handcrafted representations plus a
    tree/linear predictor on held-out graph families, do not proceed
    to a sophisticated world model.
"""
from __future__ import annotations

from .protocol import (
    Prediction, ClassificationPrediction, RankingPrediction,
    OutcomeModel, ClassificationModel, RankingModel,
    ModelLifecycle, ensure_finite_pred, safe_sigmoid, config_hash,
)
from .targets import (
    TargetType, TargetDefinition, TargetSet, DEFAULT_TARGETS,
    TARGET_REALIZED_DELTA, TARGET_SIGN_DELTA, TARGET_NORMALIZED_DELTA,
    TARGET_UTILITY_BUCKET, TARGET_CANDIDATE_RANK, TARGET_RISK, TARGET_COST,
    RISK_COMPONENTS, COST_COMPONENTS,
    compute_sign_delta, compute_normalized_delta, compute_utility_bucket,
    compute_candidate_ranks, compute_pairwise_labels,
    aggregate_risk, aggregate_cost,
)
from .baselines import (
    GlobalMeanPredictor, MutationTypeMeanPredictor, NearestExperiencePredictor,
)
from .linear import (
    LinearRegressionPredictor, RidgeRegressionPredictor, LogisticRegressionPredictor,
)
from .tree import GradientBoostedTreePredictor
from .mlp import MLPRegressor, MLPClassifier
from .ranking import PointwiseRankingModel, PairwiseRankingModel
from .uncertainty import (
    UncertaintyReport, BootstrapEnsemble, analyze_uncertainty, quantile_uncertainty,
)
from .calibration import (
    CalibrationReport, ReliabilityCurve,
    expected_calibration_error, brier_score, reliability_curve,
    prediction_interval_coverage, standardized_residual_calibration,
    calibration_drift,
)
from .evaluator import (
    RegressionMetrics, ClassificationMetrics, RankingMetrics,
    GroupMetrics, CFToRealGap,
    compute_regression_metrics, compute_classification_metrics,
    compute_ranking_metrics, compute_group_metrics,
    compute_cf_to_real_gap, compute_ood_degradation,
)
from .artifact import ModelArtifact, CompatibilityError, create_artifact
from .model_registry import ModelRegistry

__all__ = [
    # Protocol
    "Prediction", "ClassificationPrediction", "RankingPrediction",
    "OutcomeModel", "ClassificationModel", "RankingModel",
    "ModelLifecycle", "ensure_finite_pred", "safe_sigmoid", "config_hash",
    # Targets
    "TargetType", "TargetDefinition", "TargetSet", "DEFAULT_TARGETS",
    "TARGET_REALIZED_DELTA", "TARGET_SIGN_DELTA", "TARGET_NORMALIZED_DELTA",
    "TARGET_UTILITY_BUCKET", "TARGET_CANDIDATE_RANK", "TARGET_RISK", "TARGET_COST",
    "RISK_COMPONENTS", "COST_COMPONENTS",
    "compute_sign_delta", "compute_normalized_delta", "compute_utility_bucket",
    "compute_candidate_ranks", "compute_pairwise_labels",
    "aggregate_risk", "aggregate_cost",
    # Baselines
    "GlobalMeanPredictor", "MutationTypeMeanPredictor", "NearestExperiencePredictor",
    # Linear
    "LinearRegressionPredictor", "RidgeRegressionPredictor", "LogisticRegressionPredictor",
    # Tree
    "GradientBoostedTreePredictor",
    # MLP
    "MLPRegressor", "MLPClassifier",
    # Ranking
    "PointwiseRankingModel", "PairwiseRankingModel",
    # Uncertainty
    "UncertaintyReport", "BootstrapEnsemble", "analyze_uncertainty", "quantile_uncertainty",
    # Calibration
    "CalibrationReport", "ReliabilityCurve",
    "expected_calibration_error", "brier_score", "reliability_curve",
    "prediction_interval_coverage", "standardized_residual_calibration",
    "calibration_drift",
    # Evaluator
    "RegressionMetrics", "ClassificationMetrics", "RankingMetrics",
    "GroupMetrics", "CFToRealGap",
    "compute_regression_metrics", "compute_classification_metrics",
    "compute_ranking_metrics", "compute_group_metrics",
    "compute_cf_to_real_gap", "compute_ood_degradation",
    # Artifact
    "ModelArtifact", "CompatibilityError", "create_artifact",
    # Registry
    "ModelRegistry",
]
