"""exp6.8.3: Conformal Structural Advantage."""
from .advantage_dataset import (
    AdvantageRecord, compute_exact_q_h2, generate_advantage_dataset,
    records_to_arrays,
)
from .advantage_features import (
    build_full_features, extract_state_features, extract_objective_features,
    extract_pairwise_features, encode_action,
    FULL_FEATURE_DIM, STATE_FEATURE_DIM, PAIRWISE_FEATURE_DIM,
    ACTION_FEATURE_DIM, ACTION_TYPE_DIM,
)
from .advantage_models import (
    ZeroAdvantageModel, LinearRegressionModel, RidgeRegressionModel,
    MLPModel, BootstrapMLPEnsemble, QuantileMLPModel, get_model_ladder,
)
from .conformal_calibration import (
    compute_conformal_quantile, calibrate_conformal,
    compute_lcb_advantage, select_operating_alpha,
    compute_conformalized_quantile_intervals,
)
from .conformal_arbitrator import (
    conformal_arbitrate, batch_arbitrate, ConformalArbitrationResult,
)
from .risk_metrics import (
    compute_override_precision, compute_false_override_rate,
    compute_override_coverage, compute_mean_override_advantage,
    compute_regret_metrics, compute_normalized_regret,
    compute_cvar, compute_bootstrap_ci,
    compute_uncertainty_error_correlation,
    compute_confidence_decile_analysis,
)
from .coverage_analysis import compute_coverage_safety_curve, select_operating_point
from .pairwise_models import ArbitrationComparison, compare_arbitration_systems
from .no_leakage import (
    assert_no_future_oracle_leakage, assert_no_test_statistics_leakage,
    assert_train_calibration_test_isolation, assert_no_exact_mpc_in_features,
)
from .ood_diagnostics import compute_ood_scores, compute_ood_coverage_analysis
from .experiment_runner import run_exp6_8_3, Exp683Result, MechanismResult

__all__ = [
    "AdvantageRecord", "compute_exact_q_h2", "generate_advantage_dataset",
    "records_to_arrays",
    "build_full_features", "extract_state_features", "extract_objective_features",
    "extract_pairwise_features", "encode_action",
    "FULL_FEATURE_DIM", "STATE_FEATURE_DIM", "PAIRWISE_FEATURE_DIM",
    "ACTION_FEATURE_DIM", "ACTION_TYPE_DIM",
    "ZeroAdvantageModel", "LinearRegressionModel", "RidgeRegressionModel",
    "MLPModel", "BootstrapMLPEnsemble", "QuantileMLPModel", "get_model_ladder",
    "compute_conformal_quantile", "calibrate_conformal",
    "compute_lcb_advantage", "select_operating_alpha",
    "compute_conformalized_quantile_intervals",
    "conformal_arbitrate", "batch_arbitrate", "ConformalArbitrationResult",
    "compute_override_precision", "compute_false_override_rate",
    "compute_override_coverage", "compute_mean_override_advantage",
    "compute_regret_metrics", "compute_normalized_regret",
    "compute_cvar", "compute_bootstrap_ci",
    "compute_uncertainty_error_correlation",
    "compute_confidence_decile_analysis",
    "compute_coverage_safety_curve", "select_operating_point",
    "ArbitrationComparison", "compare_arbitration_systems",
    "assert_no_future_oracle_leakage", "assert_no_test_statistics_leakage",
    "assert_train_calibration_test_isolation", "assert_no_exact_mpc_in_features",
    "compute_ood_scores", "compute_ood_coverage_analysis",
    "run_exp6_8_3", "Exp683Result", "MechanismResult",
]
