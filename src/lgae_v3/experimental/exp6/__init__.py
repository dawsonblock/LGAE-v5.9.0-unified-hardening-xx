"""v6.0-exp6: Adaptive model-assisted MPC.

Architecture:
    Δz = α_G ⊙ F_θ(z, a) + β_G

where F_θ is the global structural prior and (α_G, β_G) is a tiny
topology-local calibration fitted from a few exact transitions.

The learned model assists candidate reduction and trajectory
prioritization. Exact counterfactual execution remains authoritative.
Every final action is verified through the v5.11 CommitChannel.
"""
from .calibration import (
    TopologyCalibration, fit_calibration, identity_calibration,
    compute_calibration_hash,
)
from .calibration_controller import (
    CalibrationState, CalibrationConfig, CalibrationResult,
    run_calibration_acquisition, loo_validate_calibration,
    select_diverse_samples, compute_sample_diversity_score,
)
from .trust import (
    TrustPolicyState, TrustFactors, TrustGates, TrustReport,
    compute_trust_state, compute_max_horizon, assess_trust,
)
from .prefilter import (
    Candidate, PrefilterResult,
    score_candidates, prefilter_candidates, compute_oracle_recall,
)
from .experiment_runner import (
    FamilyMPCResult, run_family_mpc, run_adaptation_curve,
)
from .candidate_generator import (
    StructuralCandidate, generate_candidates, evaluate_candidates_exact,
    apply_candidate, compute_exact_utility,
)
from .metrics import (
    RecallMetrics, RegretDistribution,
    compute_oracle_recall, compute_near_oracle_recall,
    compute_regret_distribution, compute_pruning_ratio_metrics,
    compare_filtering_strategies,
)
from .test_c import (
    TestCFamilyConfig, generate_test_c_configs, generate_test_c_graph,
)
from .exp6_1_runner import (
    FamilyResult, run_family_experiment,
)

__all__ = [
    "TopologyCalibration", "fit_calibration", "identity_calibration",
    "compute_calibration_hash",
    "CalibrationState", "CalibrationConfig", "CalibrationResult",
    "run_calibration_acquisition", "loo_validate_calibration",
    "select_diverse_samples", "compute_sample_diversity_score",
    "TrustPolicyState", "TrustFactors", "TrustGates", "TrustReport",
    "compute_trust_state", "compute_max_horizon", "assess_trust",
    "Candidate", "PrefilterResult",
    "score_candidates", "prefilter_candidates", "compute_oracle_recall",
    "FamilyMPCResult", "run_family_mpc", "run_adaptation_curve",
    "StructuralCandidate", "generate_candidates", "evaluate_candidates_exact",
    "apply_candidate", "compute_exact_utility",
    "RecallMetrics", "RegretDistribution",
    "compute_oracle_recall", "compute_near_oracle_recall",
    "compute_regret_distribution", "compute_pruning_ratio_metrics",
    "compare_filtering_strategies",
    "TestCFamilyConfig", "generate_test_c_configs", "generate_test_c_graph",
    "FamilyResult", "run_family_experiment",
]
