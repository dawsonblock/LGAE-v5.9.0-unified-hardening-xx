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
]
