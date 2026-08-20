"""exp6.8.1: Selective hybrid structural planning."""
from .deterministic_oracles import (
    compute_spectral_gap_deterministic,
    compute_effective_resistance,
    compute_curvature_estimate,
)
from .split_state import (
    SplitStructuralState, ExactState, CertifiedApproxState, LearnedState,
    EXACT_STATE_DIM, CERTIFIED_STATE_DIM, LEARNED_STATE_DIM, FULL_STATE_DIM,
)
from .learned_state_model import LearnedStateModel
from .hybrid_planner import (
    selective_hybrid_plan, run_coverage_sweep, HybridPlanResult,
)
from .risk_metrics import (
    compute_regret_distribution, compute_normalized_regret_distribution,
    compute_risk_metrics, compute_coverage_risk_curve,
)
from .experiment_runner import (
    run_exp6_8_1, Exp681Result, LOMOResult681,
)

__all__ = [
    "compute_spectral_gap_deterministic",
    "compute_effective_resistance",
    "compute_curvature_estimate",
    "SplitStructuralState", "ExactState", "CertifiedApproxState", "LearnedState",
    "EXACT_STATE_DIM", "CERTIFIED_STATE_DIM", "LEARNED_STATE_DIM", "FULL_STATE_DIM",
    "LearnedStateModel",
    "selective_hybrid_plan", "run_coverage_sweep", "HybridPlanResult",
    "compute_regret_distribution", "compute_normalized_regret_distribution",
    "compute_risk_metrics", "compute_coverage_risk_curve",
    "run_exp6_8_1", "Exp681Result", "LOMOResult681",
]
