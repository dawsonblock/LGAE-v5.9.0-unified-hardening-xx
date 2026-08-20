"""exp6.8.2: Calibrated selective planning."""
from .ensemble_model import EnsembleLearnedModel
from .lcb_planner import lcb_hybrid_plan, calibrate_kappa, LCBPlanResult, _compute_cvar
from .extended_risk_metrics import (
    compute_cvar, compute_extended_risk_metrics,
    compute_uncertainty_error_correlation,
    compute_risk_by_uncertainty_deciles,
)
from .experiment_runner import run_exp6_8_2, Exp682Result, LOMOResult682

__all__ = [
    "EnsembleLearnedModel",
    "lcb_hybrid_plan", "calibrate_kappa", "LCBPlanResult", "_compute_cvar",
    "compute_cvar", "compute_extended_risk_metrics",
    "compute_uncertainty_error_correlation",
    "compute_risk_by_uncertainty_deciles",
    "run_exp6_8_2", "Exp682Result", "LOMOResult682",
]
