"""exp6.8: Exact-transition model-based structural planning."""
from .structural_state import (
    StructuralState, compute_structural_observables,
    STRUCTURAL_OBSERVABLE_DIM, get_observable_value,
)
from .transition_model import (
    ConsequentialStateModel, exact_transition,
    roll_forward_exact, roll_forward_predicted,
)
from .recursive_planner import (
    recursive_causal_mpc, evaluate_objective_on_state,
    RecursivePlanResult,
)
from .experiment_runner import (
    run_exp6_8, Exp68Result, LOMOResult,
)

__all__ = [
    "StructuralState", "compute_structural_observables",
    "STRUCTURAL_OBSERVABLE_DIM", "get_observable_value",
    "ConsequentialStateModel", "exact_transition",
    "roll_forward_exact", "roll_forward_predicted",
    "recursive_causal_mpc", "evaluate_objective_on_state",
    "RecursivePlanResult",
    "run_exp6_8", "Exp68Result", "LOMOResult",
]
