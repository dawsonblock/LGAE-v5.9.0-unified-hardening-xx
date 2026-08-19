"""v6.0-exp5.2: Cross-family generalization program.

Goal: determine whether structural dynamics can be represented in a
topology-invariant way that transfers across unseen graph families.

Key changes from exp5.1:
- Delta-state prediction (Δz = F(z_t, a_t), reconstruct z_{t+1} = z_t + Δz)
- Normalized/scale-invariant state features
- Topology-invariant descriptors (graphlet frequencies, degree entropy, etc.)
- Leave-one-family-out cross-validation
- Family-bootstrap ensemble
- OOD distance analysis
- Adaptation curves (0-shot, 5-shot, 10-shot, 25-shot, 50-shot)
- Extended rollout horizons (h=1,2,3,5,10)
- Zero-shot vs adapted trust distinction
"""
from .state_encoding import (
    encode_normalized_state,
    encode_normalized_action,
    NormalizedStateVector,
    NormalizedActionVector,
    NORM_STATE_DIM,
    NORM_ACTION_DIM,
)
from .dynamics import (
    DeltaDynamicsModel,
    FamilyBootstrapEnsemble,
    compute_generalization_metrics,
    GeneralizationMetrics,
)
from .ood_analysis import (
    compute_family_ood_distances,
    correlate_ood_with_error,
    correlate_ood_with_uncertainty,
)
from .experiment_runner import (
    RepresentationResult,
    LOOResult,
    AdaptationResult,
    extract_normalized_data,
    run_representation_ablation,
    run_leave_one_family_out,
    run_family_bootstrap_ensemble,
    run_ood_analysis,
    run_adaptation_curves,
    run_extended_rollout,
)

__all__ = [
    "encode_normalized_state",
    "encode_normalized_action",
    "NormalizedStateVector",
    "NormalizedActionVector",
    "NORM_STATE_DIM",
    "NORM_ACTION_DIM",
    "DeltaDynamicsModel",
    "FamilyBootstrapEnsemble",
    "compute_generalization_metrics",
    "GeneralizationMetrics",
    "compute_family_ood_distances",
    "correlate_ood_with_error",
    "correlate_ood_with_uncertainty",
    "RepresentationResult",
    "LOOResult",
    "AdaptationResult",
    "extract_normalized_data",
    "run_representation_ablation",
    "run_leave_one_family_out",
    "run_family_bootstrap_ensemble",
    "run_ood_analysis",
    "run_adaptation_curves",
    "run_extended_rollout",
]
