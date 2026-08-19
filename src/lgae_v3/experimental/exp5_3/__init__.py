"""v6.0-exp5.3: Topology-invariant representation study.

Key methodology corrections from exp5.2:
- Train and evaluate on REALIZED records only (not counterfactual)
- Primary metric is delta R² on invariant dimensions, not absolute R²
- Always compare against zero-delta baseline
- Decompose state into invariant/context/derived dimensions

Experiment matrix:
- Representation ladder: R0-R7 (graphlet, spectral, curvature, hybrid, learned)
- Component-wise adaptation: bias-only, scale+offset, low-rank, full
- Dynamics-OOD distance (over (z_t, a_t, Δz_t), not just z_t)
- Parametric graph families with continuous structural variation
- TEST-C from different generators
"""
from .representations import (
    REPRESENTATION_LADDER, RepresentationConfig,
    extract_representation, extract_invariant, extract_context, extract_derived,
    INVARIANT_INDICES, CONTEXT_INDICES, DERIVED_INDICES,
    RepresentationMetrics, compute_representation_metrics,
)
from .adaptation import (
    ComponentAdapter, BiasOnlyAdapter, ScaleOffsetAdapter,
    LowRankAdapter, FullRetrainAdapter, create_adapter,
)
from .dynamics_ood import (
    compute_dynamics_ood_distance, compute_family_dynamics_ood,
    correlate_dynamics_ood_with_error,
    correlate_dynamics_ood_with_uncertainty,
)
from .parametric_families import (
    ParametricFamilyConfig, generate_parametric_graph,
    generate_parametric_dataset, generate_test_c_configs,
)
from .experiment_runner import (
    is_realized, extract_realized_data,
    run_representation_ladder,
    run_loo_with_representations,
    run_adaptation_study,
    run_dynamics_ood_analysis,
)

__all__ = [
    "REPRESENTATION_LADDER", "RepresentationConfig",
    "extract_representation", "extract_invariant", "extract_context", "extract_derived",
    "INVARIANT_INDICES", "CONTEXT_INDICES", "DERIVED_INDICES",
    "RepresentationMetrics", "compute_representation_metrics",
    "ComponentAdapter", "BiasOnlyAdapter", "ScaleOffsetAdapter",
    "LowRankAdapter", "FullRetrainAdapter", "create_adapter",
    "compute_dynamics_ood_distance", "compute_family_dynamics_ood",
    "correlate_dynamics_ood_with_error",
    "correlate_dynamics_ood_with_uncertainty",
    "ParametricFamilyConfig", "generate_parametric_graph",
    "generate_parametric_dataset", "generate_test_c_configs",
    "is_realized", "extract_realized_data",
    "run_representation_ladder",
    "run_loo_with_representations",
    "run_adaptation_study",
    "run_dynamics_ood_analysis",
]
