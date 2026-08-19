"""v6.0-exp6.6: Objective-conditioned causal foresight.

Three architectures compared:
  A. Scalar residual model: F(S,a) → R
  B. Objective-conditioned scalar: F(S,a,O) → R
  C. Causal structural effect model: F(S,a) → effects, O(effects) → R

The key hypothesis: architecture C generalizes across objectives
because it separates the physics of structural change from the
objective being optimized.
"""
from .objective_spec import (
    ObjectiveSpec, OBJECTIVE_SPECS, get_objective_spec,
    encode_objective, OBJECTIVE_ENCODING_DIM,
)
from .causal_effect_model import (
    StructuralEffect, compute_effect_labels,
    CausalEffectModel, ObjectiveEvaluator,
    ScalarResidualModel, ObjectiveConditionedModel,
    get_architecture_ladder,
)
from .honest_beam_v3 import honest_beam_search_v3, HonestBeamResultV3
from .experiment_runner import (
    run_exp6_6, Exp66Result, LOMOResultV2,
)

__all__ = [
    "ObjectiveSpec", "OBJECTIVE_SPECS", "get_objective_spec",
    "encode_objective", "OBJECTIVE_ENCODING_DIM",
    "StructuralEffect", "compute_effect_labels",
    "CausalEffectModel", "ObjectiveEvaluator",
    "ScalarResidualModel", "ObjectiveConditionedModel",
    "get_architecture_ladder",
    "honest_beam_search_v3", "HonestBeamResultV3",
    "run_exp6_6", "Exp66Result", "LOMOResultV2",
]
