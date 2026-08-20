"""v6.0-exp6.7: Multi-operator causal structural model.

Extends exp6.6 with:
  1. Heterogeneous mutations: ADD_EDGE, REMOVE_EDGE, REWEIGHT_EDGE, EDGE_SWAP
  2. 7 structural effect heads (added path_length, efficiency, curvature)
  3. >=100 non-greedy states per mechanism
  4. Paired bootstrap CIs for Recovery_C - Recovery_A
  5. Reward-formulation hold-out (train threshold, test linear/composite)
"""
from .multi_operator_candidates import (
    MUTATION_TYPES, generate_multi_operator_candidates,
    generate_multi_operator_training_data,
)
from .extended_effects import (
    ExtendedEffect, compute_extended_effect_labels,
    EXTENDED_EFFECT_DIM,
)
from .causal_effect_model_v2 import (
    CausalEffectModelV2, ObjectiveEvaluatorV2,
    ScalarResidualModelV2, get_architecture_ladder_v2,
)
from .reward_variants import (
    RewardVariant, make_reward_variant_utility,
    REWARD_VARIANTS,
)
from .experiment_runner import (
    run_exp6_7, Exp67Result, LOMOResultV3, RewardHoldoutResult,
)

__all__ = [
    "MUTATION_TYPES", "generate_multi_operator_candidates",
    "generate_multi_operator_training_data",
    "ExtendedEffect", "compute_extended_effect_labels",
    "EXTENDED_EFFECT_DIM",
    "CausalEffectModelV2", "ObjectiveEvaluatorV2",
    "ScalarResidualModelV2", "get_architecture_ladder_v2",
    "RewardVariant", "make_reward_variant_utility", "REWARD_VARIANTS",
    "run_exp6_7", "Exp67Result", "LOMOResultV3", "RewardHoldoutResult",
]
