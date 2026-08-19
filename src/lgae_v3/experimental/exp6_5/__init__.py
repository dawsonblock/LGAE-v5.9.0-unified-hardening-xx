"""v6.0-exp6.5: Cross-mechanism foresight generalization.

Goal: Prove the learned future-residual model generalizes across
different forms of non-additive structural value.

Method: Leave-one-mechanism-out evaluation.
  Train on {M_j : j != i}, Test on M_i.

Mechanisms:
  - connectivity_threshold
  - redundancy_threshold
  - hub_load_threshold
  - spectral_gap_threshold
"""
from .multi_mechanism_data import (
    MECHANISM_NAMES,
    generate_multi_mechanism_training_data,
    generate_mechanism_eval_tasks,
)
from .observable_features import (
    extract_observable_features,
    OBSERVABLE_FEATURE_DIM,
)
from .decomposed_model import (
    DecomposedModel, ScalarMLP, MultiHeadModel, EnsembleScalarMLP,
    get_decomposed_model_ladder,
)
from .adaptive_beam import (
    adaptive_beam_search, AdaptiveBeamResult,
)
from .scaling_benchmark import (
    ScalingConfig, run_scaling_benchmark,
)
from .experiment_runner import (
    run_exp6_5, Exp65Result, LOMOResult, ScalingResult,
)

__all__ = [
    "MECHANISM_NAMES",
    "generate_multi_mechanism_training_data",
    "generate_mechanism_eval_tasks",
    "extract_observable_features",
    "OBSERVABLE_FEATURE_DIM",
    "DecomposedModel", "MultiHeadModel", "get_decomposed_model_ladder",
    "adaptive_beam_search", "AdaptiveBeamResult",
    "ScalingConfig", "run_scaling_benchmark",
    "run_exp6_5", "Exp65Result", "LOMOResult", "ScalingResult",
]
