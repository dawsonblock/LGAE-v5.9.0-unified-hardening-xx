"""v6.0-exp6.4: Learned non-additive value.

Goal: Learn the residual non-additive value well enough that honest
beam search can recover non-greedy first actions without access to
the exact future utility function.

Key insight: Instead of predicting scalar bonus directly, predict the
causal intermediate — component count change and threshold reachability.
"""
from .structural_features import (
    extract_structural_features,
    ComponentInfo, compute_component_info,
)
from .causal_targets import (
    CausalTarget, compute_causal_targets,
)
from .model_ladder import (
    BonusModel, B0Zero, B1Logistic, B2Tree, B3GBT, B4MLP, B5EnsembleMLP,
    get_model_ladder,
)
from .procedural_tasks import (
    ProceduralTaskConfig, generate_procedural_tasks,
    make_procedural_graph, generate_candidates,
    generate_procedural_training_data,
)
from .test_f import (
    TestFConfig, generate_test_f_configs, generate_test_f_graph,
    make_test_f_utility,
)
from .honest_beam_v2 import (
    honest_beam_search_v2, HonestBeamResultV2,
)
from .experiment_runner import (
    run_exp6_4, Exp64FamilyResult, Exp64Result,
)

__all__ = [
    "extract_structural_features",
    "ComponentInfo", "compute_component_info",
    "CausalTarget", "compute_causal_targets",
    "BonusModel", "B0Zero", "B1Logistic", "B2Tree", "B3GBT", "B4MLP", "B5EnsembleMLP",
    "get_model_ladder",
    "ProceduralTaskConfig", "generate_procedural_tasks",
    "generate_procedural_training_data",
    "TestFConfig", "generate_test_f_configs", "generate_test_f_graph",
    "make_test_f_utility",
    "honest_beam_search_v2", "HonestBeamResultV2",
    "run_exp6_4", "Exp64FamilyResult", "Exp64Result",
]
