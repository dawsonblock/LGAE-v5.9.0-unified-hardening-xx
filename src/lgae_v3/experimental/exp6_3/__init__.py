"""v6.0-exp6.3: Long-horizon structural value.

Module layout:
- delayed_tasks.py: Non-greedy benchmark tasks with non-additive utility
- exact_mpc.py: Exact exhaustive multi-step planning
- future_value.py: Future value model ladder (V0-V6)
- beam_search.py: Deterministic beam search with UCB retention
- trust_bundle.py: Separated trust channels (dynamics/value/risk)
- horizon_policy.py: Trust-gated horizon selection
- test_e.py: Untouched TEST-E generators
- value_dataset.py: Exact-enumerated training data
- metrics.py: First-action agreement, planning regret, search savings
- experiment_runner.py: Full experiment orchestration
"""
from .delayed_tasks import (
    DelayedValueTask, get_all_delayed_value_tasks,
    make_task_graph, make_task_latent,
    UTILITY_FUNCTIONS,
)
from .exact_mpc import (
    ExactPlan, exact_mpc, exact_mpc_additive, greedy_one_step,
)
from .future_value import (
    FutureValueModel, V0Zero, V1TypeMean, V2Linear, V3Ridge, V5MLP,
)
from .beam_search import (
    BeamSearchResult, beam_search, beam_search_with_ucb,
)
from .trust_bundle import (
    TrustBundle, DynamicsTrust, ValueTrust, RiskTrust,
    compute_trust_bundle,
)
from .horizon_policy import (
    HorizonPolicy,
)
from .test_e import (
    TestEConfig, generate_test_e_configs, generate_test_e_graph,
)
from .metrics import (
    first_action_agreement, planning_regret, search_savings,
    trajectory_recall, greedy_improvement,
)
from .value_dataset import (
    ValueRecord, generate_value_dataset,
)
from .experiment_runner import (
    run_exp6_3, FamilyResult, ExperimentResult,
)

__all__ = [
    "DelayedValueTask", "get_all_delayed_value_tasks",
    "make_task_graph", "make_task_latent", "UTILITY_FUNCTIONS",
    "ExactPlan", "exact_mpc", "exact_mpc_additive", "greedy_one_step",
    "FutureValueModel", "V0Zero", "V1TypeMean", "V2Linear", "V3Ridge", "V5MLP",
    "BeamSearchResult", "beam_search", "beam_search_with_ucb",
    "TrustBundle", "DynamicsTrust", "ValueTrust", "RiskTrust",
    "compute_trust_bundle",
    "HorizonPolicy",
    "TestEConfig", "generate_test_e_configs", "generate_test_e_graph",
    "first_action_agreement", "planning_regret", "search_savings",
    "trajectory_recall", "greedy_improvement",
    "ValueRecord", "generate_value_dataset",
    "run_exp6_3", "FamilyResult", "ExperimentResult",
]
