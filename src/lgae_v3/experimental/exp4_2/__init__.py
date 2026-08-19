"""v6.0-exp4.2: Held-out structural prediction study.

This package implements the first genuinely scientific LGAE structural
prediction study. Its purpose is to determine whether information
available at structural decision time (S_t, a_t) contains enough
generalizable signal to predict the quality of structural interventions
on graph families not used for training or model selection.

Hypotheses:
    H1: There exists f(S, a) that predicts relative intervention quality
        on unseen graph families, materially outperforming baselines.
    H0: Available state/action information provides no useful generalizable
        predictive signal beyond simple statistical baselines.

A failed experiment is an acceptable outcome. Do not redefine success
after viewing held-out results.

The experiment follows a strict protocol:
    PREPARATION -> TRAINING -> VALIDATION -> MODEL_LOCKED
        -> HELDOUT_OPENED -> FINALIZED

No backward transition from HELDOUT_OPENED to model selection.
"""
from __future__ import annotations

from .experiment_state import (
    ExperimentStateError,
    ExperimentStateMachine,
    STATES as EXPERIMENT_STATES,
)
from .dataset_freeze import (
    DatasetFreeze,
    SplitFreeze,
    freeze_dataset,
    load_dataset_freeze,
)
from .targets import (
    TargetType,
    TargetDefinition,
    TARGET_DEFINITIONS,
    get_target_definition,
)
from .metrics import (
    RegretReport,
    OracleRecoveryReport,
    SelectivePredictionReport,
    ParetoFrontierEntry,
    ParetoFrontier,
    compute_regret,
    compute_oracle_recovery,
    compute_selective_prediction,
    compute_pareto_frontier,
    bootstrap_ci,
    UncertaintyCorrelationReport,
    compute_uncertainty_error_correlation,
)
from .cf_real import (
    SupervisionRegime,
    CFRealTransferReport,
    run_cf_real_experiment,
)
from .experiment_config import (
    ExperimentConfig,
    EncoderConfig,
    PredictorConfig,
    FinalistLock,
    SelectionWeights,
)
from .scientific_runner import (
    ScientificRunner,
    ScientificResult,
    ScientificConclusion,
    authorize_exp5,
)
from .report_generator import (
    generate_scientific_report,
    generate_machine_readable_conclusion,
)

__all__ = [
    "ExperimentStateError",
    "ExperimentStateMachine",
    "EXPERIMENT_STATES",
    "DatasetFreeze",
    "SplitFreeze",
    "freeze_dataset",
    "load_dataset_freeze",
    "TargetType",
    "TargetDefinition",
    "TARGET_DEFINITIONS",
    "get_target_definition",
    "RegretReport",
    "OracleRecoveryReport",
    "SelectivePredictionReport",
    "ParetoFrontierEntry",
    "ParetoFrontier",
    "compute_regret",
    "compute_oracle_recovery",
    "compute_selective_prediction",
    "compute_pareto_frontier",
    "bootstrap_ci",
    "UncertaintyCorrelationReport",
    "compute_uncertainty_error_correlation",
    "SupervisionRegime",
    "CFRealTransferReport",
    "run_cf_real_experiment",
    "ExperimentConfig",
    "EncoderConfig",
    "PredictorConfig",
    "FinalistLock",
    "SelectionWeights",
    "ScientificRunner",
    "ScientificResult",
    "ScientificConclusion",
    "authorize_exp5",
    "generate_scientific_report",
    "generate_machine_readable_conclusion",
]
