"""v6.0-exp1 + exp2: Structural world-model foundation and transition dataset.

Experimental infrastructure for v6 research. This subpackage is strictly
advisory: it never touches the v5.11 authority boundary. Learned models,
transition recorders, and benchmark harnesses propose and observe; only the
frozen v5.11 CommitChannel may mutate authoritative state.

Architecture:

    Learned intelligence
            ↓
         proposal
            ↓
        prediction
            ↓
           MPC
            ↓
     exact counterfactual
            ↓
        governance
            ↓
     v5.11 authority boundary
            ↓
      canonical transaction

Modules:
- ``graph_families``: frozen train/validation/held-out graph family splits.
- ``metrics``: v6 evaluation metrics (performance per compute, adaptation
  speed, OOD generalization, mutation count, topology complexity, failure
  rate, calibration).
- ``baselines``: eleven baseline runners (fixed-topology, random-rewiring,
  greedy, curvature-only, FoSR/BORF/effective-resistance, one-step
  counterfactual, MPC, MPC+IG, full-v5.11).
- ``benchmark_harness``: orchestrates families × baselines × metrics.
- ``transition_recorder``: instruments v5.11 runtime to produce
  (S_t, a_t, S_{t+1}, ΔU, C, R) records outside authoritative state.
- ``transition_record``: rich TransitionRecord with ~25 fields, supporting
  ObservedTransition and CounterfactualTransition provenance types.
- ``feature_extraction``: canonical structural feature extraction (global
  24-dim + local action 12-dim).
- ``dataset_generator``: generates split datasets with negative sampling,
  quality reports, and validation.
- ``dataset_validator``: validates datasets for leakage, duplicates, schema,
  hash mismatch, and nonfinite metrics.
- ``quality_report``: data quality reports with distributions and imbalance
  detection.
- ``dataset``: structural dataset schema and serialization.
- ``experiment_registry``: experiment registry with provenance tracking.
- ``reproducibility``: seed management, config hashing, run fingerprinting.
- ``world_model``: abstract world-model and outcome-model interfaces.
"""
from __future__ import annotations

from .graph_families import (
    GraphFamilySplit,
    FrozenGraphFamilyRegistry,
    FROZEN_TRAIN_FAMILIES,
    FROZEN_VALIDATION_FAMILIES,
    FROZEN_HELD_OUT_FAMILIES,
    get_frozen_registry,
)
from .metrics import (
    V6Metric,
    V6MetricReport,
    AdaptationSpeedMetric,
    PerformancePerComputeMetric,
    OODGeneralizationMetric,
    MutationCountMetric,
    TopologyComplexityMetric,
    FailureRateMetric,
    CalibrationMetric,
    adaptation_speed_metric,
    performance_per_compute_metric,
    ood_generalization_metric,
    mutation_count_metric,
    topology_complexity_metric,
    failure_rate_metric,
    calibration_metric,
    aggregate_metrics,
)
from .baselines import (
    BaselineRunner,
    FixedTopologyBaseline,
    RandomRewiringBaseline,
    GreedyBaseline,
    CurvatureOnlyBaseline,
    FoSRBaseline,
    BORFBaseline,
    EffectiveResistanceBaseline,
    OneStepCounterfactualBaseline,
    MPCBaseline,
    MPCWithIGBaseline,
    FullV511Baseline,
    ALL_V6_BASELINES,
)
from .benchmark_harness import V6BenchmarkHarness, BenchmarkRunResult
from .transition_recorder import (
    StructuralTransition,
    TransitionRecorder,
    record_runtime_step,
)
from .transition_record import (
    TransitionRecord,
    TransitionProvenance,
    AuthorizationDecision,
    AuthorityIdentity,
    StructuralStateSummary,
    DiagnosisSummary,
    CandidateSummary,
    CandidateSetSummary,
    PlannerMetadata,
    ComputeMetrics,
    make_record_id,
)
from .feature_extraction import (
    GlobalStructuralFeatures,
    LocalActionFeatures,
    StructuralFeatureVector,
    extract_global_features,
    extract_local_action_features,
)
from .dataset_generator import (
    DatasetGenerator,
    SplitDataset,
    DatasetImmutableMetadata,
    DATASET_SCHEMA_VERSION,
    GENERATOR_VERSION,
)
from .dataset_validator import (
    DatasetValidator,
    ValidationIssue,
    ValidationResult,
)
from .quality_report import (
    DataQualityReport,
    DistributionReport,
    CategoryDistribution,
    generate_quality_report,
)
from .dataset import (
    StructuralDataset,
    StructuralDatasetSchema,
    DatasetSplit,
    DatasetMetadata,
)
from .experiment_registry import (
    ExperimentRecord,
    ExperimentRegistry,
    ExperimentStatus,
)
from .reproducibility import (
    ReproducibilityConfig,
    RunFingerprint,
    seed_all,
    config_hash,
)
from .world_model import (
    WorldModelInterface,
    OutcomeModelInterface,
    StructuralStateEncoderInterface,
    ModelPrediction,
    ModelTrustReport,
)

__all__ = [
    # graph families
    "GraphFamilySplit", "FrozenGraphFamilyRegistry",
    "FROZEN_TRAIN_FAMILIES", "FROZEN_VALIDATION_FAMILIES",
    "FROZEN_HELD_OUT_FAMILIES", "get_frozen_registry",
    # metrics
    "V6Metric", "V6MetricReport",
    "AdaptationSpeedMetric", "PerformancePerComputeMetric",
    "OODGeneralizationMetric", "MutationCountMetric",
    "TopologyComplexityMetric", "FailureRateMetric",
    "CalibrationMetric",
    "adaptation_speed_metric", "performance_per_compute_metric",
    "ood_generalization_metric", "mutation_count_metric",
    "topology_complexity_metric", "failure_rate_metric",
    "calibration_metric", "aggregate_metrics",
    # baselines
    "BaselineRunner", "FixedTopologyBaseline", "RandomRewiringBaseline",
    "GreedyBaseline", "CurvatureOnlyBaseline", "FoSRBaseline",
    "BORFBaseline", "EffectiveResistanceBaseline",
    "OneStepCounterfactualBaseline", "MPCBaseline", "MPCWithIGBaseline",
    "FullV511Baseline", "ALL_V6_BASELINES",
    # harness
    "V6BenchmarkHarness", "BenchmarkRunResult",
    # transition recorder (exp1)
    "StructuralTransition", "TransitionRecorder", "record_runtime_step",
    # transition record (exp2)
    "TransitionRecord", "TransitionProvenance", "AuthorizationDecision",
    "AuthorityIdentity", "StructuralStateSummary", "DiagnosisSummary",
    "CandidateSummary", "CandidateSetSummary", "PlannerMetadata",
    "ComputeMetrics", "make_record_id",
    # feature extraction (exp2)
    "GlobalStructuralFeatures", "LocalActionFeatures",
    "StructuralFeatureVector",
    "extract_global_features", "extract_local_action_features",
    # dataset generator (exp2)
    "DatasetGenerator", "SplitDataset", "DatasetImmutableMetadata",
    "DATASET_SCHEMA_VERSION", "GENERATOR_VERSION",
    # dataset validator (exp2)
    "DatasetValidator", "ValidationIssue", "ValidationResult",
    # quality report (exp2)
    "DataQualityReport", "DistributionReport", "CategoryDistribution",
    "generate_quality_report",
    # dataset (exp1)
    "StructuralDataset", "StructuralDatasetSchema",
    "DatasetSplit", "DatasetMetadata",
    # experiment registry
    "ExperimentRecord", "ExperimentRegistry", "ExperimentStatus",
    # reproducibility
    "ReproducibilityConfig", "RunFingerprint", "seed_all", "config_hash",
    # world model
    "WorldModelInterface", "OutcomeModelInterface",
    "StructuralStateEncoderInterface", "ModelPrediction", "ModelTrustReport",
    # encoders (exp3)
    "encoders",
    # models (exp4)
    "models",
]
