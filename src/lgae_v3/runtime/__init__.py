"""v5.10 canonical runtime package.

One authoritative end-to-end governed cycle. The runtime orchestrates
existing engines (LGAEEngine, StructuralLearningLoop, StructuralReasoningLoop,
StructuralMPC, EvidenceLedger, receipts) and does not re-implement any
algorithm. Learned models propose; deterministic governance authorizes;
evidence proves.
"""
from __future__ import annotations

from .runtime_config import RuntimeConfig, RuntimeMode
from .runtime_state import RuntimeSnapshot
from .runtime_events import RuntimePhase, RuntimeEvent
from .runtime_result import RuntimeStepResult
from .state_identity import AuthorityStateIdentity
from .authority import (
    AuthorityRole, AuthorityBoundary, AuthoritativeStateGuard,
    CommitChannel, DEFAULT_BOUNDARIES,
)
from .cache_coherence import MutationImpact, CacheRegistry, depends_on, declared_dependencies
from .adaptive_diagnostics import (
    DiagnosticLevel, DiagnosticEscalationPolicy, DiagnosticResult,
    DiagnosticCascade,
)
from .certification import (
    CertificationLevel, CertificationResult, CertificationError,
    minimum_level_for, meets_requirement,
)
from .candidates import (
    Candidate, CandidateUnion, candidate_id, build_candidate_union,
)
from .candidate_retrieval import (
    RetrievalMetrics, RetrievalBenchmark, evaluate_retrieval, brute_force_top_k,
)
from .baseline_competition import (
    BaselineCompetition, CompetitionReport, PolicyResult,
    select_by_scores, learned_policy_from_scores,
)
from .observability import (
    MetricsSink, Counter, Gauge, Histogram, read_jsonl,
)
from .qualification import (
    SafetyCheckStatus, SafetyCheckResult, SafetyQualificationReport,
    SafetyGateError, run_safety_qualification, assert_safety_gate,
)
from .scientific_qualification import (
    ScientificMetric, ScientificQualificationReport,
    ScientificGateError, assert_scientific_gate,
)
from .performance_qualification import (
    ScaleTier, TIER_NODE_COUNTS, MeasurementStatus, TierMeasurement,
    PerformanceQualificationReport, measure_tier, run_performance_qualification,
)
from .promotion import (
    PromotionLevel, GateStatus, PromotionReport,
    PromotionGateError, evaluate_promotion, assert_promotion,
)
from .model_registry import (
    ModelRecord, PromotionTransition, ModelRegistry,
)
from .decision_trace import TraceEntry, DecisionTrace
from .curriculum import (
    GraphFamily, CurriculumEntry, CurriculumGenerator, generate_graph,
)
from .ood_qualification import (
    OODEvaluationResult, OODQualificationReport, evaluate_ood, to_scientific_report,
)
from .mode_enforcement import ModeEnforcer, ProductionModeViolation
from .adversarial import (
    AdversarialOutcome, AdversarialTestResult, AdversarialTestReport,
    run_adversarial_tests,
)
from .profiling import PhaseTiming, ProfileReport, RuntimeProfiler
from .merkle_evidence import MerkleProof, MerkleTree, BatchEvidence, verify_proof
from .checkpointing import Checkpoint, CheckpointChain
from .wal import (
    WALRecordType, WALRecord, WALTransaction, WriteAheadLog, recover_transactions,
    replay_committed_transactions,
)
from .replayable_decisions import (
    DecisionRecord, DecisionLedger, build_decision_record, verify_replay,
)
from .real_graphs import (
    RealGraphBenchmark, RealGraphSpec, BENCHMARK_SPECS,
    load_benchmark, list_benchmarks,
)
from .sheaf import SheafConsistencyResult, certify_sheaf_consistency
from .manifold_action import (
    LieGroup, ManifoldAction, exponential_map, compose, inverse,
    make_so3_action, make_su2_action, make_gl_action,
)
from .information_gain import (
    InformationGainEstimate, InformationDirectedReport,
    ensemble_disagreement_ig, ucb_ig, posterior_variance_reduction_ig,
    compute_ig_correlation, select_information_directed,
)
from .epistemic_uncertainty import (
    EpistemicUncertaintyEstimate,
    compute_ensemble_uncertainty, compute_distance_penalty,
    estimate_epistemic_uncertainty, verify_ood_uncertainty_property,
)
from .uncertainty_calibration import (
    CalibrationMetrics, expected_calibration_error,
    negative_log_likelihood, brier_score,
    compute_calibration_metrics, is_well_calibrated,
)
from .structural_mpc import MPCPlan, MPCPlanner, plan_with_mpc
from .joint_action import (
    SubAction, JointStructuralAction, make_joint_action,
    joint_action_authority_level,
)
from .structural_credit import (
    CreditAssignment, direct_credit, feature_based_credit,
    temporal_credit, baseline_credit,
)
from .replay import ReplayTransition, ReplayBuffer
from .hard_negative_replay import (
    HardNegative, HardNegativeMiner, augment_buffer_with_hard_negatives,
)
from .offline_rl import (
    OfflineRLConfig, QNetwork, OfflineRLTrainer,
)
from .causal_credit import (
    CausalCreditAssignment, CausalCreditAssigner,
    average_causal_effect, credit_concentration,
)
from .graph_ops import (
    compute_degrees, get_neighbors, build_adjacency_matrix,
    connected_components, shortest_path_length, count_triangles,
    graph_diameter,
)
from .sparse_graph import (
    SparseGraph, build_sparse_graph, sparse_adjacency_matrix,
    sparse_to_edge_index,
)
from .gpu_path import (
    get_device, DeviceConfig, move_to_device, batched_message_passing,
    batched_candidate_scoring, batched_feature_computation,
    is_gpu_available, device_info,
)
from .batched_counterfactuals import (
    CounterfactualResult, batched_apply_actions, batched_compute_utilities,
    batched_counterfactual_eval, select_best_counterfactual,
)
from .canonical_runtime import LGAERuntime, UnauthorizedMutationError
from .transaction import (
    StructuralTransaction, GraphDelta, FiberDelta, GaugeDelta,
    TransactionValidationError, StaleTransactionError,
    AuthorizationBindingError, make_graph_transaction,
    make_fiber_transaction, make_gauge_transaction, make_joint_transaction,
)

__all__ = [
    "AuthorityStateIdentity",
    "RuntimeConfig",
    "RuntimeMode",
    "RuntimeSnapshot",
    "RuntimePhase",
    "RuntimeEvent",
    "RuntimeStepResult",
    "LGAERuntime",
    "UnauthorizedMutationError",
    "AuthorityRole",
    "AuthorityBoundary",
    "AuthoritativeStateGuard",
    "CommitChannel",
    "DEFAULT_BOUNDARIES",
    "MutationImpact",
    "CacheRegistry",
    "depends_on",
    "declared_dependencies",
    "DiagnosticLevel",
    "DiagnosticEscalationPolicy",
    "DiagnosticResult",
    "DiagnosticCascade",
    "CertificationLevel",
    "CertificationResult",
    "CertificationError",
    "minimum_level_for",
    "meets_requirement",
    "Candidate",
    "CandidateUnion",
    "candidate_id",
    "build_candidate_union",
    "RetrievalMetrics",
    "RetrievalBenchmark",
    "evaluate_retrieval",
    "brute_force_top_k",
    "BaselineCompetition",
    "CompetitionReport",
    "PolicyResult",
    "select_by_scores",
    "learned_policy_from_scores",
    "MetricsSink",
    "Counter",
    "Gauge",
    "Histogram",
    "read_jsonl",
    "SafetyCheckStatus",
    "SafetyCheckResult",
    "SafetyQualificationReport",
    "SafetyGateError",
    "run_safety_qualification",
    "assert_safety_gate",
    "ScientificMetric",
    "ScientificQualificationReport",
    "ScientificGateError",
    "assert_scientific_gate",
    "ScaleTier",
    "TIER_NODE_COUNTS",
    "MeasurementStatus",
    "TierMeasurement",
    "PerformanceQualificationReport",
    "measure_tier",
    "run_performance_qualification",
    "PromotionLevel",
    "GateStatus",
    "PromotionReport",
    "PromotionGateError",
    "evaluate_promotion",
    "assert_promotion",
    "ModelRecord",
    "PromotionTransition",
    "ModelRegistry",
    "TraceEntry",
    "DecisionTrace",
    "GraphFamily",
    "CurriculumEntry",
    "CurriculumGenerator",
    "generate_graph",
    "OODEvaluationResult",
    "OODQualificationReport",
    "evaluate_ood",
    "to_scientific_report",
    "ModeEnforcer",
    "ProductionModeViolation",
    "AdversarialOutcome",
    "AdversarialTestResult",
    "AdversarialTestReport",
    "run_adversarial_tests",
    "PhaseTiming",
    "ProfileReport",
    "RuntimeProfiler",
    "MerkleProof",
    "MerkleTree",
    "BatchEvidence",
    "verify_proof",
    "Checkpoint",
    "CheckpointChain",
    "WALRecordType",
    "WALRecord",
    "WALTransaction",
    "WriteAheadLog",
    "recover_transactions",
    "replay_committed_transactions",
    "DecisionRecord",
    "DecisionLedger",
    "build_decision_record",
    "verify_replay",
    "RealGraphBenchmark",
    "RealGraphSpec",
    "BENCHMARK_SPECS",
    "load_benchmark",
    "list_benchmarks",
    "SheafConsistencyResult",
    "certify_sheaf_consistency",
    "LieGroup",
    "ManifoldAction",
    "exponential_map",
    "compose",
    "inverse",
    "make_so3_action",
    "make_su2_action",
    "make_gl_action",
    "InformationGainEstimate",
    "InformationDirectedReport",
    "ensemble_disagreement_ig",
    "ucb_ig",
    "posterior_variance_reduction_ig",
    "compute_ig_correlation",
    "select_information_directed",
    "EpistemicUncertaintyEstimate",
    "compute_ensemble_uncertainty",
    "compute_distance_penalty",
    "estimate_epistemic_uncertainty",
    "verify_ood_uncertainty_property",
    "CalibrationMetrics",
    "expected_calibration_error",
    "negative_log_likelihood",
    "brier_score",
    "compute_calibration_metrics",
    "is_well_calibrated",
    "MPCPlan",
    "MPCPlanner",
    "plan_with_mpc",
    "SubAction",
    "JointStructuralAction",
    "make_joint_action",
    "joint_action_authority_level",
    "CreditAssignment",
    "direct_credit",
    "feature_based_credit",
    "temporal_credit",
    "baseline_credit",
    "ReplayTransition",
    "ReplayBuffer",
    "HardNegative",
    "HardNegativeMiner",
    "augment_buffer_with_hard_negatives",
    "OfflineRLConfig",
    "QNetwork",
    "OfflineRLTrainer",
    "CausalCreditAssignment",
    "CausalCreditAssigner",
    "average_causal_effect",
    "credit_concentration",
    "compute_degrees",
    "get_neighbors",
    "build_adjacency_matrix",
    "connected_components",
    "shortest_path_length",
    "count_triangles",
    "graph_diameter",
    "SparseGraph",
    "build_sparse_graph",
    "sparse_adjacency_matrix",
    "sparse_to_edge_index",
    "get_device",
    "DeviceConfig",
    "move_to_device",
    "batched_message_passing",
    "batched_candidate_scoring",
    "batched_feature_computation",
    "is_gpu_available",
    "device_info",
    "CounterfactualResult",
    "batched_apply_actions",
    "batched_compute_utilities",
    "batched_counterfactual_eval",
    "select_best_counterfactual",
    "StructuralTransaction",
    "GraphDelta",
    "FiberDelta",
    "GaugeDelta",
    "TransactionValidationError",
    "StaleTransactionError",
    "AuthorizationBindingError",
    "make_graph_transaction",
    "make_fiber_transaction",
    "make_gauge_transaction",
    "make_joint_transaction",
]
