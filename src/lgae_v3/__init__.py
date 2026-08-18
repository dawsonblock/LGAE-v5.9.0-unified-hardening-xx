"""LGAE-v3: geometry-governed self-evolving graph/latent controller."""
from .config import LGAEConfig, load_config, config_structural_hash, config_governance_hash, ProductionConfig, ResearchConfig
from .evolution import LGAEEngine
from .fibers import FixedWidthFiberLatent, FiberController, SOConnectionBank, project_to_so_d
from .operators import DualOperatorState, SparseDualOperatorState
from .types import (
    EdgeRole,
    GraphBuffers,
    make_graph_buffers,
    make_bucketed_graph_buffers,
    round_edge_capacity,
    MutationDecision,
    MutationResult,
    CertificationLevel,
)
from .training import (
    LGAETrainCore, train_step, padded_markov_edges, refresh_padded_markov_edges_,
    padded_markov_edges_with_slots, refresh_padded_markov_edges_with_slots_,
)
from .governor import GeometryGovernor
from .mutations import (
    AddEdge, ReweightEdge, ReweightAffinity, ReweightLength, CoupledReweight,
    PruneEdge, RicciFlowReweight, MutationCooldownTracker,
    StructuralMutation, GraphMutation,
    MutationAuthorityLevel, mutation_authority_level,
    mutation_to_spec, mutation_from_spec,
)
from .neighbor_index import (
    NeighborIndex, ExactChunkedKNN, KNNGraphResult,
    build_knn_graph, recall_at_k,
)
from .executive import (
    StructuralExecutive, ExecutiveNetwork, ActionProposal, StructuralObservation,
    StructuralAction, ACTION_LIST, ACTION_TO_IDX, NUM_ACTIONS,
)
from .uncertainty import (
    EnsembleUncertainty, ConformalCalibrator, UncertaintyEstimate,
    uncertainty_gated_decision,
)
from .credit import (
    MutationCreditTracker, MutationReceipt, MutationOutcome,
)
from .consolidation import (
    StabilityPlasticityController, FiberState, FiberLifecycleStage, CapacityBudget,
)
from .counterfactual import (
    StructuralCounterfactualEngine, CounterfactualResult,
)
from .structural_loop import (
    StructuralLearningLoop, StructuralLoopResult,
)
from .action_bridge import (
    action_to_mutation, certify_action_through_governor, ActionBridgeResult,
)
from .dynamic_gauge import (
    DynamicGaugeNetwork, DynamicGaugeBank, StaticGaugeAdapter,
    gauge_transport, gauge_alignment_loss,
)
from .timescales import (
    Timescale, TimescaleSchedule, AdaptationState, MultiTimescaleController,
)
from .sheaf_diffusion import (
    sheaf_laplacian_diffusion, sheaf_adjacency_diffusion,
    gated_sheaf_diffusion, agreement_gate, compare_diffusion_methods,
    gauge_orthogonality_penalty,
)
from .ann_index import (
    ANNNeighborIndex, FAISSIndex, RandomProjectionANN, HNSWIndexNumpy,
)
from .production_dynamics import (
    CurvatureHysteresisController, LatentEquilibriumBarrier, GraphHashBaseline, GraphFeatureBaseline, compute_graph_features,
)
from .runtime.state_identity import AuthorityStateIdentity
from .transactions import GraphTransaction, DeltaGraphTransaction, graph_transaction, journaled_graph_transaction
from .cache_coherence import (
    ChangeKind, GraphCommitEvent, CommitEventBus, CacheDependency, GenerationStampedCache,
    GraphReadCoordinator, GraphReadView, ReadEpochToken, StaleReadError, run_consistent_read,
)

from .reasoning import (
    ConcreteAction, CandidateValue, CounterfactualOutcome, ReasoningPlan,
    GraphStateEncoder, CandidateQNetwork, CandidateGenerator,
    CounterfactualReplayBuffer, StructuralReasoningExecutive,
    CounterfactualFactory, certify_ranked_candidates,
)
from .reasoning_loop import StructuralReasoningLoop, StructuralReasoningStep
from .causal_edges import (
    EdgeSemantics, CausalEdge, CausalEdgeRegistry, infer_causality_from_temporal,
)
from .receipts import (
    mutation_receipt, append_receipt, verify_receipt_chain,
    ed25519_available, generate_keypair, sign_receipt, verify_receipt_signature,
)
from .topology import (
    graphbuffers_to_networkx, topology_signature, topology_drift,
    topology_signature_buffers, find_bridges_buffers,
)
from .mpc import StructuralMPC, MPCPlanResult
from .equivariant import (
    EquivariantExecutiveNetwork, MessagePassingLayer,
    graphbuffers_to_edge_index, permutation_invariance_test,
)
from .deterministic import DeterministicRNGContext, deterministic_mode, derive_seed
from .reproducibility import ReproducibilityInfo, qualification_id
from .benchmark import ACTION_ORDER, ACTION_TO_INDEX, canonical_action

from .hypergraph import (
    Hyperedge, HypergraphBuffers, hypergraph_laplacian_diffusion,
    clique_expansion, star_expansion,
)

__all__ = [
    "AuthorityStateIdentity",
    "LGAEConfig", "load_config", "config_structural_hash", "config_governance_hash",
    "LGAEEngine", "FixedWidthFiberLatent", "FiberController", "SOConnectionBank", "project_to_so_d",
    "DualOperatorState", "SparseDualOperatorState", "EdgeRole", "GraphBuffers", "make_graph_buffers", "make_bucketed_graph_buffers", "round_edge_capacity",
    "MutationDecision", "MutationResult",
    "AddEdge", "ReweightEdge", "ReweightAffinity", "ReweightLength", "CoupledReweight",
    "PruneEdge", "RicciFlowReweight", "MutationCooldownTracker",
    "mutation_to_spec", "mutation_from_spec",
    "LGAETrainCore", "train_step", "padded_markov_edges", "refresh_padded_markov_edges_", "padded_markov_edges_with_slots", "refresh_padded_markov_edges_with_slots_",
    "gauge_orthogonality_penalty", "ANNNeighborIndex", "FAISSIndex", "RandomProjectionANN",
    "CurvatureHysteresisController", "LatentEquilibriumBarrier", "GraphHashBaseline",
    "GraphTransaction", "DeltaGraphTransaction", "graph_transaction", "journaled_graph_transaction",
    "ChangeKind", "GraphCommitEvent", "CommitEventBus", "CacheDependency", "GenerationStampedCache",
    "GraphReadCoordinator", "GraphReadView", "ReadEpochToken", "StaleReadError", "run_consistent_read",
    "ProductionConfig", "ResearchConfig", "CertificationLevel",
    "MutationAuthorityLevel", "mutation_authority_level",
    "GraphFeatureBaseline", "compute_graph_features",
    "mutation_receipt", "append_receipt", "verify_receipt_chain",
    "ed25519_available", "generate_keypair", "sign_receipt", "verify_receipt_signature",
    "graphbuffers_to_networkx", "topology_signature", "topology_drift",
    "topology_signature_buffers", "find_bridges_buffers",
    "StructuralMPC", "MPCPlanResult",
    "EquivariantExecutiveNetwork", "MessagePassingLayer", "graphbuffers_to_edge_index", "permutation_invariance_test",
    "DeterministicRNGContext", "deterministic_mode", "derive_seed",
    "ReproducibilityInfo", "qualification_id", "ACTION_ORDER", "ACTION_TO_INDEX", "canonical_action",
]
from .version import VERSION as __version__

# v5.5 evidence-grounded reasoning memory
from .evidence import EvidenceLedger, EvidenceRecord, EVIDENCE_SCHEMA
from .memory import StructuralExperienceMemory, MemoryKind, MemoryNode, MemoryEdge, MemoryMatch
from .reasoning_graph import ReasoningGraph, ReasoningNode, ReasoningEvidence, ReasoningRun

# v5.7 adaptive geometry runtime
from .adaptive_geometry import (
    OperatorDependencyFootprint, DependencyRegistry, DEFAULT_OPERATOR_FOOTPRINTS,
    OrthogonalityHealth, orthogonality_error, monitor_orthogonality,
    CurvatureStage, GeometryEstimate, CascadePolicy, CascadeResult, AdaptiveCurvatureCascade,
)

# v5.8 structural intelligence qualification
from .structural_intelligence import (
    StateGroupedReplayBuffer, EnsembleStructuralQ, StructuralIntelligenceExecutive,
    ProceduralGraphFactory, ProceduralCase, effective_resistance_matrix,
    effective_resistance_candidates, exact_candidate_deltas, candidate_regret,
    uncertainty_calibration, RegretResult,
    StructuralRegime, structural_regime_features, SpectralStratifiedReplayBuffer,
    RandomizedPriorEnsembleQ, ContrastiveCandidateRetriever, fosr_candidates,
    forman_flow_candidates, merge_candidate_channels, ConservativeStructuralExecutive,
)

# v5.8.2 scalable structural intelligence
from .structural_intelligence import (
    ANNCandidateRetriever, approximate_fosr_candidates, EpistemicScaleCalibrator, CalibrationResult,
    WLDeduplicatedSpectralReplayBuffer, wl_graph_hash, contextual_lcb_beta, ScalableStructuralExecutive,
)

# v5.8.4 joint structural action runtime optimization
from .joint_structural_action import (
    JointStructuralAction, JointCertificationResult, LowRankLieGaugeHead,
    JointStructuralGaugePolicy, certify_joint_structural_action,
    LocalizedCreditResult, connection_dirichlet_energy, localized_dirichlet_credit,
    commit_joint_connection, cayley_retraction, paired_restriction_maps,
    assemble_paired_connection_laplacian, two_sided_connection_dirichlet_energy,
)

# v5.10 canonical runtime
from .runtime import (
    LGAERuntime, RuntimeConfig, RuntimeMode, RuntimeSnapshot,
    RuntimePhase, RuntimeEvent, RuntimeStepResult,
)

# v5.10 governance (mutation authority policy)
from .governance import (
    AuthorityRequirement, MutationAuthorityPolicy, DEFAULT_AUTHORITY_POLICY,
    requirement_for, classify_mutation_authority,
)
