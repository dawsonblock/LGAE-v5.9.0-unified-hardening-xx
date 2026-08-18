"""v6.0-exp3: Structural state/action encoders.

A ladder of encoders from simple to sophisticated, forcing every
higher-complexity representation to beat the simpler one.

Encoders:
    0. MinimalControlEncoder — identity floor (family, n_nodes, n_edges, action)
    1. GlobalStateEncoder — 24-dim handcrafted global features
    2. LocalActionEncoder — 24+12=36-dim global+local
    3. SemanticActionEncoder — mutation-semantic encoding
    4. LocalSubgraphEncoder — k-hop neighborhood with canonical ordering
    5. GeometricEncoder — spectral/curvature/resistance features
    6. SpectralEncoder — deterministic spectral embedding
    7. SmallLearnedGraphEncoder — 2-3 layer message passing
    8. HybridEncoder — concatenation of all representations

Lifecycle: UNFIT → FITTED_TRAIN → FROZEN

Once frozen, validation and held-out data cannot modify normalization.
"""
from __future__ import annotations

from .protocol import (
    EncodedState, EncodedAction, StateActionRepresentation,
    StructuralStateEncoder, StructuralActionEncoder, StateActionEncoder,
    EncoderLifecycle, ActionEncodingSchema, DEFAULT_ACTION_SCHEMA,
    feature_hash, safe_log1p, safe_normalize, ensure_finite,
)
from .normalization import (
    NormalizationStatistics, NormalizationState,
    FrozenNormalizationError, HeldOutFittingError,
)
from .minimal import MinimalControlEncoder
from .global_features import GlobalStateEncoder
from .local_action import LocalActionEncoder
from .semantic_action import SemanticActionEncoder
from .local_subgraph import LocalSubgraphEncoder
from .geometric import GeometricEncoder
from .spectral import SpectralEncoder
from .learned_graph import SmallLearnedGraphEncoder
from .hybrid import HybridEncoder
from .registry import EncoderRegistry, EncoderProvenance
from .probes import (
    ProbeResult, EncoderProbeReport, LogisticProbe, LinearProbe,
    run_probe_benchmark,
)
from .collision import CollisionReport, analyze_collisions
from .complexity import (
    ComplexityMetrics, RepresentationComparison,
    measure_encoding_latency, compute_effectiveness, compare_encoders,
)

__all__ = [
    # Protocol
    "EncodedState", "EncodedAction", "StateActionRepresentation",
    "StructuralStateEncoder", "StructuralActionEncoder", "StateActionEncoder",
    "EncoderLifecycle", "ActionEncodingSchema", "DEFAULT_ACTION_SCHEMA",
    "feature_hash", "safe_log1p", "safe_normalize", "ensure_finite",
    # Normalization
    "NormalizationStatistics", "NormalizationState",
    "FrozenNormalizationError", "HeldOutFittingError",
    # Encoders
    "MinimalControlEncoder", "GlobalStateEncoder", "LocalActionEncoder",
    "SemanticActionEncoder", "LocalSubgraphEncoder", "GeometricEncoder",
    "SpectralEncoder", "SmallLearnedGraphEncoder", "HybridEncoder",
    # Registry
    "EncoderRegistry", "EncoderProvenance",
    # Probes
    "ProbeResult", "EncoderProbeReport", "LogisticProbe", "LinearProbe",
    "run_probe_benchmark",
    # Collision
    "CollisionReport", "analyze_collisions",
    # Complexity
    "ComplexityMetrics", "RepresentationComparison",
    "measure_encoding_latency", "compute_effectiveness", "compare_encoders",
]
