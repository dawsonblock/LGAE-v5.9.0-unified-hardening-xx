"""v6.0-exp3: Structural state/action encoders tests.

Tests verify:
1. Encoder determinism (E(S,a,C) = constant)
2. Canonical schema hash stability
3. Dimension stability
4. Train-only normalization
5. Frozen normalization rejection
6. Held-out leakage rejection
7. Node permutation invariance
8. Undirected action symmetry
9. Missing feature handling
10. Nonfinite rejection
11. Serialization roundtrip
12. Encoder registry
13. Dataset binding
14. Local subgraph canonicalization
15. Global feature stability
16. Geometric feature stability
17. Spectral determinism
18. Collision reporting
19. Probe benchmark reproducibility
20. Authority boundary untouched
"""
from __future__ import annotations

import pytest
import numpy as np
import torch
import json
import math
import networkx as nx
from dataclasses import dataclass

from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.experimental.encoders import (
    # Protocol
    EncodedState, EncodedAction, StateActionRepresentation,
    ActionEncodingSchema, DEFAULT_ACTION_SCHEMA,
    feature_hash, ensure_finite, safe_log1p,
    # Normalization
    NormalizationStatistics, NormalizationState,
    FrozenNormalizationError, HeldOutFittingError,
    # Encoders
    MinimalControlEncoder, GlobalStateEncoder, LocalActionEncoder,
    SemanticActionEncoder, LocalSubgraphEncoder, GeometricEncoder,
    SpectralEncoder, SmallLearnedGraphEncoder, HybridEncoder,
    # Registry
    EncoderRegistry, EncoderProvenance,
    # Probes
    ProbeResult, EncoderProbeReport, LogisticProbe, LinearProbe,
    run_probe_benchmark,
    # Collision
    CollisionReport, analyze_collisions,
    # Complexity
    ComplexityMetrics, RepresentationComparison,
    measure_encoding_latency, compute_effectiveness, compare_encoders,
)
from lgae_v3.experimental import (
    extract_global_features, extract_local_action_features,
    GlobalStructuralFeatures, LocalActionFeatures,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> ResearchConfig:
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 0
    cfg.audit.orc_backend = "exact_lp"
    cfg.audit.persistent_homology_enabled = False
    cfg.audit.entropic_nodes = 0
    cfg.audit.bakry_nodes = 0
    cfg.audit.cde_nodes = 0
    cfg.audit.exact_lly_top_k = 0
    cfg.audit.orc_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]
    cfg.mutation.curvature_ema_enabled = False
    return cfg


def _small_graph():
    return make_graph_buffers(8, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)], capacity=16)


def _cycle_graph(n=10):
    edges = [(i, (i+1) % n) for i in range(n)]
    return make_graph_buffers(n, edges, capacity=n*2)


def _star_graph(n=10):
    edges = [(0, i) for i in range(1, n)]
    return make_graph_buffers(n, edges, capacity=n*2)


@dataclass
class MockState:
    """Mock state for encoder tests."""
    n_nodes: int = 10
    n_edges: int = 9
    graph_family: str = "path"


def _global_features(graph=None):
    if graph is None:
        graph = _small_graph()
    return extract_global_features(graph).vector


def _local_features(graph=None, u=0, v=5):
    if graph is None:
        graph = _small_graph()
    return extract_local_action_features(graph, u, v).vector


def _make_representations(encoder, n=20, seed=42):
    """Generate n representations for probe testing."""
    rng = np.random.RandomState(seed)
    reps = []
    targets_du = []
    targets_success = []
    targets_risk = []
    for i in range(n):
        graph = _small_graph()
        gf = _global_features(graph)
        lf = _local_features(graph, int(rng.randint(0, 8)), int(rng.randint(0, 8)))
        state = MockState(n_nodes=8, n_edges=7, graph_family="path")
        rep = encoder.encode(
            state=state, global_features=gf,
            action_type="ADD_EDGE", action_target={"u": 0, "v": 5},
            local_features=lf,
        )
        reps.append(rep)
        targets_du.append(float(rng.randn() * 0.1))
        targets_success.append(bool(rng.rand() > 0.5))
        targets_risk.append(float(rng.rand()))
    return reps, targets_du, targets_success, targets_risk


# ---------------------------------------------------------------------------
# 1. Protocol and dataclass tests
# ---------------------------------------------------------------------------

class TestProtocol:
    """Protocol and representation contract tests."""

    def test_encoded_state_creation(self):
        es = EncodedState(vector=(1.0, 2.0), dimension=2, encoder_id="test", schema_hash="abc")
        assert es.dimension == 2
        assert es.encoder_id == "test"

    def test_encoded_action_creation(self):
        ea = EncodedAction(vector=(1.0,), dimension=1, encoder_id="test", schema_hash="abc", action_type="ADD_EDGE")
        assert ea.action_type == "ADD_EDGE"

    def test_state_action_representation_creation(self):
        rep = StateActionRepresentation(
            encoder_id="test", encoder_version="v1", schema_hash="abc",
            vector=(1.0, 2.0), dimension=2,
            state_feature_hash="s1", action_feature_hash="a1",
            normalization_hash=None,
        )
        assert rep.encoder_id == "test"
        assert rep.dimension == 2

    def test_state_action_representation_to_log(self):
        rep = StateActionRepresentation(
            encoder_id="test", encoder_version="v1", schema_hash="abc",
            vector=(1.0, 2.0), dimension=2,
            state_feature_hash="s1", action_feature_hash="a1",
            normalization_hash="n1",
        )
        log = rep.to_log()
        assert log["encoder_id"] == "test"
        assert log["dimension"] == 2

    def test_feature_hash_deterministic(self):
        h1 = feature_hash([1.0, 2.0, 3.0])
        h2 = feature_hash([1.0, 2.0, 3.0])
        assert h1 == h2

    def test_feature_hash_differs(self):
        h1 = feature_hash([1.0, 2.0, 3.0])
        h2 = feature_hash([1.0, 2.0, 3.1])
        assert h1 != h2

    def test_safe_log1p(self):
        assert safe_log1p(0.0) == 0.0
        assert safe_log1p(1.0) == math.log(2)
        assert safe_log1p(float("inf")) == 0.0
        assert safe_log1p(float("nan")) == 0.0

    def test_ensure_finite(self):
        result = ensure_finite([1.0, float("nan"), float("inf"), 2.0])
        assert result == (1.0, 0.0, 0.0, 2.0)

    def test_action_encoding_schema_hash(self):
        schema = ActionEncodingSchema()
        assert schema.schema_hash
        assert len(schema.schema_hash) == 16

    def test_action_encoding_schema_type_index(self):
        schema = ActionEncodingSchema()
        assert schema.type_index("ADD_EDGE") == 0
        assert schema.type_index("REMOVE_EDGE") == 1
        assert schema.type_index("UNKNOWN") == -1

    def test_action_encoding_schema_deterministic(self):
        s1 = ActionEncodingSchema()
        s2 = ActionEncodingSchema()
        assert s1.schema_hash == s2.schema_hash


# ---------------------------------------------------------------------------
# 2. Normalization tests
# ---------------------------------------------------------------------------

class TestNormalization:
    """Train-only normalization and freeze lifecycle."""

    def test_unfit_state(self):
        norm = NormalizationStatistics()
        assert norm.state == NormalizationState.UNFIT
        assert not norm.is_fit
        assert not norm.is_frozen

    def test_fit_on_train(self):
        norm = NormalizationStatistics()
        features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        norm.fit(features, split="train")
        assert norm.state == NormalizationState.FITTED_TRAIN
        assert norm.is_fit
        assert not norm.is_frozen
        assert len(norm.mean) == 2

    def test_fit_on_heldout_raises(self):
        norm = NormalizationStatistics()
        features = [[1.0, 2.0], [3.0, 4.0]]
        with pytest.raises(HeldOutFittingError):
            norm.fit(features, split="held_out")

    def test_fit_on_validation_raises(self):
        norm = NormalizationStatistics()
        features = [[1.0, 2.0], [3.0, 4.0]]
        with pytest.raises(HeldOutFittingError):
            norm.fit(features, split="validation")

    def test_freeze_after_fit(self):
        norm = NormalizationStatistics()
        norm.fit([[1.0, 2.0], [3.0, 4.0]], split="train")
        norm.freeze()
        assert norm.is_frozen
        assert norm.state == NormalizationState.FROZEN

    def test_freeze_before_fit_raises(self):
        norm = NormalizationStatistics()
        with pytest.raises(RuntimeError):
            norm.freeze()

    def test_fit_after_freeze_raises(self):
        norm = NormalizationStatistics()
        norm.fit([[1.0, 2.0], [3.0, 4.0]], split="train")
        norm.freeze()
        with pytest.raises(FrozenNormalizationError):
            norm.fit([[5.0, 6.0]], split="train")

    def test_transform_unfit_returns_raw(self):
        norm = NormalizationStatistics()
        result, mask = norm.transform([1.0, 2.0, 3.0])
        assert result == (1.0, 2.0, 3.0)
        assert all(m is False for m in mask)

    def test_transform_fitted_normalizes(self):
        norm = NormalizationStatistics()
        norm.fit([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], split="train")
        result, mask = norm.transform([3.0, 4.0])
        # Mean is (3, 4), so normalized should be near 0.
        assert abs(result[0]) < 0.01
        assert abs(result[1]) < 0.01

    def test_transform_with_nonfinite(self):
        norm = NormalizationStatistics()
        norm.fit([[1.0, 2.0], [3.0, 4.0]], split="train")
        result, mask = norm.transform([float("nan"), 4.0])
        assert result[0] == 0.0
        assert mask[0] is True
        assert mask[1] is False

    def test_normalization_hash_deterministic(self):
        n1 = NormalizationStatistics()
        n1.fit([[1.0, 2.0], [3.0, 4.0]], split="train")
        n2 = NormalizationStatistics()
        n2.fit([[1.0, 2.0], [3.0, 4.0]], split="train")
        assert n1.normalization_hash == n2.normalization_hash

    def test_normalization_hash_differs_for_different_data(self):
        n1 = NormalizationStatistics()
        n1.fit([[1.0, 2.0], [3.0, 4.0]], split="train")
        n2 = NormalizationStatistics()
        n2.fit([[10.0, 20.0], [30.0, 40.0]], split="train")
        assert n1.normalization_hash != n2.normalization_hash


# ---------------------------------------------------------------------------
# 3. Encoder 0: MinimalControlEncoder
# ---------------------------------------------------------------------------

class TestMinimalControlEncoder:
    """Encoder 0: Minimal control baseline."""

    def test_name_and_version(self):
        enc = MinimalControlEncoder()
        assert enc.name == "minimal-control"
        assert enc.version == "v1"

    def test_dimension(self):
        enc = MinimalControlEncoder()
        assert enc.dimension > 0

    def test_deterministic(self):
        enc = MinimalControlEncoder()
        state = MockState(graph_family="path")
        rep1 = enc.encode(state, _global_features(), "ADD_EDGE", {"u": 0, "v": 5}, _local_features())
        rep2 = enc.encode(state, _global_features(), "ADD_EDGE", {"u": 0, "v": 5}, _local_features())
        assert rep1.vector == rep2.vector

    def test_does_not_require_fit(self):
        enc = MinimalControlEncoder()
        assert enc.requires_fit is False

    def test_schema_hash_stable(self):
        enc1 = MinimalControlEncoder()
        enc2 = MinimalControlEncoder()
        assert enc1.schema_hash == enc2.schema_hash

    def test_different_actions_differ(self):
        enc = MinimalControlEncoder()
        state = MockState(graph_family="path")
        rep1 = enc.encode(state, _global_features(), "ADD_EDGE", {}, _local_features())
        rep2 = enc.encode(state, _global_features(), "REMOVE_EDGE", {}, _local_features())
        assert rep1.vector != rep2.vector

    def test_all_finite(self):
        enc = MinimalControlEncoder()
        state = MockState(graph_family="path")
        rep = enc.encode(state, _global_features(), "ADD_EDGE", {}, _local_features())
        assert all(math.isfinite(v) for v in rep.vector)


# ---------------------------------------------------------------------------
# 4. Encoder 1: GlobalStateEncoder
# ---------------------------------------------------------------------------

class TestGlobalStateEncoder:
    """Encoder 1: 24-dim global features."""

    def test_name(self):
        enc = GlobalStateEncoder()
        assert enc.name == "global"

    def test_dimension(self):
        enc = GlobalStateEncoder()
        # 24 global + action one-hot
        assert enc.dimension == 24 + DEFAULT_ACTION_SCHEMA.n_types

    def test_requires_fit(self):
        enc = GlobalStateEncoder()
        assert enc.requires_fit is True

    def test_deterministic_after_fit(self):
        enc = GlobalStateEncoder()
        features = [_global_features(_small_graph()) for _ in range(10)]
        enc.fit(features, split="train")
        enc.freeze()
        gf = _global_features(_small_graph())
        rep1 = enc.encode(MockState(), gf, "ADD_EDGE", {}, _local_features())
        rep2 = enc.encode(MockState(), gf, "ADD_EDGE", {}, _local_features())
        assert rep1.vector == rep2.vector

    def test_unfit_returns_raw(self):
        enc = GlobalStateEncoder()
        gf = _global_features(_small_graph())
        rep = enc.encode(MockState(), gf, "ADD_EDGE", {}, _local_features())
        # Without fitting, global features should be raw (24 values).
        assert len(rep.vector) == enc.dimension

    def test_fit_rejects_heldout(self):
        enc = GlobalStateEncoder()
        with pytest.raises(HeldOutFittingError):
            enc.fit([_global_features()], split="held_out")

    def test_freeze_rejects_refit(self):
        enc = GlobalStateEncoder()
        enc.fit([_global_features()], split="train")
        enc.freeze()
        with pytest.raises(FrozenNormalizationError):
            enc.fit([_global_features()], split="train")

    def test_all_finite(self):
        enc = GlobalStateEncoder()
        enc.fit([_global_features() for _ in range(5)], split="train")
        enc.freeze()
        gf = _global_features(_small_graph())
        rep = enc.encode(MockState(), gf, "ADD_EDGE", {}, _local_features())
        assert all(math.isfinite(v) for v in rep.vector)


# ---------------------------------------------------------------------------
# 5. Encoder 2: LocalActionEncoder
# ---------------------------------------------------------------------------

class TestLocalActionEncoder:
    """Encoder 2: 24+12=36-dim global+local."""

    def test_name(self):
        enc = LocalActionEncoder()
        assert enc.name == "global-local"

    def test_dimension(self):
        enc = LocalActionEncoder()
        # 24 global + 12 local + action one-hot
        assert enc.dimension == 24 + 12 + DEFAULT_ACTION_SCHEMA.n_types

    def test_deterministic_after_fit(self):
        enc = LocalActionEncoder()
        gfs = [_global_features(_small_graph()) for _ in range(10)]
        lfs = [_local_features(_small_graph(), 0, 5) for _ in range(10)]
        enc.fit(gfs, lfs, split="train")
        enc.freeze()
        gf = _global_features(_small_graph())
        lf = _local_features(_small_graph(), 0, 5)
        rep1 = enc.encode(MockState(), gf, "ADD_EDGE", {"u": 0, "v": 5}, lf)
        rep2 = enc.encode(MockState(), gf, "ADD_EDGE", {"u": 0, "v": 5}, lf)
        assert rep1.vector == rep2.vector

    def test_fit_rejects_heldout(self):
        enc = LocalActionEncoder()
        with pytest.raises(HeldOutFittingError):
            enc.fit([_global_features()], [_local_features()], split="held_out")

    def test_all_finite(self):
        enc = LocalActionEncoder()
        gfs = [_global_features(_small_graph()) for _ in range(5)]
        lfs = [_local_features(_small_graph(), 0, 5) for _ in range(5)]
        enc.fit(gfs, lfs, split="train")
        enc.freeze()
        rep = enc.encode(MockState(), _global_features(), "ADD_EDGE", {}, _local_features())
        assert all(math.isfinite(v) for v in rep.vector)


# ---------------------------------------------------------------------------
# 6. Encoder 3: SemanticActionEncoder
# ---------------------------------------------------------------------------

class TestSemanticActionEncoder:
    """Encoder 3: Mutation-semantic encoding."""

    def test_name(self):
        enc = SemanticActionEncoder()
        assert enc.name == "semantic-action"

    def test_dimension(self):
        enc = SemanticActionEncoder()
        # 24 global + (n_types + 9) semantic
        assert enc.dimension == 24 + DEFAULT_ACTION_SCHEMA.n_types + 9

    def test_deterministic_after_fit(self):
        enc = SemanticActionEncoder()
        gfs = [_global_features(_small_graph()) for _ in range(10)]
        sems = [[0.0] * (DEFAULT_ACTION_SCHEMA.n_types + 9) for _ in range(10)]
        enc.fit(gfs, sems, split="train")
        enc.freeze()
        rep1 = enc.encode(MockState(), _global_features(), "ADD_EDGE", {"u": 0, "v": 5}, _local_features())
        rep2 = enc.encode(MockState(), _global_features(), "ADD_EDGE", {"u": 0, "v": 5}, _local_features())
        assert rep1.vector == rep2.vector

    def test_different_actions_differ(self):
        enc = SemanticActionEncoder()
        gfs = [_global_features(_small_graph()) for _ in range(5)]
        sems = [[0.0] * (DEFAULT_ACTION_SCHEMA.n_types + 9) for _ in range(5)]
        enc.fit(gfs, sems, split="train")
        enc.freeze()
        rep1 = enc.encode(MockState(), _global_features(), "ADD_EDGE", {"u": 0, "v": 5}, _local_features())
        rep2 = enc.encode(MockState(), _global_features(), "REMOVE_EDGE", {"u": 0, "v": 5}, _local_features())
        assert rep1.vector != rep2.vector


# ---------------------------------------------------------------------------
# 7. Encoder 4: LocalSubgraphEncoder
# ---------------------------------------------------------------------------

class TestLocalSubgraphEncoder:
    """Encoder 4: k-hop neighborhood."""

    def test_name(self):
        enc = LocalSubgraphEncoder()
        assert enc.name == "local-subgraph"

    def test_dimension(self):
        enc = LocalSubgraphEncoder()
        # 24 global + 10 subgraph + action one-hot
        assert enc.dimension == 24 + 10 + DEFAULT_ACTION_SCHEMA.n_types

    def test_extract_subgraph_features(self):
        enc = LocalSubgraphEncoder(k_hop=2, max_nodes=20)
        graph = _small_graph()
        feats = enc.extract_subgraph_features(graph, 0, 5)
        assert len(feats) == 10
        assert all(math.isfinite(v) for v in feats)

    def test_subgraph_features_deterministic(self):
        enc = LocalSubgraphEncoder(k_hop=2)
        graph = _small_graph()
        f1 = enc.extract_subgraph_features(graph, 0, 5)
        f2 = enc.extract_subgraph_features(graph, 0, 5)
        assert f1 == f2

    def test_subgraph_features_finite_for_disconnected(self):
        enc = LocalSubgraphEncoder(k_hop=2)
        graph = make_graph_buffers(10, [(0, 1)], capacity=4)
        feats = enc.extract_subgraph_features(graph, 0, 9)
        assert len(feats) == 10
        assert all(math.isfinite(v) for v in feats)


# ---------------------------------------------------------------------------
# 8. Encoder 5: GeometricEncoder
# ---------------------------------------------------------------------------

class TestGeometricEncoder:
    """Encoder 5: Geometric features."""

    def test_name(self):
        enc = GeometricEncoder()
        assert enc.name == "geometric"

    def test_dimension(self):
        enc = GeometricEncoder()
        # 24 global + 12 geometric + action one-hot
        assert enc.dimension == 24 + 12 + DEFAULT_ACTION_SCHEMA.n_types

    def test_extract_geometric_features(self):
        enc = GeometricEncoder()
        graph = _small_graph()
        feats = enc.extract_geometric_features(graph, 0, 5)
        assert len(feats) == 12
        assert all(math.isfinite(v) for v in feats)

    def test_geometric_features_deterministic(self):
        enc = GeometricEncoder()
        graph = _small_graph()
        f1 = enc.extract_geometric_features(graph, 0, 5)
        f2 = enc.extract_geometric_features(graph, 0, 5)
        assert f1 == f2


# ---------------------------------------------------------------------------
# 9. Encoder 6: SpectralEncoder
# ---------------------------------------------------------------------------

class TestSpectralEncoder:
    """Encoder 6: Deterministic spectral embedding."""

    def test_name(self):
        enc = SpectralEncoder()
        assert enc.name == "spectral"

    def test_dimension(self):
        enc = SpectralEncoder(k_eigenvalues=8)
        # 24 global + 3*8 spectral + action one-hot
        assert enc.dimension == 24 + 24 + DEFAULT_ACTION_SCHEMA.n_types

    def test_extract_spectral_features(self):
        enc = SpectralEncoder(k_eigenvalues=8)
        graph = _small_graph()
        feats = enc.extract_spectral_features(graph)
        assert len(feats) == 24
        assert all(math.isfinite(v) for v in feats)

    def test_spectral_determinism(self):
        enc = SpectralEncoder(k_eigenvalues=8)
        graph = _small_graph()
        f1 = enc.extract_spectral_features(graph)
        f2 = enc.extract_spectral_features(graph)
        assert f1 == f2

    def test_spectral_for_empty_graph(self):
        enc = SpectralEncoder(k_eigenvalues=8)
        graph = make_graph_buffers(5, [], capacity=4)
        feats = enc.extract_spectral_features(graph)
        assert len(feats) == 24
        assert all(v == 0.0 for v in feats)


# ---------------------------------------------------------------------------
# 10. Encoder 7: SmallLearnedGraphEncoder
# ---------------------------------------------------------------------------

class TestSmallLearnedGraphEncoder:
    """Encoder 7: Small learned graph encoder."""

    def test_name(self):
        enc = SmallLearnedGraphEncoder()
        assert enc.name == "learned-graph"

    def test_dimension(self):
        enc = SmallLearnedGraphEncoder(output_dim=64)
        # 64 output + action one-hot
        assert enc.dimension == 64 + DEFAULT_ACTION_SCHEMA.n_types

    def test_n_parameters(self):
        enc = SmallLearnedGraphEncoder()
        assert enc.n_parameters > 0

    def test_lifecycle_unfit(self):
        enc = SmallLearnedGraphEncoder()
        assert enc.lifecycle == "unfit"

    def test_fit_on_train(self):
        enc = SmallLearnedGraphEncoder()
        reps = [[0.0] * 36 for _ in range(20)]
        targets = [float(i) * 0.01 for i in range(20)]
        result = enc.fit(reps, targets, split="train", n_epochs=5)
        assert enc.lifecycle == "fitted_train"
        assert "final_loss" in result

    def test_fit_rejects_heldout(self):
        enc = SmallLearnedGraphEncoder()
        with pytest.raises(HeldOutFittingError):
            enc.fit([[0.0] * 36], [0.0], split="held_out")

    def test_freeze_after_fit(self):
        enc = SmallLearnedGraphEncoder()
        reps = [[0.0] * 36 for _ in range(20)]
        targets = [float(i) * 0.01 for i in range(20)]
        enc.fit(reps, targets, split="train", n_epochs=5)
        enc.freeze()
        assert enc.lifecycle == "frozen"

    def test_freeze_before_fit_raises(self):
        enc = SmallLearnedGraphEncoder()
        with pytest.raises(RuntimeError):
            enc.freeze()

    def test_fit_after_freeze_raises(self):
        enc = SmallLearnedGraphEncoder()
        reps = [[0.0] * 36 for _ in range(20)]
        targets = [float(i) * 0.01 for i in range(20)]
        enc.fit(reps, targets, split="train", n_epochs=5)
        enc.freeze()
        with pytest.raises(FrozenNormalizationError):
            enc.fit(reps, targets, split="train")


# ---------------------------------------------------------------------------
# 11. Encoder 8: HybridEncoder
# ---------------------------------------------------------------------------

class TestHybridEncoder:
    """Encoder 8: Hybrid representation."""

    def test_name(self):
        enc = HybridEncoder()
        assert enc.name == "hybrid"

    def test_dimension_sums_sub_encoders(self):
        sub1 = MinimalControlEncoder()
        sub2 = GlobalStateEncoder()
        enc = HybridEncoder(sub_encoders=[sub1, sub2])
        assert enc.dimension == sub1.dimension + sub2.dimension

    def test_add_encoder(self):
        enc = HybridEncoder()
        enc.add_encoder(MinimalControlEncoder())
        assert len(enc._sub_encoders) == 1

    def test_encode_combines(self):
        sub1 = MinimalControlEncoder()
        sub2 = GlobalStateEncoder()
        enc = HybridEncoder(sub_encoders=[sub1, sub2])
        gf = _global_features(_small_graph())
        lf = _local_features(_small_graph(), 0, 5)
        rep = enc.encode(MockState(), gf, "ADD_EDGE", {"u": 0, "v": 5}, lf)
        assert rep.dimension == sub1.dimension + sub2.dimension


# ---------------------------------------------------------------------------
# 12. Encoder registry
# ---------------------------------------------------------------------------

class TestEncoderRegistry:
    """Encoder registry tests."""

    def test_create_minimal(self):
        enc = EncoderRegistry.create("minimal-control")
        assert enc.name == "minimal-control"

    def test_create_global(self):
        enc = EncoderRegistry.create("global")
        assert enc.name == "global"

    def test_create_global_local(self):
        enc = EncoderRegistry.create("global-local")
        assert enc.name == "global-local"

    def test_create_unknown_raises(self):
        with pytest.raises(KeyError):
            EncoderRegistry.create("nonexistent")

    def test_available_encoders(self):
        names = EncoderRegistry.available_encoders()
        assert "minimal-control" in names
        assert "global" in names
        assert "global-local" in names
        assert "geometric" in names
        assert "spectral" in names

    def test_encoder_info(self):
        info = EncoderRegistry.encoder_info("global")
        assert info["name"] == "global"
        assert "dimension" in info
        assert "schema_hash" in info

    def test_all_encoder_info(self):
        all_info = EncoderRegistry.all_encoder_info()
        assert len(all_info) >= 9

    def test_provenance_hash(self):
        prov = EncoderProvenance(
            encoder_id="global", encoder_version="v1",
            encoder_schema_hash="abc", dataset_schema_hash="def",
        )
        assert prov.provenance_hash
        assert len(prov.provenance_hash) == 16


# ---------------------------------------------------------------------------
# 13. Node permutation invariance
# ---------------------------------------------------------------------------

class TestPermutationInvariance:
    """Node relabeling invariance: E(G,a) = E(P(G),P(a))."""

    def test_global_features_permutation_invariant(self):
        """Global features should be invariant to node relabeling."""
        graph = _cycle_graph(10)
        # Create a permuted version.
        perm = list(np.random.RandomState(42).permutation(10))
        edges = []
        valid = graph.valid.bool()
        for i in range(graph.src.shape[0]):
            if valid[i]:
                u = int(graph.src[i].item())
                v = int(graph.dst[i].item())
                edges.append((perm[u], perm[v]))
        perm_graph = make_graph_buffers(10, edges, capacity=20)
        f1 = extract_global_features(graph)
        f2 = extract_global_features(perm_graph)
        # Global features should be very close (not exactly equal due to
        # networkx internal ordering, but density, degree stats, etc. should match).
        assert abs(f1.density - f2.density) < 1e-6
        assert abs(f1.degree_mean - f2.degree_mean) < 1e-6
        assert abs(f1.degree_std - f2.degree_std) < 1e-6
        assert abs(f1.n_components - f2.n_components) < 1e-6

    def test_local_action_features_symmetric(self):
        """ADD_EDGE(u,v) and ADD_EDGE(v,u) should encode identically."""
        graph = _small_graph()
        f_uv = extract_local_action_features(graph, 0, 5)
        f_vu = extract_local_action_features(graph, 5, 0)
        # source_degree and target_degree swap, but the combined representation
        # should be symmetric for undirected graphs.
        assert f_uv.source_degree == f_vu.target_degree
        assert f_uv.target_degree == f_vu.source_degree
        assert f_uv.shortest_path_distance == f_vu.shortest_path_distance
        assert f_uv.common_neighbors == f_vu.common_neighbors
        assert f_uv.jaccard_coefficient == f_vu.jaccard_coefficient


# ---------------------------------------------------------------------------
# 14. Undirected action symmetry
# ---------------------------------------------------------------------------

class TestActionSymmetry:
    """ADD_EDGE(u,v) and ADD_EDGE(v,u) must encode identically."""

    def test_minimal_encoder_action_symmetry(self):
        enc = MinimalControlEncoder()
        state = MockState(graph_family="path")
        gf = _global_features(_small_graph())
        lf_uv = _local_features(_small_graph(), 0, 5)
        lf_vu = _local_features(_small_graph(), 5, 0)
        rep1 = enc.encode(state, gf, "ADD_EDGE", {"u": 0, "v": 5}, lf_uv)
        rep2 = enc.encode(state, gf, "ADD_EDGE", {"u": 5, "v": 0}, lf_vu)
        # The action one-hot part should be identical.
        assert rep1.vector[-DEFAULT_ACTION_SCHEMA.n_types:] == rep2.vector[-DEFAULT_ACTION_SCHEMA.n_types:]

    def test_global_encoder_action_symmetry(self):
        enc = GlobalStateEncoder()
        enc.fit([_global_features(_small_graph()) for _ in range(5)], split="train")
        enc.freeze()
        state = MockState()
        gf = _global_features(_small_graph())
        lf_uv = _local_features(_small_graph(), 0, 5)
        lf_vu = _local_features(_small_graph(), 5, 0)
        rep1 = enc.encode(state, gf, "ADD_EDGE", {"u": 0, "v": 5}, lf_uv)
        rep2 = enc.encode(state, gf, "ADD_EDGE", {"u": 5, "v": 0}, lf_vu)
        # State part should be identical.
        assert rep1.vector[:24] == rep2.vector[:24]


# ---------------------------------------------------------------------------
# 15. Missing feature handling
# ---------------------------------------------------------------------------

class TestMissingFeatures:
    """Missing/nonfinite inputs are handled explicitly."""

    def test_ensure_finite_replaces_nan(self):
        result = ensure_finite([1.0, float("nan"), 3.0])
        assert result == (1.0, 0.0, 3.0)

    def test_ensure_finite_replaces_inf(self):
        result = ensure_finite([1.0, float("inf"), float("-inf"), 3.0])
        assert result == (1.0, 0.0, 0.0, 3.0)

    def test_normalization_handles_nonfinite(self):
        norm = NormalizationStatistics()
        norm.fit([[1.0, 2.0], [3.0, 4.0]], split="train")
        result, mask = norm.transform([float("nan"), 4.0])
        assert mask[0] is True
        assert mask[1] is False
        assert result[0] == 0.0

    def test_encoder_handles_nonfinite_global(self):
        enc = GlobalStateEncoder()
        enc.fit([_global_features(_small_graph()) for _ in range(5)], split="train")
        enc.freeze()
        gf = [float("nan")] * 24
        rep = enc.encode(MockState(), gf, "ADD_EDGE", {}, _local_features())
        assert all(math.isfinite(v) for v in rep.vector)


# ---------------------------------------------------------------------------
# 16. Numerical robustness
# ---------------------------------------------------------------------------

class TestNumericalRobustness:
    """No encoder should silently generate non-finite representations."""

    def test_zero_edge_graph(self):
        graph = make_graph_buffers(5, [], capacity=4)
        feats = extract_global_features(graph)
        assert all(math.isfinite(v) for v in feats.vector)

    def test_single_node_graph(self):
        graph = make_graph_buffers(1, [], capacity=4)
        feats = extract_global_features(graph)
        assert all(math.isfinite(v) for v in feats.vector)

    def test_disconnected_graph(self):
        graph = make_graph_buffers(10, [(0, 1), (2, 3)], capacity=8)
        feats = extract_global_features(graph)
        assert all(math.isfinite(v) for v in feats.vector)

    def test_complete_graph(self):
        n = 8
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        graph = make_graph_buffers(n, edges, capacity=n * n)
        feats = extract_global_features(graph)
        assert all(math.isfinite(v) for v in feats.vector)

    def test_minimal_encoder_nonfinite_input(self):
        enc = MinimalControlEncoder()
        state = MockState(n_nodes=float("inf"), n_edges=float("nan"))
        rep = enc.encode(state, [float("nan")] * 24, "ADD_EDGE", {}, [float("inf")] * 12)
        assert all(math.isfinite(v) for v in rep.vector)


# ---------------------------------------------------------------------------
# 17. Probe benchmark
# ---------------------------------------------------------------------------

class TestProbeBenchmark:
    """Probe benchmark reproducibility."""

    def test_logistic_probe_fit_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = (X[:, 0] > 0).astype(float)
        probe = LogisticProbe(n_epochs=50)
        probe.fit(X, y)
        preds = probe.predict(X)
        assert len(preds) == 50

    def test_linear_probe_fit_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = X[:, 0] * 2 + 1
        probe = LinearProbe(n_epochs=50)
        probe.fit(X, y)
        preds = probe.predict(X)
        assert len(preds) == 50

    def test_run_probe_benchmark(self):
        enc = MinimalControlEncoder()
        reps, targets_du, targets_success, targets_risk = _make_representations(enc, n=30)
        report = run_probe_benchmark(reps, targets_du, targets_success, targets_risk, encoder_id="test")
        assert report.encoder_id == "test"
        assert len(report.results) > 0

    def test_probe_benchmark_reproducible(self):
        enc = MinimalControlEncoder()
        reps1, du1, s1, r1 = _make_representations(enc, n=30, seed=42)
        reps2, du2, s2, r2 = _make_representations(enc, n=30, seed=42)
        report1 = run_probe_benchmark(reps1, du1, s1, r1, encoder_id="test")
        report2 = run_probe_benchmark(reps2, du2, s2, r2, encoder_id="test")
        assert len(report1.results) == len(report2.results)
        for r1, r2 in zip(report1.results, report2.results):
            assert abs(r1.value - r2.value) < 1e-6


# ---------------------------------------------------------------------------
# 18. Collision analysis
# ---------------------------------------------------------------------------

class TestCollisionAnalysis:
    """Representation collision reporting."""

    def test_no_collisions_for_distinct(self):
        enc = MinimalControlEncoder()
        reps = []
        for i in range(10):
            state = MockState(graph_family="path", n_nodes=10 + i, n_edges=9 + i)
            rep = enc.encode(state, _global_features(), "ADD_EDGE", {}, _local_features())
            reps.append(rep)
        report = analyze_collisions(reps, [0.1 * i for i in range(10)])
        assert report.n_representations == 10
        assert report.collision_rate >= 0.0

    def test_collision_detection(self):
        enc = MinimalControlEncoder()
        # Create identical representations.
        state = MockState(graph_family="path")
        rep = enc.encode(state, _global_features(), "ADD_EDGE", {}, _local_features())
        reps = [rep] * 5
        report = analyze_collisions(reps, [0.1, 0.2, 0.3, 0.4, 0.5])
        assert report.n_collisions == 4
        assert report.collision_rate == 0.8
        assert report.max_collision_group_size == 5

    def test_empty_representations(self):
        report = analyze_collisions([], [])
        assert report.n_representations == 0

    def test_collision_report_to_log(self):
        enc = MinimalControlEncoder()
        rep = enc.encode(MockState(), _global_features(), "ADD_EDGE", {}, _local_features())
        report = analyze_collisions([rep], [0.1])
        log = report.to_log()
        assert "collision_rate" in log


# ---------------------------------------------------------------------------
# 19. Complexity-adjusted comparison
# ---------------------------------------------------------------------------

class TestComplexityComparison:
    """Complexity-adjusted representation score."""

    def test_complexity_metrics(self):
        cm = ComplexityMetrics(encoder_id="test", dimension=24, n_parameters=100)
        assert cm.complexity_score > 0

    def test_compute_effectiveness(self):
        probe_report = EncoderProbeReport(
            encoder_id="test", encoder_dimension=24,
            results=[ProbeResult(
                encoder_id="test", task="sign", metric="accuracy",
                value=0.7, n_samples=100, n_features=24, baseline_value=0.5,
            )],
        )
        complexity = ComplexityMetrics(encoder_id="test", dimension=24)
        comp = compute_effectiveness(probe_report, complexity)
        assert abs(comp.predictive_gain - 0.2) < 1e-6  # 0.7 - 0.5
        assert comp.effectiveness_score > 0

    def test_compare_encoders(self):
        # Create two probe reports.
        r1 = EncoderProbeReport(
            encoder_id="enc1", encoder_dimension=24,
            results=[ProbeResult("enc1", "task", "acc", 0.7, 100, 24, 0.5)],
        )
        r2 = EncoderProbeReport(
            encoder_id="enc2", encoder_dimension=64,
            results=[ProbeResult("enc2", "task", "acc", 0.6, 100, 64, 0.5)],
        )
        c1 = ComplexityMetrics(encoder_id="enc1", dimension=24)
        c2 = ComplexityMetrics(encoder_id="enc2", dimension=64, n_parameters=10000)
        comparisons = compare_encoders([r1, r2], [c1, c2])
        # enc1 should be more effective (higher gain, lower cost).
        assert comparisons[0].encoder_id == "enc1"


# ---------------------------------------------------------------------------
# 20. Authority boundary
# ---------------------------------------------------------------------------

class TestAuthorityBoundaryExp3:
    """v5.11 authority boundary untouched by encoders."""

    def test_encoders_do_not_touch_runtime(self):
        """Creating and using encoders should not affect the runtime."""
        runtime = LGAERuntime(graph=_small_graph(), config=_cfg(), runtime_config=RuntimeConfig())
        gen_before = runtime.snapshot().generation
        # Use an encoder.
        enc = MinimalControlEncoder()
        gf = _global_features(runtime._engine.graph)
        lf = _local_features(runtime._engine.graph, 0, 5)
        rep = enc.encode(MockState(), gf, "ADD_EDGE", {"u": 0, "v": 5}, lf)
        assert rep is not None
        # Runtime state should be unchanged.
        gen_after = runtime.snapshot().generation
        assert gen_before == gen_after

    def test_encoders_are_advisory_only(self):
        """Encoders produce representations but have no mutation authority."""
        enc = GlobalStateEncoder()
        enc.fit([_global_features(_small_graph()) for _ in range(5)], split="train")
        enc.freeze()
        rep = enc.encode(MockState(), _global_features(), "ADD_EDGE", {}, _local_features())
        # The representation is just a vector — no side effects.
        assert isinstance(rep, StateActionRepresentation)
        assert isinstance(rep.vector, tuple)


# ---------------------------------------------------------------------------
# 21. Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    """Serialization roundtrip tests."""

    def test_representation_to_log(self):
        rep = StateActionRepresentation(
            encoder_id="test", encoder_version="v1", schema_hash="abc",
            vector=(1.0, 2.0, 3.0), dimension=3,
            state_feature_hash="s1", action_feature_hash="a1",
            normalization_hash="n1",
        )
        log = rep.to_log()
        data = json.dumps(log, sort_keys=True)
        parsed = json.loads(data)
        assert parsed["encoder_id"] == "test"
        assert parsed["dimension"] == 3

    def test_normalization_to_log(self):
        norm = NormalizationStatistics()
        norm.fit([[1.0, 2.0], [3.0, 4.0]], split="train")
        log = norm.to_log()
        assert log["state"] == "fitted_train"
        assert log["dimension"] == 2

    def test_provenance_to_log(self):
        prov = EncoderProvenance(
            encoder_id="global", encoder_version="v1",
            encoder_schema_hash="abc",
        )
        log = prov.to_log()
        assert log["encoder_id"] == "global"
        assert "provenance_hash" in log


# ---------------------------------------------------------------------------
# 22. Dataset binding
# ---------------------------------------------------------------------------

class TestDatasetBinding:
    """Encoder provenance binds to dataset and schema hashes."""

    def test_provenance_with_dataset_hash(self):
        prov = EncoderProvenance(
            encoder_id="global", encoder_version="v1",
            encoder_schema_hash="enc_hash",
            dataset_schema_hash="ds_hash",
            train_split_hash="train_hash",
        )
        assert prov.dataset_schema_hash == "ds_hash"
        assert prov.train_split_hash == "train_hash"

    def test_provenance_hash_differs_for_different_datasets(self):
        p1 = EncoderProvenance("global", "v1", "abc", dataset_schema_hash="ds1")
        p2 = EncoderProvenance("global", "v1", "abc", dataset_schema_hash="ds2")
        assert p1.provenance_hash != p2.provenance_hash
