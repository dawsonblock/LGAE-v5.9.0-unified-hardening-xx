from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from lgae_v3.ann_index import ANNNeighborIndex
from lgae_v3.config import LGAEConfig, config_governance_hash
from lgae_v3.credit import MutationCreditTracker
from lgae_v3.curvature.bakry_emery import (
    analytic_markov_generator,
    bakry_emery_curvature,
    stationary_symmetrized_markov_generator,
)
from lgae_v3.evolution import LGAEEngine
from lgae_v3.executive import StructuralAction
from lgae_v3.neighbor_index import ExactChunkedKNN
from lgae_v3.production_dynamics import CurvatureHysteresisController, LatentEquilibriumBarrier
from lgae_v3.sheaf_diffusion import gauge_orthogonality_penalty, sheaf_laplacian_diffusion
from lgae_v3.timescales import MultiTimescaleController, TimescaleSchedule
from lgae_v3.transactions import graph_transaction
from lgae_v3.types import GraphBuffers


def _graph(n: int = 3, capacity: int = 4) -> GraphBuffers:
    src = torch.zeros(capacity, dtype=torch.long)
    dst = torch.zeros(capacity, dtype=torch.long)
    w = torch.zeros(capacity)
    valid = torch.zeros(capacity, dtype=torch.bool)
    edges = [(i, i + 1) for i in range(n - 1)]
    for i, (u, v) in enumerate(edges):
        src[i], dst[i], w[i], valid[i] = u, v, 1.0, True
    return GraphBuffers(n, src, dst, w, valid)


def test_sheaf_transport_is_nonexpansive_for_bad_external_map():
    z = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    src = torch.tensor([0])
    dst = torch.tensor([1])
    U = (2.0 * torch.eye(2)).unsqueeze(0)  # deliberately expansive / non-orthogonal
    out = sheaf_laplacian_diffusion(z, src, dst, U, torch.ones(1), eta=1.0, transport_norm_ratio=1.0)
    assert torch.linalg.vector_norm(out[1]).item() <= 1.0 + 1e-6


def test_gauge_penalty_detects_nonorthogonal_maps():
    good = torch.eye(3).unsqueeze(0)
    bad = (1.5 * torch.eye(3)).unsqueeze(0)
    assert gauge_orthogonality_penalty(good).item() == pytest.approx(0.0, abs=1e-8)
    assert gauge_orthogonality_penalty(bad).item() > 1.0


def test_curvature_ema_hysteresis_requires_persistence_then_allows():
    c = CurvatureHysteresisController(alpha=0.5, variance_alpha=0.5, min_samples=3, sigma_guard=1.0)
    for _ in range(2):
        c.update({(0, 1): -1.0})
    allowed, meta = c.allows("add", 0, 2, add_threshold=-0.2, prune_threshold=0.2)
    assert not allowed and meta["reason"] == "curvature_ema_warmup"
    c.update({(0, 1): -1.0})
    allowed, meta = c.allows("add", 0, 2, add_threshold=-0.2, prune_threshold=0.2)
    assert allowed
    assert meta["ema_curvature"] < -0.2


def test_curvature_hysteresis_blocks_noisy_signal():
    c = CurvatureHysteresisController(alpha=0.5, variance_alpha=0.5, min_samples=3, sigma_guard=1.0)
    for value in (-2.0, 2.0, -2.0, 2.0, -2.0, 2.0):
        c.update({(0, 1): value})
    allowed, meta = c.allows("prune", 0, 1, add_threshold=-0.2, prune_threshold=0.2)
    assert not allowed
    assert meta["reason"] == "curvature_noise_exceeds_hysteresis_band"


def test_directed_markov_gamma2_symmetrization_has_real_symmetric_similarity():
    P = torch.tensor([
        [0.10, 0.90, 0.00],
        [0.00, 0.10, 0.90],
        [0.80, 0.20, 0.00],
    ], dtype=torch.float64)
    Q, pi, S = stationary_symmetrized_markov_generator(P)
    assert torch.allclose(Q.sum(-1), torch.zeros(3, dtype=Q.dtype), atol=1e-10)
    assert torch.allclose(S, S.T, atol=1e-10)
    assert torch.all(pi > 0)
    assert torch.isfinite(torch.linalg.eigvalsh(S)).all()
    assert math.isfinite(bakry_emery_curvature(Q, 0))
    q2, pi2, mode = analytic_markov_generator(P, directed_policy="symmetrize")
    assert "symmetrized" in mode
    assert torch.allclose(q2, Q, atol=1e-9)
    assert torch.allclose(pi2, pi, atol=1e-9)


def test_graph_transaction_rolls_back_and_invalidates_index():
    g = _graph()
    z = torch.randn(3, 4)
    index = ExactChunkedKNN()
    index.build(z)
    before = g.state_hash()
    assert not index.cache_dirty
    with graph_transaction(g, index):
        # destructive tentative edit; no commit call -> automatic rollback
        g.valid[0] = False
        g.weight[0] = 0.0
        g.length[0] = 0.0
        g.bump_version()
    assert g.state_hash() == before
    assert index.cache_dirty
    assert index.cache_metadata()["reason"] == "transaction_rollback"


def test_engine_ann_generation_invalidates_on_authoritative_state_change():
    g = _graph()
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 4
    cfg.operator.diagnostic_full_kernel_max_nodes = 2  # force sparse operator on N=3
    engine = LGAEEngine(g, cfg)
    index = ANNNeighborIndex(dim=4, n_candidates=3, n_final=2, backend="numpy")
    engine.attach_neighbor_index(index)
    _ = engine.governor.operators(engine.graph, engine.fibers().detach())
    generation = index.cache_generation
    assert generation >= 1 and not index.cache_dirty
    engine._invalidate_neighbor_indices("unit_test_commit")
    assert index.cache_dirty
    _ = engine.governor.operators(engine.graph, engine.fibers().detach())
    assert index.cache_generation > generation
    assert not index.cache_dirty


def test_latent_equilibrium_barrier_requires_consecutive_convergence():
    b = LatentEquilibriumBarrier(delta_tol=1e-3, required_consecutive=2)
    z = torch.ones(4, 3)
    assert not b.observe(z)
    assert not b.observe(z + 1e-5)
    assert b.observe(z + 1e-5)
    assert b.is_equilibrated
    assert b.summary()["last_relative_delta"] < 1e-3


def test_timescale_slow_update_waits_for_equilibrium():
    ctrl = MultiTimescaleController(
        schedule=TimescaleSchedule(fast_interval=1, medium_interval=1, slow_interval=1),
        min_fast_before_medium=0,
        min_medium_before_slow=0,
        equilibrium_delta_tol=1e-3,
        equilibrium_required_steps=2,
    )
    z = torch.ones(3, 2)
    ctrl.observe_latent(z)
    ctrl.update(0)
    assert not ctrl.can_adapt_topology(0)
    ctrl.observe_latent(z)
    ctrl.observe_latent(z)
    assert ctrl.can_adapt_topology(1)


def test_credit_uses_counterfactual_baseline_advantage():
    tracker = MutationCreditTracker(gamma=1.0, horizons=[1])
    r = tracker.record_mutation(
        StructuralAction.ADD_EDGE, step=0, predicted_delta_u=0.5, predicted_uncertainty=0.1,
        governance_decision="accept", governance_reasons=[], graph_hash_before="g0",
        graph_hash_after="g1", config_governance_hash="cfg", counterfactual_baseline=0.4,
    )
    tracker.record_utility(0, 0.0)
    tracker.record_utility(1, 1.0)
    out = tracker.get_outcomes()[0]
    assert out.receipt_id == r.receipt_id
    assert out.discounted_return == pytest.approx(1.0)
    assert out.baseline_return == pytest.approx(0.4)
    assert out.advantage == pytest.approx(0.6)
    assert out.retained


def test_credit_baseline_state_roundtrip(tmp_path: Path):
    tracker = MutationCreditTracker(gamma=1.0, horizons=[1])
    tracker.record_mutation(
        StructuralAction.PRUNE_EDGE, 0, 0.0, 0.0, "accept", [], "before", "after", "cfg",
        counterfactual_baseline=0.25,
    )
    tracker.record_utility(0, 0.0)
    tracker.record_utility(1, 1.0)
    path = tmp_path / "credit.json"
    tracker.save_state(str(path))
    restored = MutationCreditTracker()
    restored.load_state(str(path))
    out = restored.get_outcomes()[0]
    assert out.advantage == pytest.approx(0.75)
    assert restored.baseline_estimator.state_dict()["counts"] == tracker.baseline_estimator.state_dict()["counts"]


def test_new_production_policy_fields_are_governance_committed():
    a = LGAEConfig()
    b = LGAEConfig()
    b.mutation.curvature_ema_enabled = True
    assert config_governance_hash(a) != config_governance_hash(b)
    c = LGAEConfig()
    c.audit.directed_gamma2_policy = "reject"
    assert config_governance_hash(a) != config_governance_hash(c)
