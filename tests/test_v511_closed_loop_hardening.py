from __future__ import annotations

import types
import numpy as np
import pytest
import torch

from lgae_v3 import (
    LGAEConfig,
    LGAEEngine,
    make_graph_buffers,
    MutationDecision,
    MutationResult,
    StructuralExecutive,
    StructuralLearningLoop,
    StructuralAction,
    ActionProposal,
    EnsembleUncertainty,
    ConformalCalibrator,
    DynamicGaugeBank,
    ANNNeighborIndex,
    EdgeSemantics,
    CausalEdgeRegistry,
    HypergraphBuffers,
)
from lgae_v3.counterfactual import CounterfactualResult
from lgae_v3.sheaf_diffusion import sheaf_laplacian_diffusion, sheaf_adjacency_diffusion
from lgae_v3.topology import bottleneck_distance
from lgae_v3.benchmark import BenchmarkHarness, TaskA_Bottleneck
from lgae_v3.version import VERSION


def _cfg(*, gauge_dim: int = 0) -> LGAEConfig:
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = gauge_dim
    cfg.audit.orc_backend = "exact_lp"
    cfg.audit.persistent_homology_enabled = False
    cfg.audit.entropic_nodes = 0
    cfg.audit.bakry_nodes = 0
    cfg.audit.cde_nodes = 0
    cfg.audit.exact_lly_top_k = 0
    cfg.audit.orc_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]
    return cfg


def _engine(*, gauge_dim: int = 0) -> LGAEEngine:
    graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    return LGAEEngine(graph, _cfg(gauge_dim=gauge_dim))


def _accept_latent(self, *args, **kwargs):
    return MutationResult(MutationDecision.ACCEPT, ["forced_accept"], metadata=dict(kwargs.get("metadata", {})))


def _quarantine_latent(self, *args, **kwargs):
    return MutationResult(MutationDecision.QUARANTINE, ["forced_quarantine"], metadata=dict(kwargs.get("metadata", {})))


def _accept_graph(self, graph, z, mutation, **kwargs):
    shadow = graph.clone()
    metadata = mutation.apply(shadow)
    return MutationResult(MutationDecision.ACCEPT, ["forced_accept"], metadata=metadata), shadow


def _quarantine_graph(self, graph, z, mutation, **kwargs):
    shadow = graph.clone()
    metadata = mutation.apply(shadow)
    return MutationResult(MutationDecision.QUARANTINE, ["forced_quarantine"], metadata=metadata), shadow


def test_uncertainty_estimate_is_read_only_for_authoritative_executive():
    torch.manual_seed(0)
    executive = StructuralExecutive(hidden_dim=16)
    ensemble = EnsembleUncertainty(executive, ensemble_size=3)
    before = {k: v.detach().clone() for k, v in executive.network.state_dict().items()}
    obs = torch.randn(executive._obs_dim)
    ensemble.estimate(obs, 1)
    after = executive.network.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_uncertainty_ensemble_updates_member_parameters():
    torch.manual_seed(1)
    executive = StructuralExecutive(hidden_dim=16)
    ensemble = EnsembleUncertainty(executive, ensemble_size=2, bootstrap_probability=1.0)
    before = {k: v.detach().clone() for k, v in ensemble.members[0].state_dict().items()}
    obs = torch.randn(executive._obs_dim)
    updated = ensemble.update(obs, 1, 2.0, cost_target=0.1, risk_target=0.0, ig_target=0.0)
    assert updated == 2
    after = ensemble.members[0].state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_conformal_uses_finite_sample_order_statistic():
    cal = ConformalCalibrator(alpha=0.1)
    q = cal.calibrate([0.0] * 100, [float(i) for i in range(100)])
    # ceil(101 * .9)=91 => 91st sorted residual => 90.0
    assert q == pytest.approx(90.0)


def test_sheaf_laplacian_moves_destination_toward_source():
    z = torch.tensor([[1.0], [0.0]])
    src = torch.tensor([0])
    dst = torch.tensor([1])
    U = torch.ones((1, 1, 1))
    w = torch.ones(1)
    out = sheaf_laplacian_diffusion(z, src, dst, U, w, eta=0.1)
    assert out[1, 0].item() == pytest.approx(0.1)


def test_sheaf_adjacency_preserves_isolated_node():
    z = torch.tensor([[1.0], [0.0], [3.0]])
    src = torch.tensor([0])
    dst = torch.tensor([1])
    U = torch.ones((1, 1, 1))
    w = torch.ones(1)
    out = sheaf_adjacency_diffusion(z, src, dst, U, w, eta=0.1)
    assert out[2, 0].item() == pytest.approx(3.0)


def test_bottleneck_distance_handles_unmatched_large_features():
    a = np.array([[0.0, 10.0], [0.0, 8.0]])
    b = np.empty((0, 2), dtype=float)
    assert bottleneck_distance(a, b) == pytest.approx(5.0)


def test_dynamic_gauge_reverse_edges_are_inverse_transposes():
    torch.manual_seed(4)
    bank = DynamicGaugeBank(edge_capacity=8, dim=3, hidden_dim=16)
    z = torch.randn(3, 3)
    src = torch.tensor([0, 1])
    dst = torch.tensor([1, 0])
    U = bank.matrices(z, src, dst)
    assert torch.allclose(U[1], U[0].transpose(-1, -2), atol=1e-5, rtol=1e-5)
    eye = torch.eye(3)
    assert torch.allclose(U[0].T @ U[0], eye, atol=1e-5, rtol=1e-5)
    assert torch.linalg.det(U[0]).item() == pytest.approx(1.0, abs=1e-5)


def test_ann_returns_k_valid_nonself_neighbors():
    torch.manual_seed(5)
    z = torch.randn(24, 5)
    index = ANNNeighborIndex(dim=5, n_candidates=16, n_final=5, backend="numpy", refresh_interval=0)
    index.build(z)
    ids, dist = index.query(z, 5)
    assert ids.shape == (24, 5)
    assert dist.shape == (24, 5)
    for i in range(24):
        assert bool((ids[i] >= 0).all())
        assert i not in ids[i].tolist()
        assert len(set(ids[i].tolist())) == 5


def test_causal_reclassification_removes_cached_causal_path():
    reg = CausalEdgeRegistry()
    reg.register(0, 1, EdgeSemantics.CAUSAL)
    assert 1 in reg.causal_children(0)
    reg.register(0, 1, EdgeSemantics.ASSOCIATION)
    assert 1 not in reg.causal_children(0)
    assert reg.causal_paths(0, 1) == []


def test_hypergraph_rejects_out_of_range_node():
    h = HypergraphBuffers(num_nodes=3, hyperedge_capacity=4)
    with pytest.raises((ValueError, IndexError)):
        h.add_hyperedge([0, 1, 99])


def test_benchmark_harness_respects_task_subset():
    harness = BenchmarkHarness(tasks=[TaskA_Bottleneck()])
    result = harness.run_oracle(seed=7)
    assert result.total_tasks == 1
    assert len(result.diagnosis_results) == 1


def test_checkpoint_payload_uses_canonical_current_version():
    engine = _engine()
    payload = engine.checkpoint_payload()
    assert payload["version"] == VERSION == "5.11.0"


def test_transactional_fiber_action_commits_only_on_accept():
    engine = _engine()
    engine.governor.evaluate_latent_transition = types.MethodType(_accept_latent, engine.governor)
    before = int(engine.fibers.capacity[0].item())
    result = engine.evaluate_fiber_action("spawn_fiber", node=0)
    assert result.decision == MutationDecision.ACCEPT
    assert int(engine.fibers.capacity[0].item()) == before + 1

    # Quarantine must restore the authoritative fiber bank and retain the shadow.
    engine.governor.evaluate_latent_transition = types.MethodType(_quarantine_latent, engine.governor)
    before_snapshot = engine.fibers.snapshot()
    result = engine.evaluate_fiber_action("spawn_fiber", node=0)
    assert result.decision == MutationDecision.QUARANTINE
    assert torch.equal(engine.fibers.active_mask, before_snapshot.active_mask)
    assert engine.quarantine[-1].kind == "fiber"
    assert engine.quarantine[-1].shadow_fibers is not None


def test_transactional_gauge_action_preserves_so_and_quarantine_shadow():
    engine = _engine(gauge_dim=2)
    engine.governor.evaluate_latent_transition = types.MethodType(_accept_latent, engine.governor)
    before = engine.gauge_connections.raw_generators.detach().clone()
    result = engine.evaluate_gauge_action(u=0, v=1, magnitude=0.05)
    assert result.decision == MutationDecision.ACCEPT
    assert not torch.equal(before, engine.gauge_connections.raw_generators)
    U = engine.gauge_connections.matrices()[0]
    assert torch.allclose(U.T @ U, torch.eye(2), atol=1e-5, rtol=1e-5)
    assert torch.linalg.det(U).item() == pytest.approx(1.0, abs=1e-5)

    engine.governor.evaluate_latent_transition = types.MethodType(_quarantine_latent, engine.governor)
    raw_before = engine.gauge_connections.raw_generators.detach().clone()
    result = engine.evaluate_gauge_action(u=0, v=1, magnitude=0.02)
    assert result.decision == MutationDecision.QUARANTINE
    assert torch.equal(raw_before, engine.gauge_connections.raw_generators)
    assert engine.quarantine[-1].kind == "gauge"
    assert engine.quarantine[-1].shadow_gauge_raw is not None


def _force_counterfactual(action: StructuralAction) -> CounterfactualResult:
    prop = ActionProposal(action, 1.0, 0.0, 0.0, 0.0, 1.0, uncertainty=0.0, lcb=1.0)
    no = ActionProposal(StructuralAction.NO_OP, 0.0, 0.0, 0.0, 0.0, 0.0, uncertainty=0.0, lcb=0.0)
    return CounterfactualResult([prop, no], action, prop, no, True)


def _force_uncertainty_accept(loop: StructuralLearningLoop):
    from lgae_v3.uncertainty import UncertaintyEstimate
    loop.uncertainty_estimator.estimate = lambda obs, idx: UncertaintyEstimate(1.0, 0.0, 1.0, 1.0)


def test_structural_loop_engine_accept_commits_authoritative_graph():
    engine = _engine()
    loop = StructuralLearningLoop(engine.cfg, engine=engine)
    loop.counterfactual.evaluate = lambda *a, **k: _force_counterfactual(StructuralAction.ADD_EDGE)
    loop.executive.select_target = lambda *a, **k: {"u": 0, "v": 3, "weight": 1.0}
    _force_uncertainty_accept(loop)
    engine.governor.evaluate_mutation = types.MethodType(_accept_graph, engine.governor)
    before_hash = engine.graph.state_hash()
    before_edges = engine.graph.edge_count
    result = loop.step(engine.graph, engine.fibers().detach(), utility_fn=lambda g, z: float(g.edge_count))
    assert result.executed is True
    assert result.governance_decision == "accept"
    assert engine.graph.edge_count == before_edges + 1
    assert engine.graph.state_hash() != before_hash


def test_structural_loop_quarantine_never_executes_or_credits():
    engine = _engine()
    loop = StructuralLearningLoop(engine.cfg, engine=engine)
    loop.counterfactual.evaluate = lambda *a, **k: _force_counterfactual(StructuralAction.ADD_EDGE)
    loop.executive.select_target = lambda *a, **k: {"u": 0, "v": 3, "weight": 1.0}
    _force_uncertainty_accept(loop)
    engine.governor.evaluate_mutation = types.MethodType(_quarantine_graph, engine.governor)
    before_hash = engine.graph.state_hash()
    before_receipts = len(loop.credit_tracker.get_receipts())
    before_exp = len(loop.executive._experience)
    result = loop.step(engine.graph, engine.fibers().detach(), utility_fn=lambda g, z: float(g.edge_count))
    assert result.governance_decision == "quarantine"
    assert result.executed is False
    assert engine.graph.state_hash() == before_hash
    # Quarantine creates no committed-action credit receipt, but v5.2 records
    # risk-only supervision so the executive can learn to avoid unsafe proposals.
    assert len(loop.credit_tracker.get_receipts()) == before_receipts
    assert len(loop.executive._experience) == before_exp + 1
    assert loop.executive._experience[-1]["supervise_delta_u"] is False
    assert loop.executive._experience[-1]["risk_target"] == 0.5
    assert engine.quarantine[-1].kind == "graph"


def test_safe_checkpoint_roundtrips_fiber_quarantine(tmp_path):
    engine = _engine()
    engine.governor.evaluate_latent_transition = types.MethodType(_quarantine_latent, engine.governor)
    result = engine.evaluate_fiber_action("spawn_fiber", node=0)
    assert result.decision == MutationDecision.QUARANTINE
    expected = engine.quarantine[-1].shadow_fibers.clone()
    path = tmp_path / "safe"
    engine.save_checkpoint(path)

    restored = _engine()
    restored.load_checkpoint_(path)
    assert len(restored.quarantine) == 1
    item = restored.quarantine[0]
    assert item.kind == "fiber"
    assert item.shadow_fibers is not None
    assert torch.equal(item.shadow_fibers.active_mask, expected.active_mask)
    assert torch.allclose(item.shadow_fibers.latent, expected.latent)


def test_safe_checkpoint_roundtrips_gauge_quarantine(tmp_path):
    engine = _engine(gauge_dim=2)
    engine.governor.evaluate_latent_transition = types.MethodType(_quarantine_latent, engine.governor)
    result = engine.evaluate_gauge_action(u=0, v=1, magnitude=0.03)
    assert result.decision == MutationDecision.QUARANTINE
    expected = engine.quarantine[-1].shadow_gauge_raw.clone()
    path = tmp_path / "safe"
    engine.save_checkpoint(path)

    restored = _engine(gauge_dim=2)
    restored.load_checkpoint_(path)
    item = restored.quarantine[0]
    assert item.kind == "gauge"
    assert item.shadow_gauge_raw is not None
    assert torch.allclose(item.shadow_gauge_raw, expected)
