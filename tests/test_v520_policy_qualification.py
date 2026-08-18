from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from lgae_v3.benchmark.tasks import StructuralAction, TaskA_Bottleneck
from lgae_v3.benchmark.policy_qualification import qualify_structural_policy
from lgae_v3.credit import MutationCreditTracker
from lgae_v3.executive import StructuralExecutive
from lgae_v3.config import LGAEConfig
from lgae_v3.types import make_graph_buffers


def test_policy_qualification_heldout_gate():
    # CI-light version of the release qualification.
    _, result = qualify_structural_policy(
        train_seeds=range(8), heldout_seeds=(101, 102), gradient_steps=250, seed=0,
    )
    assert result.diagnosis_accuracy >= 0.75
    assert result.mean_regret < 0.40


def test_latent_diagnostics_are_in_observation():
    task = TaskA_Bottleneck()
    state = task.initial_state(seed=7)
    ex = StructuralExecutive(state.config)
    obs = ex.observe(state.graph, state.z)
    assert obs.latent_mean_norm > 0
    assert obs.latent_max_norm >= obs.latent_mean_norm
    assert obs.edge_latent_mismatch_max >= obs.edge_latent_mismatch_mean >= 0


def test_rejected_governance_trains_risk_without_delta_target():
    ex = StructuralExecutive()
    task = TaskA_Bottleneck()
    state = task.initial_state(seed=3)
    obs = ex.observe(state.graph, state.z)
    ex.record_governance_outcome(obs, StructuralAction.PRUNE_EDGE, "reject", cost_target=0.5)
    exp = ex._experience[-1]
    assert exp["risk_target"] == 1.0
    assert exp["supervise_delta_u"] is False
    assert exp["cost_target"] == 0.5


def test_learned_magnitudes_are_bounded():
    task = TaskA_Bottleneck()
    state = task.initial_state(seed=5)
    ex = StructuralExecutive(state.config)
    for action in (
        StructuralAction.ADD_EDGE,
        StructuralAction.REWEIGHT_AFFINITY,
        StructuralAction.REWEIGHT_LENGTH,
        StructuralAction.COUPLED_REWEIGHT,
        StructuralAction.CHANGE_GAUGE,
    ):
        target = ex.select_target(action, state.graph, state.z)
        assert target
        if "factor" in target:
            assert 0.5 <= target["factor"] <= 2.0
        if "weight" in target:
            assert 0.5 <= target["weight"] <= 2.0
        if "magnitude" in target:
            assert 0.0025 <= target["magnitude"] <= 0.1


def test_credit_pending_roundtrip_survives_restart(tmp_path: Path):
    tr = MutationCreditTracker(gamma=0.99, horizons=[2, 4])
    receipt = tr.record_mutation(
        StructuralAction.ADD_EDGE, 10, 0.5, 0.2, "accept", [], "a", "b", "cfg",
        metadata={"target": {"u": 0, "v": 1}},
    )
    tr.record_utility(10, 1.0)
    tr.record_utility(12, 1.5)
    p = tmp_path / "credit.json"
    tr.save_state(str(p))

    restored = MutationCreditTracker(gamma=0.5, horizons=[1])
    restored.load_state(str(p))
    assert receipt.receipt_id in restored._pending
    assert restored._pending[receipt.receipt_id]["baseline_utility"] == 1.0
    assert restored._pending[receipt.receipt_id]["utility_samples"] == [(2, 1.5)]
    assert restored.get_receipts()[0].metadata["target"] == {"u": 0, "v": 1}
    restored.record_utility(14, 2.0)
    out = restored.get_outcomes()[0]
    assert out.utility_by_horizon == {2: 1.5, 4: 2.0}


def test_fiber_target_proposes_width_within_spawn_limit():
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 8
    cfg.fiber.spawn_width = 3
    graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    z = torch.randn(4, 2)
    ex = StructuralExecutive(cfg)
    target = ex.select_target(StructuralAction.SPAWN_FIBER, graph, z)
    assert 1 <= target["width"] <= cfg.fiber.spawn_width
