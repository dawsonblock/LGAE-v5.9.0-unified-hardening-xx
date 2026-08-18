"""Regression coverage for the v5.11 canonical commit convergence repair."""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import LGAERuntime, make_graph_transaction, StructuralTransaction
from lgae_v3.runtime.contracts import AuthorizationResult, AuthorizationStatus
from lgae_v3.types import MutationResult


def _cfg():
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.audit.persistent_homology_enabled = False
    cfg.audit.entropic_nodes = 0
    cfg.audit.bakry_nodes = 0
    cfg.audit.cde_nodes = 0
    cfg.audit.exact_lly_top_k = 0
    cfg.audit.orc_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]
    cfg.mutation.curvature_ema_enabled = False
    return cfg


def _graph():
    return make_graph_buffers(
        6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=12,
    )


def _bound_weight_tx(rt: LGAERuntime):
    shadow = rt.engine.graph.clone()
    shadow.weight[0] = shadow.weight[0] * 3.0
    tx = make_graph_transaction(
        base_state_version=int(rt.engine.graph.version),
        base_state_hash=rt.authority_hash,
        shadow_graph=shadow,
        mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        step=0,
    )
    tx = StructuralTransaction(
        transaction_id=tx.transaction_id,
        base_state_version=tx.base_state_version,
        base_state_hash=tx.base_state_hash,
        graph_delta=tx.graph_delta,
        authorization_id=tx.authorization_binding_hash(),
        delta_hash=tx.delta_hash,
        mutation_result=tx.mutation_result,
    )
    auth = AuthorizationResult(
        snapshot_id="s",
        state_version=int(rt.engine.graph.version),
        state_hash=rt.authority_hash,
        status=AuthorizationStatus.AUTHORIZED,
        transaction_hash=tx.transaction_id,
    )
    return tx, auth


def test_fresh_transaction_uses_full_authority_identity_and_commits():
    torch.manual_seed(42)
    rt = LGAERuntime(_graph(), _cfg())
    tx, auth = _bound_weight_tx(rt)
    assert tx.base_state_hash == rt.authority_hash
    before = rt.authority_hash
    result = rt.commit_channel.commit(tx, auth)
    assert result.committed
    assert rt.authority_hash != before


@pytest.mark.parametrize("failpoint", ["before_state_swap", "after_state_swap"])
def test_exception_failpoints_restore_complete_authority_state(failpoint):
    torch.manual_seed(42)
    rt = LGAERuntime(_graph(), _cfg())
    tx, auth = _bound_weight_tx(rt)
    before_hash = rt.authority_hash
    before_weight = float(rt.engine.graph.weight[0])
    before_fibers = rt.engine.fibers().detach().clone()
    rt.commit_channel.set_failpoint(failpoint)
    with pytest.raises(RuntimeError, match="failpoint"):
        rt.commit_channel.commit(tx, auth)
    assert rt.authority_hash == before_hash
    assert float(rt.engine.graph.weight[0]) == before_weight
    assert torch.equal(rt.engine.fibers().detach(), before_fibers)


def test_runtime_config_has_active_multiobjective_weights():
    rt = LGAERuntime(_graph(), _cfg())
    assert rt.runtime_config.information_gain_weight > 0
    assert rt.runtime_config.cost_weight > 0
    assert rt.runtime_config.risk_weight > 0
