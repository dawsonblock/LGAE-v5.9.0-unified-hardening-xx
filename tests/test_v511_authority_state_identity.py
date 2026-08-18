"""Phase 2 tests: AuthorityStateIdentity first-class token verification."""
from __future__ import annotations

import json
import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.types import MutationResult
from lgae_v3.runtime import (
    LGAERuntime,
    AuthorityStateIdentity,
    make_graph_transaction,
    make_fiber_transaction,
    make_gauge_transaction,
    StructuralTransaction,
    StaleTransactionError,
)
from lgae_v3.runtime.contracts import (
    ObservationSnapshot,
    AuthorizationResult,
    AuthorizationStatus,
    CommitResult,
)


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


def test_authority_identity_roundtrip():
    ident = AuthorityStateIdentity(version=42, authority_hash="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789")
    d = ident.to_dict()
    ident2 = AuthorityStateIdentity.from_dict(d)
    assert ident == ident2
    assert ident.version == 42
    assert ident.authority_hash == "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"


def test_authority_identity_serialization():
    ident = AuthorityStateIdentity(version=1, authority_hash="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef")
    s = json.dumps(ident.to_dict())
    loaded = json.loads(s)
    assert loaded["version"] == 1
    assert loaded["authority_hash"] == "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"


def test_authority_identity_changes_on_graph():
    rt = LGAERuntime(_graph(), _cfg())
    before = rt.state_identity
    assert isinstance(before, AuthorityStateIdentity)
    
    # Mutate graph
    shadow = rt.engine.graph.clone()
    shadow.weight[0] = shadow.weight[0] * 2.0
    tx = make_graph_transaction(
        base_state_version=before.version,
        base_state_hash=before.authority_hash,
        shadow_graph=shadow,
        mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        step=0,
    ).with_authorization()
    auth = AuthorizationResult(
        snapshot_id="s",
        state_version=before.version,
        state_hash=before.authority_hash,
        status=AuthorizationStatus.AUTHORIZED,
        transaction_hash=tx.transaction_id,
    )
    res = rt.commit_channel.commit(tx, auth)
    after = rt.state_identity
    assert after.version == before.version + 1
    assert after.authority_hash != before.authority_hash
    assert res.pre_identity == before
    assert res.post_identity == after


def test_authority_identity_changes_on_fiber():
    rt = LGAERuntime(_graph(), _cfg())
    before = rt.state_identity
    
    # Mutate fiber
    snap = rt.engine.fibers.snapshot()
    snap.latent.data.add_(0.5)
    tx = make_fiber_transaction(
        base_state_version=before.version,
        base_state_hash=before.authority_hash,
        shadow_fiber_snapshot=snap,
        action="spawn_fiber",
        mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        step=0,
    ).with_authorization()
    auth = AuthorizationResult(
        snapshot_id="s",
        state_version=before.version,
        state_hash=before.authority_hash,
        status=AuthorizationStatus.AUTHORIZED,
        transaction_hash=tx.transaction_id,
    )
    res = rt.commit_channel.commit(tx, auth)
    after = rt.state_identity
    assert after.authority_hash != before.authority_hash
    assert res.pre_identity == before
    assert res.post_identity == after


def test_authority_identity_changes_on_gauge():
    cfg = _cfg()
    cfg.fiber.gauge_dim = 2
    rt = LGAERuntime(_graph(), cfg)
    if rt.engine.gauge_connections is None:
        pytest.skip("No gauge connections initialized")
    before = rt.state_identity
    
    # Mutate gauge
    raw = rt.engine.gauge_connections.raw_generators.clone()
    raw.add_(0.1)
    tx = make_gauge_transaction(
        base_state_version=before.version,
        base_state_hash=before.authority_hash,
        shadow_gauge_raw=raw,
        action="rotate_gauge",
        mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        step=0,
    ).with_authorization()
    auth = AuthorizationResult(
        snapshot_id="s",
        state_version=before.version,
        state_hash=before.authority_hash,
        status=AuthorizationStatus.AUTHORIZED,
        transaction_hash=tx.transaction_id,
    )
    res = rt.commit_channel.commit(tx, auth)
    after = rt.state_identity
    assert after.authority_hash != before.authority_hash
    assert res.pre_identity == before
    assert res.post_identity == after


def test_authority_identity_version_monotonic():
    rt = LGAERuntime(_graph(), _cfg())
    v0 = rt.state_identity.version
    for i in range(3):
        cur = rt.state_identity
        shadow = rt.engine.graph.clone()
        shadow.weight[i] = shadow.weight[i] * 1.5
        tx = make_graph_transaction(
            base_state_version=cur.version,
            base_state_hash=cur.authority_hash,
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=i,
        ).with_authorization()
        auth = AuthorizationResult(
            snapshot_id="s",
            state_version=cur.version,
            state_hash=cur.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=tx.transaction_id,
        )
        rt.commit_channel.commit(tx, auth)
        assert rt.state_identity.version == cur.version + 1
    assert rt.state_identity.version == v0 + 3


def test_transaction_identity_binding():
    rt = LGAERuntime(_graph(), _cfg())
    cur = rt.state_identity
    
    # Create transaction with wrong base state hash
    shadow = rt.engine.graph.clone()
    shadow.weight[0] = shadow.weight[0] * 2.0
    tx = make_graph_transaction(
        base_state_version=cur.version,
        base_state_hash="0000000000000000000000000000000000000000000000000000000000000000",
        shadow_graph=shadow,
        mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        step=0,
    ).with_authorization()
    auth = AuthorizationResult(
        snapshot_id="s",
        state_version=cur.version,
        state_hash=cur.authority_hash,
        status=AuthorizationStatus.AUTHORIZED,
        transaction_hash=tx.transaction_id,
    )
    with pytest.raises(StaleTransactionError):
        rt.commit_channel.commit(tx, auth)
