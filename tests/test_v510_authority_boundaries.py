"""v5.10 Phase 2: strict authority boundary regression tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import (
    ResearchConfig,
    make_graph_buffers,
    LGAERuntime,
)
from lgae_v3.runtime import (
    AuthorityRole,
    AuthorityBoundary,
    AuthoritativeStateGuard,
    CommitChannel,
    UnauthorizedMutationError,
    DEFAULT_BOUNDARIES,
)
from lgae_v3.runtime.authority import DEFAULT_BOUNDARIES as BOUNDARIES


def _cfg():
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 0
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
    return make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=12)


def test_default_boundary_classifies_components():
    assert BOUNDARIES["executive"] == AuthorityRole.PROPOSAL
    assert BOUNDARIES["governor"] == AuthorityRole.VERIFICATION
    assert BOUNDARIES["engine"] == AuthorityRole.COMMIT
    assert BOUNDARIES["ann_retrieval"] == AuthorityRole.PROPOSAL
    assert BOUNDARIES["exact_orc_lly"] == AuthorityRole.VERIFICATION


def test_boundary_rejects_mutation_by_non_commit():
    b = AuthorityBoundary()
    b.register("my_proposer", AuthorityRole.PROPOSAL)
    with pytest.raises(UnauthorizedMutationError):
        b.assert_can_mutate("my_proposer")
    # commit role can mutate
    b.register("my_engine", AuthorityRole.COMMIT)
    b.assert_can_mutate("my_engine")  # no raise


def test_boundary_rejects_verification_by_proposal():
    b = AuthorityBoundary()
    b.register("p", AuthorityRole.PROPOSAL)
    with pytest.raises(UnauthorizedMutationError):
        b.assert_can_verify("p")
    b.register("v", AuthorityRole.VERIFICATION)
    b.assert_can_verify("v")  # no raise


def test_guard_blocks_attribute_mutation():
    torch.manual_seed(0)
    rt = LGAERuntime(_graph(), _cfg())
    guard = rt.guard_for("executive")
    assert guard.role == AuthorityRole.PROPOSAL
    with pytest.raises(UnauthorizedMutationError):
        guard.graph = "x"  # type: ignore[assignment]
    # Read-only access works.
    snap = guard.snapshot()
    assert snap.authority_hash == rt.authority_hash
    assert guard.graph.num_nodes == 6


def test_guard_for_commit_role_is_rejected():
    torch.manual_seed(1)
    rt = LGAERuntime(_graph(), _cfg())
    with pytest.raises(UnauthorizedMutationError):
        rt.guard_for("engine")


def test_commit_channel_only_for_commit_role():
    b = AuthorityBoundary()
    b.register("p", AuthorityRole.PROPOSAL)
    with pytest.raises(UnauthorizedMutationError):
        CommitChannel(object(), b, component="p")


def test_runtime_commit_channel_delegates_to_engine():
    torch.manual_seed(2)
    rt = LGAERuntime(_graph(), _cfg())
    ch = rt.commit_channel
    assert ch.authority_hash() == rt.authority_hash
    snap = ch.snapshot()
    assert snap.authority_hash == rt.authority_hash


def test_runtime_step_enforces_authority_no_regression():
    torch.manual_seed(3)
    rt = LGAERuntime(_graph(), _cfg())
    res = rt.step()
    # The runtime must complete a full step without authority violations.
    assert res is not None
    assert rt.boundary.role_of("engine") == AuthorityRole.COMMIT
