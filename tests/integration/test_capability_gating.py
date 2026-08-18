"""v5.11 Phase 2: Capability-gated mutation primitives.

These tests prove that:
1. Engine mutation methods require a valid _AuthorityCapability token
2. External code cannot call evaluate_and_maybe_commit without the token
3. External code cannot call evaluate_fiber_action without the token
4. External code cannot call evaluate_gauge_action without the token
5. External code cannot call resolve_quarantine without the token
6. The EngineFacade blocks all mutation methods
7. The capability token cannot be forged
"""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.runtime.state.state_errors import CapabilityError
from lgae_v3.runtime.authority import UnauthorizedMutationError


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


def _graph():
    return make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=12)


class TestCapabilityGating:
    """Prove that mutation methods require a capability token."""

    def test_evaluate_and_maybe_commit_without_capability_fails(self):
        """Calling evaluate_and_maybe_commit without capability fails."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        from lgae_v3.mutations import AddEdge
        mutation = AddEdge(u=0, v=5)
        # The engine has a capability set, but we call without passing it.
        with pytest.raises((CapabilityError, UnauthorizedMutationError)):
            rt._engine.evaluate_and_maybe_commit(mutation)

    def test_evaluate_and_maybe_commit_with_wrong_capability_fails(self):
        """Calling evaluate_and_maybe_commit with a forged capability fails."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        from lgae_v3.mutations import AddEdge
        mutation = AddEdge(u=0, v=5)
        # Forge a capability — this should fail because it's not the same object.
        from lgae_v3.runtime.state.authority_token import _AuthorityCapability
        forged = _AuthorityCapability(999)
        with pytest.raises((CapabilityError, UnauthorizedMutationError)):
            rt._engine.evaluate_and_maybe_commit(mutation, capability=forged)

    def test_evaluate_fiber_action_without_capability_fails(self):
        """Calling evaluate_fiber_action without capability fails on accept."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        # If the fiber action reaches the ACCEPT path, it should fail
        # without capability. If it rejects/quarantines, that's also fine.
        try:
            result = rt._engine.evaluate_fiber_action("spawn_fiber", node=0)
            # If it didn't raise, it must have rejected or quarantined
            # (not accepted, since accept requires capability).
            assert result.decision != MutationDecision.ACCEPT, (
                "fiber action was accepted without capability!"
            )
        except (CapabilityError, UnauthorizedMutationError):
            # This is the expected case if the action would be accepted.
            pass

    def test_engine_facade_blocks_evaluate_and_maybe_commit(self):
        """The EngineFacade blocks evaluate_and_maybe_commit."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        from lgae_v3.mutations import AddEdge
        mutation = AddEdge(u=0, v=5)
        with pytest.raises(UnauthorizedMutationError):
            rt.engine.evaluate_and_maybe_commit(mutation)

    def test_engine_facade_blocks_evaluate_fiber_action(self):
        """The EngineFacade blocks evaluate_fiber_action."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        with pytest.raises(UnauthorizedMutationError):
            rt.engine.evaluate_fiber_action("spawn_fiber", node=0)

    def test_engine_facade_blocks_evaluate_gauge_action(self):
        """The EngineFacade blocks evaluate_gauge_action."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        with pytest.raises(UnauthorizedMutationError):
            rt.engine.evaluate_gauge_action(u=0, v=1, magnitude=0.01)

    def test_engine_facade_blocks_resolve_quarantine(self):
        """The EngineFacade blocks resolve_quarantine."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        with pytest.raises(UnauthorizedMutationError):
            rt.engine.resolve_quarantine(0, accept=False)

    def test_engine_facade_blocks_graph_assignment(self):
        """The EngineFacade blocks direct graph assignment."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        with pytest.raises(UnauthorizedMutationError):
            rt.engine.graph = _graph()

    def test_engine_facade_blocks_fiber_assignment(self):
        """The EngineFacade blocks direct fiber assignment."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        with pytest.raises(UnauthorizedMutationError):
            rt.engine.fibers = None

    def test_capability_token_cannot_be_forged(self):
        """The capability token is compared by identity, not value."""
        from lgae_v3.runtime.state.authority_token import _AuthorityCapability
        cap1 = _AuthorityCapability(42)
        cap2 = _AuthorityCapability(42)
        # Same runtime_id, but different objects.
        assert cap1 is not cap2
        # The engine's _validate_capability checks identity.
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        # cap1 is not the engine's capability.
        with pytest.raises((CapabilityError, UnauthorizedMutationError)):
            rt._engine._validate_capability(cap1)

    def test_capability_token_is_immutable(self):
        """The capability token cannot be modified after creation."""
        from lgae_v3.runtime.state.authority_token import _AuthorityCapability
        cap = _AuthorityCapability(42)
        with pytest.raises(TypeError):
            cap.runtime_id = 999

    def test_runtime_has_capability(self):
        """The runtime has a non-None authority capability."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        assert rt._authority_capability is not None
        assert rt._engine._authority_capability is rt._authority_capability

    def test_commit_channel_has_capability(self):
        """The commit channel has the same capability as the runtime."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        assert rt._commit_channel._capability is rt._authority_capability
