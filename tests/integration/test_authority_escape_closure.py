"""v5.11-RC Phase 1: Authority escape closure tests.

Tests that external code cannot obtain a mutable authoritative object
through supported runtime APIs.

Attack vectors tested:
- facade._engine attribute access
- guard._engine attribute access
- object.__getattribute__ bypass
- reflection attacks
- public facade cannot change state hash
"""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime
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


class TestEngineFacadeNoRawEngineEscape:
    """facade._engine must not be accessible."""

    def test_facade_engine_attribute_blocked(self):
        """Accessing facade._engine raises UnauthorizedMutationError."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        facade = rt.engine
        with pytest.raises(UnauthorizedMutationError):
            _ = facade._engine

    def test_facade_engine_not_in_dict(self):
        """The facade has no __dict__ (uses __slots__)."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        facade = rt.engine
        assert not hasattr(facade, "__dict__")

    def test_public_facade_cannot_change_state_hash(self):
        """No supported API on the facade can change the authority hash."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        hash_before = rt.authority_hash
        facade = rt.engine
        # Try every read method — none should change state.
        _ = facade.graph
        _ = facade.fibers
        _ = facade.gauge_connections
        _ = facade.step_index
        _ = facade.num_nodes
        _ = facade.config
        _ = facade.audit()
        _ = facade.authority_hash()
        _ = facade.state_hash()
        _ = facade.fibers_raw()
        hash_after = rt.authority_hash
        assert hash_before == hash_after, (
            "Public facade API changed authority hash!"
        )

    def test_facade_setattr_blocked(self):
        """Setting any attribute on the facade raises."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        facade = rt.engine
        with pytest.raises(UnauthorizedMutationError):
            facade.graph = _graph()
        with pytest.raises(UnauthorizedMutationError):
            facade._engine = None

    def test_facade_delattr_blocked(self):
        """Deleting any attribute on the facade raises."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        facade = rt.engine
        with pytest.raises(UnauthorizedMutationError):
            del facade._engine


class TestGuardNoRawEngineEscape:
    """guard._engine must not be accessible."""

    def test_guard_engine_attribute_blocked(self):
        """Accessing guard._engine raises UnauthorizedMutationError."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        guard = rt.guard_for("executive")
        with pytest.raises(UnauthorizedMutationError):
            _ = guard._engine

    def test_guard_setattr_blocked(self):
        """Setting any attribute on the guard raises."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        guard = rt.guard_for("executive")
        with pytest.raises(UnauthorizedMutationError):
            guard._engine = None
        with pytest.raises(UnauthorizedMutationError):
            guard.custom = 123

    def test_guard_delattr_blocked(self):
        """Deleting any attribute on the guard raises."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        guard = rt.guard_for("executive")
        with pytest.raises(UnauthorizedMutationError):
            del guard._engine


class TestReflectionAttack:
    """object.__getattribute__ bypass attempts."""

    def test_reflection_attack_on_facade(self):
        """object.__getattribute__(facade, '_engine') still works but
        is an explicit bypass — document that this is the boundary."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        facade = rt.engine
        # The supported API blocks _engine. An explicit object.__getattribute__
        # bypass is the boundary of what Python can enforce in-process.
        # This test documents that boundary honestly.
        raw = object.__getattribute__(facade, "_engine")
        assert raw is not None
        # But the normal access path is blocked.
        with pytest.raises(UnauthorizedMutationError):
            _ = facade._engine

    def test_reflection_attack_on_guard(self):
        """object.__getattribute__(guard, '_engine') still works but
        is an explicit bypass — document that this is the boundary."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        guard = rt.guard_for("executive")
        raw = object.__getattribute__(guard, "_engine")
        assert raw is not None
        # But the normal access path is blocked.
        with pytest.raises(UnauthorizedMutationError):
            _ = guard._engine

    def test_normal_access_does_not_reach_mutation_api(self):
        """Normal facade usage does not expose mutation methods."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        facade = rt.engine
        # Mutation methods should raise.
        with pytest.raises(UnauthorizedMutationError):
            facade.evaluate_and_maybe_commit(None)
        with pytest.raises(UnauthorizedMutationError):
            facade.evaluate_fiber_action("spawn", node=0)
        with pytest.raises(UnauthorizedMutationError):
            facade.evaluate_gauge_action("test", node=0)
        with pytest.raises(UnauthorizedMutationError):
            facade.resolve_quarantine()
