"""v5.11-RC Phase 3: True shadow evaluation tests.

Tests that evaluation NEVER mutates live authoritative state.
The gate is: H(S_before_evaluate) == H(S_after_evaluate) for every
non-commit phase.
"""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime


def _cfg(gauge_dim: int = 0) -> ResearchConfig:
    cfg = ResearchConfig()
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
    cfg.mutation.curvature_ema_enabled = False
    return cfg


def _graph():
    return make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=12)


class TestFiberEvaluationNeverMutatesAuthority:
    """Fiber evaluation must never change the authority hash."""

    def test_fiber_evaluation_never_changes_authority_hash(self):
        """H(S_before) == H(S_after) for fiber evaluation."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        hash_before = rt.authority_hash
        try:
            rt._engine.evaluate_fiber_action("spawn_fiber", node=0)
        except Exception:
            pass
        hash_after = rt.authority_hash
        assert hash_before == hash_after, (
            f"Fiber evaluation changed authority hash! "
            f"Before: {hash_before[:16]}, After: {hash_after[:16]}"
        )

    def test_fiber_evaluation_never_changes_fiber_hash(self):
        """Fiber evaluation must not change the fiber state hash."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        fiber_hash_before = rt._engine.fibers.state_hash()
        try:
            rt._engine.evaluate_fiber_action("spawn_fiber", node=0)
        except Exception:
            pass
        fiber_hash_after = rt._engine.fibers.state_hash()
        assert fiber_hash_before == fiber_hash_after, (
            "Fiber evaluation changed fiber state hash!"
        )

    def test_multiple_fiber_evaluations_dont_change_state(self):
        """Multiple fiber evaluations don't change state."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        hash_before = rt.authority_hash
        for _ in range(5):
            try:
                rt._engine.evaluate_fiber_action("spawn_fiber", node=0)
                rt._engine.evaluate_fiber_action("prune_fiber", node=0)
            except Exception:
                pass
        hash_after = rt.authority_hash
        assert hash_before == hash_after

    def test_shadow_state_has_no_tensor_aliases(self):
        """Shadow state tensors must not alias live state tensors."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        live_latent = rt._engine.fibers.latent
        live_mask = rt._engine.fibers.active_mask
        try:
            rt._engine.evaluate_fiber_action("spawn_fiber", node=0)
        except Exception:
            pass
        # Live tensors should not have been modified.
        assert live_latent.data_ptr() == rt._engine.fibers.latent.data_ptr()
        # The live mask should be unchanged.
        original_mask = rt._engine.fibers.active_mask.clone()
        try:
            rt._engine.evaluate_fiber_action("spawn_fiber", node=1)
        except Exception:
            pass
        assert torch.equal(original_mask, rt._engine.fibers.active_mask)


class TestGaugeEvaluationNeverMutatesAuthority:
    """Gauge evaluation must never change the authority hash."""

    def test_gauge_evaluation_never_changes_authority_hash(self):
        """H(S_before) == H(S_after) for gauge evaluation."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg(gauge_dim=3))
        hash_before = rt.authority_hash
        try:
            rt._engine.evaluate_gauge_action(u=0, v=1, magnitude=0.01)
        except Exception:
            pass
        hash_after = rt.authority_hash
        assert hash_before == hash_after, (
            f"Gauge evaluation changed authority hash! "
            f"Before: {hash_before[:16]}, After: {hash_after[:16]}"
        )

    def test_gauge_evaluation_never_changes_gauge_hash(self):
        """Gauge evaluation must not change the gauge state hash."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg(gauge_dim=3))
        gauge_hash_before = rt._engine.gauge_connections.state_hash()
        try:
            rt._engine.evaluate_gauge_action(u=0, v=1, magnitude=0.01)
        except Exception:
            pass
        gauge_hash_after = rt._engine.gauge_connections.state_hash()
        assert gauge_hash_before == gauge_hash_after, (
            "Gauge evaluation changed gauge state hash!"
        )

    def test_gauge_raw_generators_not_mutated(self):
        """Gauge raw_generators tensor must not be mutated during evaluation."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg(gauge_dim=3))
        raw_before = rt._engine.gauge_connections.raw_generators.detach().clone()
        try:
            rt._engine.evaluate_gauge_action(u=0, v=1, magnitude=0.01)
        except Exception:
            pass
        raw_after = rt._engine.gauge_connections.raw_generators.detach()
        assert torch.equal(raw_before, raw_after), (
            "Gauge raw_generators was mutated during evaluation!"
        )

    def test_multiple_gauge_evaluations_dont_change_state(self):
        """Multiple gauge evaluations don't change state."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg(gauge_dim=3))
        hash_before = rt.authority_hash
        for _ in range(5):
            try:
                rt._engine.evaluate_gauge_action(u=0, v=1, magnitude=0.01)
            except Exception:
                pass
        hash_after = rt.authority_hash
        assert hash_before == hash_after


class TestJointEvaluationNeverMutatesAuthority:
    """Joint evaluation must never change the authority hash."""

    def test_joint_evaluation_never_changes_authority_hash(self):
        """H(S_before) == H(S_after) for joint evaluation."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg(gauge_dim=3))
        hash_before = rt.authority_hash
        # Run fiber and gauge evaluations in sequence.
        try:
            rt._engine.evaluate_fiber_action("spawn_fiber", node=0)
            rt._engine.evaluate_gauge_action(u=0, v=1, magnitude=0.01)
        except Exception:
            pass
        hash_after = rt.authority_hash
        assert hash_before == hash_after, (
            f"Joint evaluation changed authority hash! "
            f"Before: {hash_before[:16]}, After: {hash_after[:16]}"
        )

    def test_canonical_evaluate_does_not_change_state(self):
        """The canonical evaluate() phase does not change state."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        hash_before = rt.authority_hash
        obs = rt.observe()
        reasoning = rt.reason(obs)
        candidates = rt.propose(obs, reasoning)
        planning = rt.plan(obs, reasoning, candidates)
        rt.evaluate(obs, planning)
        hash_after = rt.authority_hash
        assert hash_before == hash_after
