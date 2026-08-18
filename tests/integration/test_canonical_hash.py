"""v5.11-RC Phase 5: Complete authoritative state (canonical_hash) tests.

Tests that StateBundle.canonical_hash:
- Covers all authoritative state fields
- Is deterministic
- Changes when any component changes
- Distinguishes absent fields from empty fields
"""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime
from lgae_v3.runtime.state.state_bundle import StateBundle
from lgae_v3.runtime.state.authoritative_state import (
    AuthoritativeState, CalibrationState, ModelReference,
)


def _cfg() -> ResearchConfig:
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 3
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


def _make_bundle():
    """Create a StateBundle directly from a fresh runtime."""
    torch.manual_seed(42)
    rt = LGAERuntime(_graph(), _cfg())
    return StateBundle(
        graph=rt._engine.graph,
        fibers=rt._engine.fibers,
        gauges=rt._engine.gauge_connections,
        calibration=CalibrationState(),
        model_ref=ModelReference(),
        version=0,
    )


def _make_bundle_2():
    """Create a second StateBundle from a fresh runtime (same seed)."""
    torch.manual_seed(42)
    rt = LGAERuntime(_graph(), _cfg())
    return StateBundle(
        graph=rt._engine.graph,
        fibers=rt._engine.fibers,
        gauges=rt._engine.gauge_connections,
        calibration=CalibrationState(),
        model_ref=ModelReference(),
        version=0,
    )


class TestCanonicalHash:
    """StateBundle.canonical_hash covers complete authoritative state."""

    def test_canonical_hash_is_deterministic(self):
        """canonical_hash is deterministic for the same state."""
        bundle1 = _make_bundle()
        bundle2 = _make_bundle_2()
        assert bundle1.canonical_hash == bundle2.canonical_hash

    def test_canonical_hash_changes_with_graph(self):
        """canonical_hash changes when graph changes."""
        bundle1 = _make_bundle()
        bundle2 = _make_bundle_2()
        bundle2.graph.weight[0] *= 3.0
        assert bundle1.canonical_hash != bundle2.canonical_hash

    def test_canonical_hash_changes_with_fiber(self):
        """canonical_hash changes when fiber state changes."""
        bundle1 = _make_bundle()
        bundle2 = _make_bundle_2()
        if hasattr(bundle2.fibers, 'latent'):
            bundle2.fibers.latent.data.fill_(0.5)
        assert bundle1.canonical_hash != bundle2.canonical_hash

    def test_canonical_hash_changes_with_gauge(self):
        """canonical_hash changes when gauge state changes."""
        bundle1 = _make_bundle()
        bundle2 = _make_bundle_2()
        if bundle2.gauges is not None and hasattr(bundle2.gauges, 'raw_generators'):
            bundle2.gauges.raw_generators.data.fill_(0.123)
        assert bundle1.canonical_hash != bundle2.canonical_hash

    def test_canonical_hash_changes_with_version(self):
        """canonical_hash changes when version changes."""
        bundle1 = _make_bundle()
        bundle2 = _make_bundle_2()
        bundle2.version = 1
        assert bundle1.canonical_hash != bundle2.canonical_hash

    def test_canonical_hash_differs_from_state_hash(self):
        """canonical_hash includes more fields than state_hash."""
        bundle = _make_bundle()
        # canonical_hash should be different from state_hash because it
        # includes extended state fields.
        assert bundle.canonical_hash != bundle.state_hash

    def test_canonical_hash_includes_extended_fields(self):
        """canonical_hash includes extended state fields when attached."""
        bundle1 = _make_bundle()
        bundle2 = _make_bundle_2()
        # Attach an extended field.
        object.__setattr__(bundle2, 'step_index', 42)
        # canonical_hash should differ because step_index is now present.
        assert bundle1.canonical_hash != bundle2.canonical_hash

    def test_canonical_hash_distinguishes_absent_from_empty(self):
        """canonical_hash distinguishes absent fields from empty fields."""
        bundle1 = _make_bundle()
        bundle2 = _make_bundle_2()
        # bundle1 has no quarantine field (absent).
        # bundle2 has an empty quarantine list (present but empty).
        object.__setattr__(bundle2, 'quarantine', [])
        assert bundle1.canonical_hash != bundle2.canonical_hash
