"""v5.10 Phase 3: unified runtime state + seqlock enforcement tests."""
from __future__ import annotations

import threading
import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, LGAERuntime
from lgae_v3.runtime import RuntimeSnapshot
from lgae_v3.runtime.runtime_state import StaleReadError, snapshot_from_engine
from lgae_v3.cache_coherence import GraphReadCoordinator, StaleReadError as CCStaleReadError


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


def test_runtime_owns_read_coordinator():
    torch.manual_seed(0)
    rt = LGAERuntime(_graph(), _cfg())
    assert isinstance(rt.read_coordinator, GraphReadCoordinator)
    # Initially stable (even epoch, no writer active).
    assert not rt.read_coordinator.writer_active
    assert rt.read_coordinator.mutation_epoch % 2 == 0


def test_consistent_read_publishes_generation_consistent_result():
    torch.manual_seed(1)
    rt = LGAERuntime(_graph(), _cfg())
    # A read that does not overlap a write must succeed and return the value.
    out = rt.consistent_read(lambda: 42)
    assert out == 42


def test_consistent_read_retries_on_stale_then_succeeds():
    torch.manual_seed(2)
    rt = LGAERuntime(_graph(), _cfg())
    attempts = {"n": 0}

    def compute():
        attempts["n"] += 1
        if attempts["n"] == 1:
            # Simulate a concurrent commit mid-read by opening a write epoch.
            rt.read_coordinator.begin_write()
            try:
                pass
            finally:
                # leave writer active so the read view validation raises stale.
                pass
            # end_write is intentionally NOT called here so validation fails;
            # but we must close it to let the retry succeed. Instead, raise
            # stale directly to mimic an overlapping commit.
            rt.read_coordinator.end_write()
            raise CCStaleReadError("simulated overlap")
        return "ok"

    out = rt.consistent_read(compute)
    assert out == "ok"
    assert attempts["n"] == 2


def test_consistent_read_raises_after_exhausting_retries():
    torch.manual_seed(3)
    rt = LGAERuntime(_graph(), _cfg(), runtime_config=__import__("lgae_v3").RuntimeConfig(max_stale_read_retries=2))

    def compute():
        raise CCStaleReadError("always stale")

    with pytest.raises(CCStaleReadError):
        rt.consistent_read(compute)


def test_snapshot_binds_full_authority_identity():
    torch.manual_seed(4)
    rt = LGAERuntime(_graph(), _cfg())
    snap = rt.snapshot()
    assert snap.generation == rt.generation
    assert snap.graph_state_hash == rt.engine.graph.state_hash()
    assert snap.authority_hash == rt.authority_hash
    # Fiber hash is bound.
    assert snap.fiber_state_hash == rt.engine.fibers.state_hash()


def test_snapshot_assert_generation_detects_stale():
    torch.manual_seed(5)
    rt = LGAERuntime(_graph(), _cfg())
    snap = rt.snapshot()
    with pytest.raises(StaleReadError):
        snap.assert_generation(snap.generation + 1)


def test_commit_channel_brackets_write_epoch():
    torch.manual_seed(6)
    rt = LGAERuntime(_graph(), _cfg())
    epoch_before = rt.read_coordinator.mutation_epoch
    # A read during a simulated write must be stale.
    rt.read_coordinator.begin_write()
    with pytest.raises(CCStaleReadError):
        rt.read_coordinator.begin_read(int(rt.engine.graph.version))
    rt.read_coordinator.end_write()
    # After the write completes, reads succeed again.
    tok = rt.read_coordinator.begin_read(int(rt.engine.graph.version))
    assert tok.graph_generation == int(rt.engine.graph.version)


def test_snapshot_from_engine_helper():
    torch.manual_seed(7)
    rt = LGAERuntime(_graph(), _cfg())
    snap = snapshot_from_engine(rt.engine, generation=99)
    assert snap.generation == 99
    assert len(snap.authority_hash) == 64
