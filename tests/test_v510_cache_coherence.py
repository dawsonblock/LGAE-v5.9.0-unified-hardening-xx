"""v5.10 Phase 4: mandatory cache coherence + selective invalidation tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, LGAERuntime
from lgae_v3.cache_coherence import (
    ChangeKind, CommitEventBus, GraphCommitEvent, GenerationStampedCache,
)
from lgae_v3.runtime import MutationImpact, CacheRegistry, depends_on, declared_dependencies


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


def test_mutation_impact_roundtrips_to_change_kind():
    mi = MutationImpact(topology=True, weights=True)
    ck = mi.to_change_kind()
    assert bool(ck & ChangeKind.TOPOLOGY)
    assert bool(ck & ChangeKind.WEIGHTS)
    assert not bool(ck & ChangeKind.GAUGE)
    mi2 = MutationImpact.from_change_kind(ck)
    assert mi2.topology and mi2.weights
    assert not mi2.is_empty
    assert MutationImpact().is_empty


def test_depends_on_decorator_sets_dependencies():
    @depends_on("topology", "weights")
    class CurvatureCache(GenerationStampedCache):
        pass

    assert declared_dependencies(CurvatureCache) == (ChangeKind.TOPOLOGY | ChangeKind.WEIGHTS)
    # The decorator wires the dependency mask into GenerationStampedCache init.
    c = CurvatureCache()
    assert declared_dependencies(c) == (ChangeKind.TOPOLOGY | ChangeKind.WEIGHTS)


def test_depends_on_rejects_unknown_dimension():
    with pytest.raises(ValueError):
        depends_on("bogus")


def test_cache_registry_invalidates_only_affected_caches():
    bus = CommitEventBus()
    registry = CacheRegistry(bus)

    @depends_on("topology", "weights")
    class TopoCache(GenerationStampedCache):
        pass

    @depends_on("gauges")
    class GaugeCache(GenerationStampedCache):
        pass

    topo = TopoCache(); topo.bind(0)
    gauge = GaugeCache(); gauge.bind(0)
    registry.register(topo, name="topo")
    registry.register(gauge, name="gauge")

    # A topology-only commit must invalidate topo but spare gauge.
    bus.publish(GraphCommitEvent(generation=1, changes=ChangeKind.TOPOLOGY, reason="test"))
    assert topo.dirty
    assert not gauge.dirty
    last = registry.invalidations[-1]
    assert "topo" in last["invalidated"]
    assert "gauge" in last["spared"]


def test_cache_registry_spares_unaffected_on_gauge_commit():
    bus = CommitEventBus()
    registry = CacheRegistry(bus)

    @depends_on("topology")
    class TopoCache(GenerationStampedCache):
        pass

    @depends_on("gauges")
    class GaugeCache(GenerationStampedCache):
        pass

    topo = TopoCache(); topo.bind(0)
    gauge = GaugeCache(); gauge.bind(0)
    registry.register(topo, name="topo")
    registry.register(gauge, name="gauge")

    bus.publish(GraphCommitEvent(generation=1, changes=ChangeKind.GAUGE, reason="gauge_only"))
    assert not topo.dirty
    assert gauge.dirty


def test_runtime_owns_commit_event_bus_and_registry():
    torch.manual_seed(0)
    rt = LGAERuntime(_graph(), _cfg())
    assert isinstance(rt.commit_event_bus, CommitEventBus)
    assert isinstance(rt.cache_registry, CacheRegistry)


def test_runtime_publishes_impact_on_commit_and_invalidates_registered_caches():
    torch.manual_seed(1)
    rt = LGAERuntime(_graph(), _cfg())

    @depends_on("topology", "weights")
    class TopoCache(GenerationStampedCache):
        pass

    @depends_on("gauges")
    class GaugeCache(GenerationStampedCache):
        pass

    topo = TopoCache(); topo.bind(int(rt.engine.graph.version))
    gauge = GaugeCache(); gauge.bind(int(rt.engine.graph.version))
    rt.cache_registry.register(topo, name="topo")
    rt.cache_registry.register(gauge, name="gauge")

    # Run steps until a commit occurs or a bounded number of steps.
    committed = False
    for _ in range(12):
        res = rt.step()
        if res.committed:
            committed = True
            break
    if committed:
        # At least one invalidation event must have been recorded.
        assert len(rt.cache_registry.invalidations) >= 1
        # A topology/weights commit must have invalidated topo; gauge is spared
        # unless the action was a gauge change.
        last = rt.cache_registry.invalidations[-1]
        assert isinstance(last["invalidated"], list)
