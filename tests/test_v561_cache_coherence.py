import pytest
import torch

from lgae_v3.cache_coherence import ChangeKind, CommitEventBus, GenerationStampedCache
from lgae_v3.transactions import graph_transaction, journaled_graph_transaction
from lgae_v3.types import make_graph_buffers


def _graph():
    return make_graph_buffers(4, [(0, 1, 1.0), (1, 2, 1.0)], capacity=6)


def test_commit_event_isolation_and_generation_stamp():
    g = _graph()
    bus = CommitEventBus()
    cache = GenerationStampedCache(dependencies=ChangeKind.TOPOLOGY)
    cache.bind(g.version)
    bus.register(cache)

    with journaled_graph_transaction(g, event_bus=bus) as tx:
        tx.set_slot(2, src=2, dst=3, weight=1.0, length=1.0, valid=True)
        # Uncommitted state is private: no event/invalidation yet.
        assert cache.dirty is False
        assert bus.last_generation is None
        tx.commit()

    assert g.version == 1
    assert bus.last_generation == 1
    assert cache.dirty is True
    assert cache.bound_generation == 1
    with pytest.raises(RuntimeError, match="stale derived cache"):
        cache.require_current(g.version)


def test_rollback_emits_no_commit_event():
    g = _graph()
    bus = CommitEventBus()
    cache = GenerationStampedCache(dependencies=ChangeKind.TOPOLOGY)
    cache.bind(g.version)
    bus.register(cache)
    before = g.state_hash()

    with journaled_graph_transaction(g, event_bus=bus) as tx:
        tx.set_slot(2, src=2, dst=3, weight=1.0, length=1.0, valid=True)
        # no commit -> rollback

    assert g.state_hash() == before
    assert bus.last_generation is None
    assert cache.dirty is False
    cache.require_current(g.version)


def test_selective_invalidation_advances_unaffected_cache_generation():
    g = _graph()
    bus = CommitEventBus()
    topology = GenerationStampedCache(dependencies=ChangeKind.TOPOLOGY)
    weights = GenerationStampedCache(dependencies=ChangeKind.WEIGHTS)
    topology.bind(g.version); weights.bind(g.version)
    bus.register(topology); bus.register(weights)

    with journaled_graph_transaction(g, event_bus=bus) as tx:
        tx.set_slot(0, weight=2.0, bump_generation=False)
        tx.commit()

    assert weights.dirty is True
    assert topology.dirty is False
    assert topology.bound_generation == g.version
    topology.require_current(g.version)


def test_direct_snapshot_transaction_publishes_one_new_generation():
    g = _graph()
    bus = CommitEventBus()
    cache = GenerationStampedCache(dependencies=ChangeKind.WEIGHTS)
    cache.bind(g.version); bus.register(cache)
    with graph_transaction(g, event_bus=bus, changes=ChangeKind.WEIGHTS) as tx:
        g.weight[0] = 1.5
        tx.commit()
    assert g.version == 1
    assert bus.last_generation == 1
    assert cache.dirty is True


def test_commit_bus_rejects_non_monotonic_generations():
    from lgae_v3.cache_coherence import GraphCommitEvent
    bus = CommitEventBus()
    bus.publish(GraphCommitEvent(1, ChangeKind.WEIGHTS))
    with pytest.raises(ValueError, match="strictly monotonic"):
        bus.publish(GraphCommitEvent(1, ChangeKind.TOPOLOGY))
