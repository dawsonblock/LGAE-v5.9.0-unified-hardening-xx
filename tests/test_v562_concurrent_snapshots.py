import threading
import time

import pytest

from lgae_v3.cache_coherence import (
    ChangeKind,
    CommitEventBus,
    GenerationStampedCache,
    GraphReadCoordinator,
    GraphReadView,
    StaleReadError,
    run_consistent_read,
)
from lgae_v3.transactions import graph_transaction, journaled_graph_transaction
from lgae_v3.types import make_graph_buffers


def _graph():
    return make_graph_buffers(5, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)], capacity=8)


def test_read_view_rejects_result_that_overlaps_committed_write():
    g = _graph()
    rc = GraphReadCoordinator()
    started = threading.Event()
    proceed = threading.Event()
    result = {}

    def reader():
        try:
            with GraphReadView(rc, lambda: g.version):
                started.set()
                proceed.wait(timeout=2)
                _ = float(g.weight[0])
            result["ok"] = True
        except Exception as exc:  # test captures exact stale class below
            result["exc"] = exc

    t = threading.Thread(target=reader)
    t.start()
    assert started.wait(timeout=2)
    with journaled_graph_transaction(g, read_coordinator=rc) as tx:
        tx.set_slot(0, weight=2.0, bump_generation=False)
        tx.commit()
    proceed.set()
    t.join(timeout=2)
    assert isinstance(result.get("exc"), StaleReadError)
    assert g.version == 1


def test_reader_cannot_start_while_writer_epoch_is_open():
    g = _graph()
    rc = GraphReadCoordinator()
    with journaled_graph_transaction(g, read_coordinator=rc) as tx:
        tx.set_slot(0, weight=2.0, bump_generation=False)
        with pytest.raises(StaleReadError, match="mutation is in progress"):
            with GraphReadView(rc, lambda: g.version):
                pass
        tx.commit()
    with GraphReadView(rc, lambda: g.version) as view:
        assert view.generation == 1


def test_rollback_invalidates_overlapping_read_but_restores_generation():
    g = _graph()
    rc = GraphReadCoordinator()
    before = g.state_hash()
    token = rc.begin_read(g.version)
    with journaled_graph_transaction(g, read_coordinator=rc) as tx:
        tx.set_slot(0, weight=9.0, bump_generation=False)
        # no commit => rollback on exit
        pass
    assert g.state_hash() == before
    assert g.version == 0
    with pytest.raises(StaleReadError):
        rc.validate(token, g.version)
    # A new read is clean after rollback.
    with GraphReadView(rc, lambda: g.version):
        pass


def test_run_consistent_read_retries_after_writer_finishes():
    g = _graph()
    rc = GraphReadCoordinator()
    rc.begin_write()
    attempts = {"n": 0}

    def release_writer():
        time.sleep(0.03)
        rc.end_write()

    t = threading.Thread(target=release_writer)
    t.start()

    def compute():
        attempts["n"] += 1
        return float(g.weight[0])

    value = run_consistent_read(rc, lambda: g.version, compute, max_retries=20, retry_delay_s=0.005)
    t.join(timeout=2)
    assert value == pytest.approx(1.0)
    assert attempts["n"] == 1  # compute is not entered while writer epoch is odd


def test_snapshot_transaction_integrates_read_epoch_and_commit_bus():
    g = _graph()
    rc = GraphReadCoordinator()
    bus = CommitEventBus()
    cache = GenerationStampedCache(dependencies=ChangeKind.WEIGHTS)
    cache.bind(g.version)
    bus.register(cache)

    with graph_transaction(g, event_bus=bus, changes=ChangeKind.WEIGHTS, read_coordinator=rc) as tx:
        g.weight[0] = 3.0
        tx.commit()

    assert g.version == 1
    assert bus.last_generation == 1
    assert rc.mutation_epoch == 2
    assert rc.writer_active is False
    assert cache.dirty is True


def test_commit_bus_is_thread_safe_for_strictly_ordered_publications():
    bus = CommitEventBus()
    cache = GenerationStampedCache(dependencies=ChangeKind.WEIGHTS)
    cache.bind(0)
    bus.register(cache)
    # Publish from distinct threads but enforce event ordering externally; the bus
    # itself must maintain monotonicity without corrupting its consumer list.
    for generation in range(1, 21):
        t = threading.Thread(
            target=lambda gen=generation: bus.publish(
                __import__("lgae_v3.cache_coherence", fromlist=["GraphCommitEvent"]).GraphCommitEvent(
                    gen, ChangeKind.WEIGHTS
                )
            )
        )
        t.start(); t.join(timeout=2)
    assert bus.last_generation == 20
    assert cache.bound_generation == 20
    assert cache.dirty is True
