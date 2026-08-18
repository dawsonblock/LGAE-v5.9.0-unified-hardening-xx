from lgae_v3.cache_coherence import (
    ChangeKind, CommitEventBus, GraphCommitEvent,
    LocalizedGenerationCache, SpatialCacheDependency,
)


def _line_neighbors(seeds, radius):
    out = set(seeds)
    frontier = set(seeds)
    for _ in range(radius):
        nxt = set()
        for n in frontier:
            if n > 0: nxt.add(n - 1)
            if n < 9: nxt.add(n + 1)
        out |= nxt; frontier = nxt
    return out


def test_radius_one_invalidation_preserves_untouched_partition():
    bus = CommitEventBus()
    cache = LocalizedGenerationCache[int, str](
        dependency=SpatialCacheDependency(ChangeKind.TOPOLOGY, radius=1),
        neighborhood_resolver=_line_neighbors,
    )
    cache.bind(0)
    for n in range(10): cache.put(n, f"v{n}", generation=0)
    bus.register(cache)
    bus.publish(GraphCommitEvent(1, ChangeKind.TOPOLOGY, changed_nodes=(5,)))
    assert set(cache.last_invalidated) == {4, 5, 6}
    assert cache.get(0, generation=1) == "v0"
    assert cache.get(9, generation=1) == "v9"
    assert 5 not in cache.keys()


def test_radius_two_matches_two_hop_dirty_region():
    cache = LocalizedGenerationCache[int, int](
        dependency=SpatialCacheDependency(ChangeKind.TOPOLOGY, radius=2),
        neighborhood_resolver=_line_neighbors,
    )
    cache.bind(0)
    for n in range(10): cache.put(n, n, generation=0)
    cache.on_graph_commit(GraphCommitEvent(1, ChangeKind.TOPOLOGY, changed_nodes=(5,)))
    assert set(cache.last_invalidated) == {3, 4, 5, 6, 7}
    assert set(cache.keys()) == {0, 1, 2, 8, 9}


def test_global_dependency_is_conservatively_flushed():
    cache = LocalizedGenerationCache[int, int](
        dependency=SpatialCacheDependency(ChangeKind.TOPOLOGY, radius=None)
    )
    cache.bind(0); cache.put(1, 1, generation=0); cache.put(8, 8, generation=0)
    cache.on_graph_commit(GraphCommitEvent(1, ChangeKind.TOPOLOGY, changed_nodes=(1,)))
    assert cache.keys() == ()
    assert cache.bound_generation == 1


def test_irrelevant_change_advances_authority_without_eviction():
    cache = LocalizedGenerationCache[int, str](
        dependency=SpatialCacheDependency(ChangeKind.TOPOLOGY, radius=1),
        neighborhood_resolver=_line_neighbors,
    )
    cache.bind(0); cache.put(2, "keep", generation=0)
    cache.on_graph_commit(GraphCommitEvent(1, ChangeKind.WEIGHTS, changed_nodes=(2,)))
    assert cache.get(2, generation=1) == "keep"
    assert not cache.last_invalidated


def test_missing_locality_metadata_fails_closed():
    cache = LocalizedGenerationCache[int, int](
        dependency=SpatialCacheDependency(ChangeKind.TOPOLOGY, radius=1),
        neighborhood_resolver=_line_neighbors,
    )
    cache.bind(0); cache.put(2, 2, generation=0)
    cache.on_graph_commit(GraphCommitEvent(1, ChangeKind.TOPOLOGY))
    assert cache.keys() == ()
