"""Authoritative graph commit events and generation-stamped cache coherence.

v5.6.3 extends v5.6.1 cache invalidation with optimistic read epochs. v5.6.1 centralizes cache invalidation at the transaction commit boundary.  A
rollback never emits a commit event.  Consumers may selectively invalidate on
changes they actually depend on while retaining a graph-generation stamp that
can fail closed when a stale cache is accessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag, auto
from typing import Any, Callable, Generic, Iterable, Protocol, TypeVar
import threading
import time
import weakref


class ChangeKind(IntFlag):
    NONE = 0
    TOPOLOGY = auto()
    WEIGHTS = auto()
    LENGTHS = auto()
    FIBERS = auto()
    LATENTS = auto()
    GAUGE = auto()
    ROLES = auto()
    ALL = TOPOLOGY | WEIGHTS | LENGTHS | FIBERS | LATENTS | GAUGE | ROLES


@dataclass(frozen=True, slots=True)
class GraphCommitEvent:
    generation: int
    changes: ChangeKind
    changed_nodes: tuple[int, ...] = ()
    changed_edges: tuple[tuple[int, int], ...] = ()
    reason: str = "transaction_commit"

    @property
    def topology_changed(self) -> bool:
        return bool(self.changes & ChangeKind.TOPOLOGY)

    @property
    def weights_changed(self) -> bool:
        return bool(self.changes & ChangeKind.WEIGHTS)

    @property
    def fibers_changed(self) -> bool:
        return bool(self.changes & ChangeKind.FIBERS)


class CommitConsumer(Protocol):
    def on_graph_commit(self, event: GraphCommitEvent) -> None: ...


@dataclass(slots=True)
class CacheDependency:
    """Dependency mask for a derived cache."""
    changes: ChangeKind = ChangeKind.ALL

    def affected_by(self, event: GraphCommitEvent) -> bool:
        return bool(self.changes & event.changes)


class CommitEventBus:
    """Weak-reference event bus for commit-only graph invalidation.

    The bus intentionally has no rollback event.  Until commit, mutated state is
    private to the transaction and consumers retain the last authoritative cache.
    """

    def __init__(self) -> None:
        self._consumers: list[weakref.ReferenceType[Any]] = []
        self._last_generation: int | None = None
        self._lock = threading.RLock()

    def register(self, consumer: CommitConsumer) -> None:
        with self._lock:
            self._consumers = [r for r in self._consumers if r() is not None and r() is not consumer]
            self._consumers.append(weakref.ref(consumer))

    def unregister(self, consumer: CommitConsumer) -> None:
        with self._lock:
            self._consumers = [r for r in self._consumers if r() is not None and r() is not consumer]

    @property
    def last_generation(self) -> int | None:
        with self._lock:
            return self._last_generation

    def publish(self, event: GraphCommitEvent) -> None:
        # Serialize generation assignment and take a stable consumer snapshot.
        # Callbacks execute outside the lock so a consumer can safely register,
        # unregister, or perform another non-recursive coherence operation.
        with self._lock:
            if self._last_generation is not None and event.generation <= self._last_generation:
                raise ValueError("commit generations must be strictly monotonic")
            self._last_generation = int(event.generation)
            refs = list(self._consumers)
        live: list[weakref.ReferenceType[Any]] = []
        for ref in refs:
            obj = ref()
            if obj is None:
                continue
            live.append(ref)
            obj.on_graph_commit(event)
        with self._lock:
            known = {id(r()) for r in live if r() is not None}
            # Preserve consumers registered concurrently while dropping dead refs.
            additions = [r for r in self._consumers if r() is not None and id(r()) not in known]
            self._consumers = live + additions


class GenerationStampedCache:
    """Small reusable coherence primitive for derived caches.

    `require_current` raises instead of silently serving stale data.  A consumer
    can rebuild and call `mark_current` after observing a commit event.
    """

    def __init__(self, *, dependencies: ChangeKind = ChangeKind.ALL) -> None:
        self.dependencies = CacheDependency(dependencies)
        self.bound_generation: int | None = None
        self.dirty = True
        self.dirty_reason: str | None = "unbuilt"

    def bind(self, generation: int) -> None:
        self.bound_generation = int(generation)
        self.dirty = False
        self.dirty_reason = None

    def invalidate(self, generation: int, reason: str) -> None:
        self.bound_generation = int(generation)
        self.dirty = True
        self.dirty_reason = str(reason)

    def on_graph_commit(self, event: GraphCommitEvent) -> None:
        if self.dependencies.affected_by(event):
            self.invalidate(event.generation, event.reason)
        elif self.bound_generation is not None:
            # Unaffected caches remain valid but advance their authority stamp.
            self.bound_generation = int(event.generation)

    def require_current(self, generation: int) -> None:
        if self.dirty or self.bound_generation != int(generation):
            raise RuntimeError(
                f"stale derived cache: bound={self.bound_generation}, current={int(generation)}, "
                f"dirty={self.dirty}, reason={self.dirty_reason}"
            )



class StaleReadError(RuntimeError):
    """Raised when a derived computation overlaps an authoritative mutation."""


@dataclass(frozen=True, slots=True)
class ReadEpochToken:
    mutation_epoch: int
    graph_generation: int


class GraphReadCoordinator:
    """Optimistic snapshot coordinator using a seqlock-style mutation epoch.

    The mutation epoch is even while authoritative graph state is stable and odd
    while an in-place writer is active. Readers never block writers: they capture
    an even token, perform a calculation, then validate that neither the mutation
    epoch nor graph generation changed. Results that overlap a write are discarded.

    This is RCU-inspired rather than a full userspace RCU implementation. It is
    appropriate for LGAE's fixed-capacity tensor buffers where writers update slots
    in place and readers can safely retry a derived calculation.
    """

    def __init__(self) -> None:
        self._mutation_epoch = 0
        self._lock = threading.Lock()
        self._writer_active = False

    @property
    def mutation_epoch(self) -> int:
        with self._lock:
            return int(self._mutation_epoch)

    @property
    def writer_active(self) -> bool:
        with self._lock:
            return bool(self._writer_active)

    def begin_write(self) -> int:
        with self._lock:
            if self._writer_active:
                raise RuntimeError("concurrent authoritative writers are not supported")
            self._writer_active = True
            self._mutation_epoch += 1  # odd => unstable
            return int(self._mutation_epoch)

    def end_write(self) -> int:
        with self._lock:
            if not self._writer_active:
                return int(self._mutation_epoch)
            self._writer_active = False
            self._mutation_epoch += 1  # even => stable
            return int(self._mutation_epoch)

    def begin_read(self, graph_generation: int) -> ReadEpochToken:
        with self._lock:
            epoch = int(self._mutation_epoch)
            if self._writer_active or epoch & 1:
                raise StaleReadError("authoritative graph mutation is in progress")
            return ReadEpochToken(epoch, int(graph_generation))

    def validate(self, token: ReadEpochToken, graph_generation: int) -> None:
        with self._lock:
            epoch = int(self._mutation_epoch)
            writer = bool(self._writer_active)
        if writer or (epoch & 1) or epoch != int(token.mutation_epoch) or int(graph_generation) != int(token.graph_generation):
            raise StaleReadError(
                "derived result is stale: "
                f"read_epoch={token.mutation_epoch}, current_epoch={epoch}, "
                f"read_generation={token.graph_generation}, current_generation={int(graph_generation)}"
            )


T = TypeVar("T")


class GraphReadView:
    """Context-managed optimistic read token for one authoritative generation."""

    def __init__(self, coordinator: GraphReadCoordinator, generation_getter: Callable[[], int]) -> None:
        self.coordinator = coordinator
        self.generation_getter = generation_getter
        self.token: ReadEpochToken | None = None

    def __enter__(self) -> "GraphReadView":
        self.token = self.coordinator.begin_read(int(self.generation_getter()))
        return self

    def validate(self) -> None:
        if self.token is None:
            raise RuntimeError("read view has not been entered")
        self.coordinator.validate(self.token, int(self.generation_getter()))

    @property
    def generation(self) -> int:
        if self.token is None:
            raise RuntimeError("read view has not been entered")
        return int(self.token.graph_generation)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.validate()
        return False


def run_consistent_read(
    coordinator: GraphReadCoordinator,
    generation_getter: Callable[[], int],
    compute_fn: Callable[[], T],
    *,
    max_retries: int = 3,
    retry_delay_s: float = 0.0,
) -> T:
    """Run a derived calculation and publish only a generation-consistent result."""
    last: StaleReadError | None = None
    for _ in range(max(1, int(max_retries))):
        try:
            with GraphReadView(coordinator, generation_getter):
                result = compute_fn()
            return result
        except StaleReadError as exc:
            last = exc
            if retry_delay_s > 0:
                time.sleep(float(retry_delay_s))
    assert last is not None
    raise last


def normalize_edges(edges: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges}))

@dataclass(frozen=True, slots=True)
class SpatialCacheDependency:
    """Spatial dependency descriptor for selectively invalidated cache entries.

    `radius=None` denotes a graph-global dependency. A non-negative radius means
    an entry keyed by node id depends on mutations within that many hops.
    `changes` still controls which mutation kinds are relevant.
    """
    changes: ChangeKind = ChangeKind.ALL
    radius: int | None = None

    def __post_init__(self) -> None:
        if self.radius is not None and int(self.radius) < 0:
            raise ValueError("cache dependency radius must be >= 0 or None")

    @property
    def global_dependency(self) -> bool:
        return self.radius is None

    def affected_by(self, event: GraphCommitEvent) -> bool:
        return bool(self.changes & event.changes)


K = TypeVar("K")
V = TypeVar("V")


class LocalizedGenerationCache(Generic[K, V]):
    """Generation-aware derived cache with local dirty-region invalidation.

    The global authoritative generation always advances on commit, but only
    entries whose dependency region intersects the changed subgraph are evicted.
    Graph-global dependencies are conservatively cleared. A caller supplies a
    neighborhood resolver so topology representation stays outside this module.
    """

    def __init__(
        self,
        *,
        dependency: SpatialCacheDependency,
        neighborhood_resolver: Callable[[Iterable[int], int], Iterable[int]] | None = None,
    ) -> None:
        self.dependency = dependency
        self.neighborhood_resolver = neighborhood_resolver
        self.bound_generation: int | None = None
        self._entries: dict[K, V] = {}
        self._entry_generation: dict[K, int] = {}
        self._lock = threading.RLock()
        self.last_invalidated: frozenset[K] = frozenset()

    def bind(self, generation: int) -> None:
        with self._lock:
            self.bound_generation = int(generation)

    def put(self, key: K, value: V, *, generation: int) -> None:
        with self._lock:
            self._entries[key] = value
            self._entry_generation[key] = int(generation)
            self.bound_generation = int(generation)

    def get(self, key: K, *, generation: int) -> V:
        with self._lock:
            if self.bound_generation != int(generation):
                raise RuntimeError(
                    f"cache authority generation mismatch: bound={self.bound_generation}, current={int(generation)}"
                )
            if key not in self._entries:
                raise KeyError(key)
            # Entries not invalidated by a later local commit are still valid even
            # though they were computed at an older generation; authority is carried
            # by the cache-level bound_generation and commit processing below.
            return self._entries[key]

    def keys(self) -> tuple[K, ...]:
        with self._lock:
            return tuple(self._entries.keys())

    def clear(self, *, generation: int) -> None:
        with self._lock:
            old = frozenset(self._entries)
            self._entries.clear(); self._entry_generation.clear()
            self.bound_generation = int(generation)
            self.last_invalidated = old

    def on_graph_commit(self, event: GraphCommitEvent) -> None:
        with self._lock:
            if not self.dependency.affected_by(event):
                self.bound_generation = int(event.generation)
                self.last_invalidated = frozenset()
                return
            if self.dependency.global_dependency:
                old = frozenset(self._entries)
                self._entries.clear(); self._entry_generation.clear()
                self.bound_generation = int(event.generation)
                self.last_invalidated = old
                return

            radius = int(self.dependency.radius or 0)
            changed: set[int] = set(int(n) for n in event.changed_nodes)
            if not changed:
                # Missing locality metadata means we cannot prove a local entry is
                # unaffected. Fail closed rather than retain potentially stale data.
                old = frozenset(self._entries)
                self._entries.clear(); self._entry_generation.clear()
                self.bound_generation = int(event.generation)
                self.last_invalidated = old
                return
            if radius > 0:
                if self.neighborhood_resolver is None:
                    old = frozenset(self._entries)
                    self._entries.clear(); self._entry_generation.clear()
                    self.bound_generation = int(event.generation)
                    self.last_invalidated = old
                    return
                changed.update(int(n) for n in self.neighborhood_resolver(changed, radius))

            invalid = {k for k in self._entries if isinstance(k, int) and int(k) in changed}
            for key in invalid:
                self._entries.pop(key, None); self._entry_generation.pop(key, None)
            self.bound_generation = int(event.generation)
            self.last_invalidated = frozenset(invalid)
