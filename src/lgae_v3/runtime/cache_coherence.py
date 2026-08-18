"""Mandatory cache coherence for the canonical runtime (Phase 4).

Every committed mutation describes which state dimensions changed via a
``MutationImpact``. Caches declare dependencies with the ``@depends_on``
decorator. The ``CacheRegistry`` listens to the commit event bus and
invalidates only caches whose declared dependencies intersect the mutation
impact. Unaffected caches survive local mutations.

This builds on the v5.8 ``ChangeKind`` / ``CommitEventBus`` /
``GenerationStampedCache`` infrastructure rather than replacing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..cache_coherence import (
    ChangeKind,
    CommitEventBus,
    CommitConsumer,
    GraphCommitEvent,
    GenerationStampedCache,
)


@dataclass(frozen=True, slots=True)
class MutationImpact:
    """Structured description of which authoritative state dimensions changed.

    Maps 1:1 to ``ChangeKind`` flags but exposes named boolean fields so policy
    code and receipts can describe impacts readably.
    """
    topology: bool = False
    weights: bool = False
    metric: bool = False      # edge lengths / metric
    gauges: bool = False
    fibers: bool = False
    latents: bool = False
    roles: bool = False

    def to_change_kind(self) -> ChangeKind:
        ck = ChangeKind.NONE
        if self.topology: ck |= ChangeKind.TOPOLOGY
        if self.weights: ck |= ChangeKind.WEIGHTS
        if self.metric: ck |= ChangeKind.LENGTHS
        if self.gauges: ck |= ChangeKind.GAUGE
        if self.fibers: ck |= ChangeKind.FIBERS
        if self.latents: ck |= ChangeKind.LATENTS
        if self.roles: ck |= ChangeKind.ROLES
        return ck

    @classmethod
    def from_change_kind(cls, ck: ChangeKind) -> "MutationImpact":
        return cls(
            topology=bool(ck & ChangeKind.TOPOLOGY),
            weights=bool(ck & ChangeKind.WEIGHTS),
            metric=bool(ck & ChangeKind.LENGTHS),
            gauges=bool(ck & ChangeKind.GAUGE),
            fibers=bool(ck & ChangeKind.FIBERS),
            latents=bool(ck & ChangeKind.LATENTS),
            roles=bool(ck & ChangeKind.ROLES),
        )

    @property
    def is_empty(self) -> bool:
        return self.to_change_kind() == ChangeKind.NONE

    def to_log(self) -> dict[str, bool]:
        return {
            "topology": self.topology, "weights": self.weights, "metric": self.metric,
            "gauges": self.gauges, "fibers": self.fibers, "latents": self.latents,
            "roles": self.roles,
        }


# Dimension name -> ChangeKind flag mapping for the @depends_on decorator.
_DIM_TO_FLAG: dict[str, ChangeKind] = {
    "topology": ChangeKind.TOPOLOGY,
    "weights": ChangeKind.WEIGHTS,
    "metric": ChangeKind.LENGTHS,
    "lengths": ChangeKind.LENGTHS,
    "gauges": ChangeKind.GAUGE,
    "gauge": ChangeKind.GAUGE,
    "fibers": ChangeKind.FIBERS,
    "fiber": ChangeKind.FIBERS,
    "latents": ChangeKind.LATENTS,
    "latent": ChangeKind.LATENTS,
    "roles": ChangeKind.ROLES,
    "role": ChangeKind.ROLES,
}


def depends_on(*dims: str) -> Callable[[type], type]:
    """Class decorator declaring a cache's state-dimension dependencies.

    Example::

        @depends_on("topology", "weights")
        class CurvatureCache(GenerationStampedCache):
            ...
    """
    flags = ChangeKind.NONE
    for d in dims:
        key = str(d).lower()
        if key not in _DIM_TO_FLAG:
            raise ValueError(f"unknown cache dependency dimension: {d!r}")
        flags |= _DIM_TO_FLAG[key]
    if flags == ChangeKind.NONE:
        raise ValueError("depends_on requires at least one dimension")

    def decorator(cls: type) -> type:
        cls.__cache_dependencies__ = flags  # type: ignore[attr-defined]
        # If the class subclasses GenerationStampedCache, wire its dependency
        # mask automatically so callers do not need to pass it manually.
        if issubclass(cls, GenerationStampedCache):
            orig_init = cls.__init__

            def __init__(self, *args, **kwargs):  # type: ignore[no-redef]
                if "dependencies" not in kwargs:
                    kwargs["dependencies"] = flags
                orig_init(self, *args, **kwargs)
            cls.__init__ = __init__  # type: ignore[assignment]
        return cls

    return decorator


def declared_dependencies(obj: Any) -> ChangeKind:
    """Return the ChangeKind dependency mask declared for a cache class/instance."""
    flags = getattr(obj, "__cache_dependencies__", None)
    if flags is not None:
        return ChangeKind(flags)
    if isinstance(obj, GenerationStampedCache):
        return ChangeKind(obj.dependencies.changes)
    return ChangeKind.ALL


class CacheRegistry(CommitConsumer):
    """Registry of derived caches with selective invalidation.

    Listens to the commit event bus. On each commit, only caches whose
    declared dependencies intersect the commit's change mask are invalidated.
    Unaffected caches retain their bound generation and remain valid.
    """

    def __init__(self, event_bus: CommitEventBus) -> None:
        self.event_bus = event_bus
        self._caches: list[Any] = []
        self._invalidations: list[dict[str, Any]] = []
        event_bus.register(self)

    def register(self, cache: Any, *, name: str | None = None) -> None:
        if name is not None:
            setattr(cache, "__cache_name__", name)
        if cache not in self._caches:
            self._caches.append(cache)

    def unregister(self, cache: Any) -> None:
        if cache in self._caches:
            self._caches.remove(cache)

    @property
    def caches(self) -> list[Any]:
        return list(self._caches)

    @property
    def invalidations(self) -> list[dict[str, Any]]:
        return list(self._invalidations)

    def on_graph_commit(self, event: GraphCommitEvent) -> None:
        invalidated: list[str] = []
        spared: list[str] = []
        for cache in list(self._caches):
            deps = declared_dependencies(cache)
            if bool(deps & event.changes):
                name = getattr(cache, "__cache_name__", type(cache).__name__)
                if hasattr(cache, "invalidate"):
                    cache.invalidate(int(event.generation), reason=str(event.reason))
                elif hasattr(cache, "mark_dirty"):
                    cache.mark_dirty(graph_version=int(event.generation), reason=str(event.reason))
                invalidated.append(name)
            else:
                spared.append(getattr(cache, "__cache_name__", type(cache).__name__))
        self._invalidations.append({
            "generation": int(event.generation),
            "changes": int(event.changes),
            "invalidated": invalidated,
            "spared": spared,
            "reason": str(event.reason),
        })

    def summary(self) -> dict[str, Any]:
        return {
            "registered": len(self._caches),
            "commit_count": len(self._invalidations),
            "last_invalidations": self._invalidations[-1] if self._invalidations else None,
        }
