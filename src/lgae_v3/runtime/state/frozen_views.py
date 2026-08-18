"""Frozen (immutable) views of authoritative state (v5.11 Phase 3).

The v5.10 defect: AuthoritativeStateGuard.graph returns raw mutable
GraphBuffers. Callers can do guard.graph.weight[...] = ... to mutate
authoritative state, bypassing the authority model.

The fix: frozen views that return detached clones of tensors. Any attempt
to mutate through the view raises UnauthorizedMutationError.

For large graphs where copying is expensive, copy-on-write can be added
later. For now, defensive cloning is the safe default.
"""
from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

from ...types import GraphBuffers
from ..authority import UnauthorizedMutationError


class FrozenGraphView:
    """Immutable view of a GraphBuffers.

    All tensor properties return detached clones. Any attempt to set
    attributes raises UnauthorizedMutationError.
    """

    __slots__ = ("_graph", "_cache")

    def __init__(self, graph: GraphBuffers) -> None:
        # Store a reference but never expose it directly.
        object.__setattr__(self, "_graph", graph)
        object.__setattr__(self, "_cache", {})

    def _clone_tensor(self, name: str) -> Tensor:
        cache = object.__getattribute__(self, "_cache")
        if name not in cache:
            graph = object.__getattribute__(self, "_graph")
            t = getattr(graph, name)
            cache[name] = t.detach().clone() if t is not None else None
        return cache[name]

    @property
    def src(self) -> Tensor:
        return self._clone_tensor("src")

    @property
    def dst(self) -> Tensor:
        return self._clone_tensor("dst")

    @property
    def weight(self) -> Tensor:
        return self._clone_tensor("weight")

    @property
    def length(self) -> Tensor:
        return self._clone_tensor("length")

    @property
    def valid(self) -> Tensor:
        return self._clone_tensor("valid")

    @property
    def role(self) -> Tensor:
        return self._clone_tensor("role")

    @property
    def slot_gen(self) -> Tensor:
        return self._clone_tensor("slot_gen")

    @property
    def num_nodes(self) -> int:
        return int(object.__getattribute__(self, "_graph").num_nodes)

    @property
    def capacity(self) -> int:
        return int(object.__getattribute__(self, "_graph").capacity)

    @property
    def version(self) -> int:
        return int(object.__getattribute__(self, "_graph").version)

    def state_hash(self) -> str:
        return object.__getattribute__(self, "_graph").state_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "capacity": self.capacity,
            "version": self.version,
            "state_hash": self.state_hash(),
        }

    def clone(self) -> GraphBuffers:
        """Return a detached clone of the underlying graph as a real GraphBuffers.

        This is safe because the clone is a new object — mutating it does not
        affect authoritative state.
        """
        import dataclasses
        graph = object.__getattribute__(self, "_graph")
        new = dataclasses.replace(graph)
        for f in dataclasses.fields(graph):
            val = getattr(graph, f.name)
            if isinstance(val, Tensor):
                setattr(new, f.name, val.detach().clone())
        return new

    def validate(self) -> None:
        """Validate the underlying graph. Delegates to the real graph."""
        object.__getattribute__(self, "_graph").validate()

    def active(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return (src, dst, weight) for active edges as detached clones."""
        graph = object.__getattribute__(self, "_graph")
        s, d, w = graph.active()
        return s.detach().clone(), d.detach().clone(), w.detach().clone()

    def active_length(self) -> Tensor:
        """Return active edge lengths as detached clones."""
        return object.__getattribute__(self, "_graph").active_length().detach().clone()

    def active_roles(self) -> Tensor:
        """Return active edge roles as detached clones."""
        return object.__getattribute__(self, "_graph").active_roles().detach().clone()

    def __setattr__(self, name: str, value: Any) -> None:
        raise UnauthorizedMutationError(
            f"cannot set attribute '{name}' on FrozenGraphView; "
            "authoritative state is mutated only through the commit channel"
        )

    def __delattr__(self, name: str) -> None:
        raise UnauthorizedMutationError(
            f"cannot delete attribute '{name}' on FrozenGraphView; "
            "authoritative state is mutated only through the commit channel"
        )


class FrozenFiberView:
    """Immutable view of fiber state.

    Returns detached clones of fiber tensors.
    """

    __slots__ = ("_fiber_fn", "_cache")

    def __init__(self, fiber_fn: Callable[[], Tensor]) -> None:
        object.__setattr__(self, "_fiber_fn", fiber_fn)
        object.__setattr__(self, "_cache", {})

    def _get_z(self) -> Tensor:
        cache = object.__getattribute__(self, "_cache")
        if "z" not in cache:
            fn = object.__getattribute__(self, "_fiber_fn")
            z = fn()
            cache["z"] = z.detach().clone() if z is not None else None
        return cache["z"]

    @property
    def z(self) -> Tensor:
        return self._get_z()

    @property
    def latent(self) -> Tensor:
        """Detached clone of the fiber latent tensor."""
        fn = object.__getattribute__(self, "_fiber_fn")
        if hasattr(fn, "latent"):
            return fn.latent.detach().clone()
        return self._get_z()

    @property
    def capacity(self) -> Tensor:
        """Detached clone of the fiber capacity tensor."""
        fn = object.__getattribute__(self, "_fiber_fn")
        if hasattr(fn, "capacity"):
            return fn.capacity.detach().clone()
        raise AttributeError("FrozenFiberView has no 'capacity'")

    @property
    def active_mask(self) -> Tensor:
        """Detached clone of the active mask tensor."""
        fn = object.__getattribute__(self, "_fiber_fn")
        if hasattr(fn, "active_mask"):
            return fn.active_mask.detach().clone()
        raise AttributeError("FrozenFiberView has no 'active_mask'")

    @property
    def dim(self) -> int:
        """Fiber dimension."""
        fn = object.__getattribute__(self, "_fiber_fn")
        if hasattr(fn, "dim"):
            return int(fn.dim)
        raise AttributeError("FrozenFiberView has no 'dim'")

    def effective_mask(self) -> Tensor:
        """Detached clone of the effective mask."""
        fn = object.__getattribute__(self, "_fiber_fn")
        if hasattr(fn, "effective_mask"):
            return fn.effective_mask().detach().clone()
        raise AttributeError("FrozenFiberView has no 'effective_mask'")

    def snapshot(self) -> Any:
        """Return a fiber snapshot for shadow evaluation.

        Snapshots are immutable copies of fiber state. They are safe to
        return because they are used for restore() which creates a new state.
        """
        fn = object.__getattribute__(self, "_fiber_fn")
        if hasattr(fn, "snapshot"):
            return fn.snapshot()
        raise AttributeError("FrozenFiberView has no 'snapshot'")

    def state_hash(self) -> str:
        """Deterministic hash of the fiber state."""
        fn = object.__getattribute__(self, "_fiber_fn")
        # fn is the fiber bank (FixedWidthFiberLatent), which is callable
        # (returns z) and also has state_hash().
        if hasattr(fn, "state_hash"):
            return fn.state_hash()
        # Fallback: hash the tensor deterministically.
        fibers = fn() if callable(fn) else fn
        from .state_hashing import state_hash
        if fibers is not None:
            return state_hash(fibers)
        return "none"

    def __call__(self) -> Tensor:
        """Return a detached clone of the fiber latent z.

        This makes FrozenFiberView callable like the original fiber bank,
        but returns a safe clone that cannot mutate authoritative state.
        """
        return self._get_z()

    def __setattr__(self, name: str, value: Any) -> None:
        raise UnauthorizedMutationError(
            f"cannot set attribute '{name}' on FrozenFiberView; "
            "authoritative state is mutated only through the commit channel"
        )

    def __delattr__(self, name: str) -> None:
        raise UnauthorizedMutationError(
            f"cannot delete attribute '{name}' on FrozenFiberView"
        )


class FrozenGaugeView:
    """Immutable view of gauge connections.

    Returns detached clones of gauge tensors.
    """

    __slots__ = ("_gauge", "_cache")

    def __init__(self, gauge: Any) -> None:
        object.__setattr__(self, "_gauge", gauge)
        object.__setattr__(self, "_cache", {})

    @property
    def state_hash(self) -> str:
        g = object.__getattribute__(self, "_gauge")
        return g.state_hash() if g is not None and hasattr(g, "state_hash") else ""

    @property
    def raw_generators(self) -> Tensor:
        """Detached clone of the gauge raw generators."""
        g = object.__getattribute__(self, "_gauge")
        if g is not None and hasattr(g, "raw_generators"):
            return g.raw_generators.detach().clone()
        raise AttributeError("raw_generators not available on frozen gauge view")

    def __setattr__(self, name: str, value: Any) -> None:
        raise UnauthorizedMutationError(
            f"cannot set attribute '{name}' on FrozenGaugeView; "
            "authoritative state is mutated only through the commit channel"
        )

    def __delattr__(self, name: str) -> None:
        raise UnauthorizedMutationError(
            f"cannot delete attribute '{name}' on FrozenGaugeView"
        )


class StaleSnapshotError(RuntimeError):
    """Raised when a candidate's source state version doesn't match
    the current authoritative state version. This prevents TOCTOU problems."""
