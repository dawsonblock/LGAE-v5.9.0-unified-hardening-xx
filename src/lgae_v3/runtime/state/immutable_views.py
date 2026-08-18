"""Immutable views for runtime public API (v5.11 Phase 1).

The runtime exposes read-only views of its internal state through
these facades. The actual engine is private (_engine) and cannot
be accessed directly by external code.

The EngineFacade provides:
- graph: FrozenGraphView (read-only)
- fibers: FrozenFiberView (read-only)
- gauges: FrozenGaugeView (read-only)
- audit(): read-only audit
- state_hash(): deterministic hash
- version: int

All mutation methods on the facade raise UnauthorizedMutationError.
"""
from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

from ...types import GraphBuffers
from ..authority import UnauthorizedMutationError
from .frozen_views import FrozenGraphView, FrozenFiberView, FrozenGaugeView


class EngineFacade:
    """Read-only facade over the internal LGAEEngine.

    This is what runtime.engine returns. It exposes read methods
    but blocks all mutation. The actual engine is private.

    v5.11-RC Phase 1: The raw engine is stored via object.__setattr__
    and accessed via object.__getattribute__ internally. External
    attribute access to '_engine' is blocked by __getattribute__.
    This closes the facade._engine escape hatch.
    """

    __slots__ = ("_engine",)

    def __init__(self, engine: Any) -> None:
        object.__setattr__(self, "_engine", engine)

    def __getattribute__(self, name: str) -> Any:
        # Block external access to the raw engine reference.
        # Internal methods use object.__getattribute__ to bypass this.
        if name == "_engine":
            raise UnauthorizedMutationError(
                "access to raw engine via _engine is blocked; "
                "authoritative state is accessed only through the commit channel"
            )
        return object.__getattribute__(self, name)

    @property
    def graph(self) -> FrozenGraphView:
        """Frozen graph view — mutations raise UnauthorizedMutationError."""
        return FrozenGraphView(object.__getattribute__(self, "_engine").graph)

    @property
    def fibers(self) -> FrozenFiberView:
        """Frozen fiber view — mutations raise UnauthorizedMutationError."""
        engine = object.__getattribute__(self, "_engine")
        return FrozenFiberView(engine.fibers)

    @property
    def fiber_state(self) -> Any:
        """Frozen fiber state (alias for compatibility)."""
        return self.fibers

    @property
    def gauge_connections(self) -> FrozenGaugeView | None:
        """Frozen gauge view — mutations raise UnauthorizedMutationError."""
        engine = object.__getattribute__(self, "_engine")
        gauges = engine.gauge_connections
        if gauges is None:
            return None
        return FrozenGaugeView(gauges)

    @property
    def step_index(self) -> int:
        return int(object.__getattribute__(self, "_engine").step_index)

    @property
    def num_nodes(self) -> int:
        return int(object.__getattribute__(self, "_engine").graph.num_nodes)

    @property
    def config(self) -> Any:
        return object.__getattribute__(self, "_engine").cfg

    @property
    def governor(self) -> Any:
        """Read-only access to the governor (governor itself is read-only for evaluation)."""
        return object.__getattribute__(self, "_engine").governor

    def audit(self) -> Any:
        """Read-only audit of current state."""
        return object.__getattribute__(self, "_engine").audit()

    def authority_hash(self) -> str:
        """Deterministic hash of the authoritative state."""
        return object.__getattribute__(self, "_engine").authority_hash()

    def state_hash(self) -> str:
        """Deterministic hash of the authoritative state."""
        return object.__getattribute__(self, "_engine").authority_hash()

    def fibers_raw(self) -> Tensor:
        """Detached clone of the fiber latent (for read-only computation)."""
        return object.__getattribute__(self, "_engine").fibers().detach().clone()

    def __setattr__(self, name: str, value: Any) -> None:
        raise UnauthorizedMutationError(
            f"cannot set attribute '{name}' on EngineFacade; "
            "authoritative state is mutated only through the commit channel"
        )

    def __delattr__(self, name: str) -> None:
        raise UnauthorizedMutationError(
            f"cannot delete attribute '{name}' on EngineFacade"
        )

    def evaluate_and_maybe_commit(self, *args: Any, **kwargs: Any) -> Any:
        raise UnauthorizedMutationError(
            "evaluate_and_maybe_commit is blocked on EngineFacade; "
            "use the canonical runtime step() path instead"
        )

    def evaluate_fiber_action(self, *args: Any, **kwargs: Any) -> Any:
        raise UnauthorizedMutationError(
            "evaluate_fiber_action is blocked on EngineFacade; "
            "use the canonical runtime step() path instead"
        )

    def evaluate_gauge_action(self, *args: Any, **kwargs: Any) -> Any:
        raise UnauthorizedMutationError(
            "evaluate_gauge_action is blocked on EngineFacade; "
            "use the canonical runtime step() path instead"
        )

    def resolve_quarantine(self, *args: Any, **kwargs: Any) -> Any:
        raise UnauthorizedMutationError(
            "resolve_quarantine is blocked on EngineFacade; "
            "use the canonical runtime step() path instead"
        )
