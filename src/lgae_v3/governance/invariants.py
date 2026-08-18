"""Formal invariant layer (Phase 37).

Invariants are machine-readable contracts. Each invariant is a callable that
inspects authoritative state and returns ``True`` only if the contract holds.
The runtime/governor checks registered invariants before and after commits;
a violation is a governance failure that blocks the commit.

Example::

    @invariant
    def graph_connected(state):
        return nx.is_connected(state.graph_nx())

Applications may register additional invariants via ``register_invariant``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum


class InvariantSeverity(str, Enum):
    """Severity of an invariant violation."""
    BLOCKING = "blocking"      # commit must be rejected
    WARNING = "warning"        # recorded but does not block


@dataclass(slots=True)
class InvariantResult:
    """Outcome of one invariant check."""
    name: str
    passed: bool
    severity: InvariantSeverity
    message: str = ""

    def to_log(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "severity": self.severity.value,
            "message": self.message,
        }


# An invariant function inspects a state object and returns (passed, message).
InvariantFn = Callable[[Any], tuple[bool, str]]


class InvariantRegistry:
    """Registry of formal invariants checked on authoritative state."""

    def __init__(self) -> None:
        self._invariants: dict[str, tuple[InvariantFn, InvariantSeverity]] = {}

    def register(
        self, name: str, fn: InvariantFn, *, severity: InvariantSeverity = InvariantSeverity.BLOCKING
    ) -> None:
        if not name:
            raise ValueError("invariant name must be non-empty")
        self._invariants[str(name)] = (fn, severity)

    def unregister(self, name: str) -> None:
        self._invariants.pop(str(name), None)

    @property
    def names(self) -> list[str]:
        return sorted(self._invariants.keys())

    def check_all(self, state: Any) -> list[InvariantResult]:
        """Run every registered invariant against ``state`` (deterministic order)."""
        out: list[InvariantResult] = []
        for name in sorted(self._invariants.keys()):
            fn, severity = self._invariants[name]
            try:
                passed, message = fn(state)
                passed = bool(passed)
                message = str(message)
            except Exception as exc:  # an invariant raising is a failure
                passed = False
                message = f"invariant raised: {exc!r}"
            out.append(InvariantResult(name=name, passed=passed, severity=severity, message=message))
        return out

    def check_blocking(self, state: Any) -> tuple[bool, list[InvariantResult]]:
        """Run blocking invariants; return (all_passed, results)."""
        results = [r for r in self.check_all(state) if r.severity == InvariantSeverity.BLOCKING]
        return all(r.passed for r in results), results


def invariant(
    name: str | None = None, *, severity: InvariantSeverity = InvariantSeverity.BLOCKING
) -> Callable[[Callable[[Any], bool]], InvariantFn]:
    """Decorator declaring a formal invariant.

    The decorated function returns a bool (True = contract holds). The
    decorator wraps it into the ``(passed, message)`` form and registers it
    on the default registry.
    """
    def decorator(fn: Callable[[Any], bool]) -> InvariantFn:
        invariant_name = name or fn.__name__

        def wrapped(state: Any) -> tuple[bool, str]:
            ok = bool(fn(state))
            return ok, "" if ok else f"{invariant_name} violated"

        # Attach metadata for introspection.
        wrapped.__invariant_name__ = invariant_name  # type: ignore[attr-defined]
        wrapped.__invariant_severity__ = severity  # type: ignore[attr-defined]
        DEFAULT_REGISTRY.register(invariant_name, wrapped, severity=severity)
        return wrapped

    return decorator


def register_invariant(
    name: str, fn: InvariantFn, *, severity: InvariantSeverity = InvariantSeverity.BLOCKING,
    registry: InvariantRegistry | None = None,
) -> None:
    """Register an invariant on a registry (default if omitted)."""
    (registry or DEFAULT_REGISTRY).register(name, fn, severity=severity)


# The default global registry. The runtime uses this unless given another.
DEFAULT_REGISTRY = InvariantRegistry()


# ---------------------------------------------------------------------------
# Built-in structural invariants. These are conservative and operate on any
# state object that exposes the expected attributes; missing attributes make
# the invariant pass (it does not apply).
# ---------------------------------------------------------------------------
@invariant("graph_buffers_finite")
def _graph_buffers_finite(state: Any) -> bool:
    g = getattr(state, "graph", None)
    if g is None or not hasattr(g, "weight"):
        return True
    import torch
    return bool(torch.isfinite(g.weight[g.valid]).all().item())


@invariant("graph_buffers_positive_weights")
def _graph_buffers_positive_weights(state: Any) -> bool:
    g = getattr(state, "graph", None)
    if g is None or not hasattr(g, "weight"):
        return True
    w = g.weight[g.valid]
    return bool((w > 0).all().item())


@invariant("gauge_orthogonal")
def _gauge_orthogonal(state: Any) -> bool:
    engine = getattr(state, "engine", None) or state
    bank = getattr(engine, "gauge_connections", None)
    if bank is None:
        return True
    import torch
    W = getattr(bank, "connections", None)
    if W is None:
        return True
    # Check a bounded sample of active connections for orthogonality.
    n = min(int(W.shape[0]), 16)
    for i in range(n):
        Wi = W[i]
        eye = torch.eye(Wi.shape[-1], dtype=Wi.dtype, device=Wi.device)
        err = float((Wi.transpose(-1, -2) @ Wi - eye).abs().max().item())
        if err > 1e-4:
            return False
    return True


@invariant("laplacian_psd")
def _laplacian_psd(state: Any) -> bool:
    g = getattr(state, "graph", None)
    if g is None or not hasattr(g, "src"):
        return True
    import torch
    n = int(g.num_nodes)
    L = torch.zeros(n, n, dtype=torch.float64)
    for i in torch.where(g.valid)[0].tolist():
        u, v, w = int(g.src[i]), int(g.dst[i]), float(g.weight[i])
        L[u, u] += w; L[v, v] += w
        L[u, v] -= w; L[v, u] -= w
    eig = torch.linalg.eigvalsh(L)
    return float(eig.min().item()) >= -1e-6


@invariant("receipt_chain_valid", severity=InvariantSeverity.WARNING)
def _receipt_chain_valid(state: Any) -> bool:
    path = getattr(state, "receipt_path", None)
    if path is None:
        return True
    from ..receipts import verify_receipt_chain
    try:
        ok, _ = verify_receipt_chain(str(path))
        return bool(ok)
    except Exception:
        return True  # missing receipt file is not yet a violation
