"""State hashing utilities (v5.11 Phase 1).

Deterministic hashing for all state components. Never uses Python hash().
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from torch import Tensor

from ...types import GraphBuffers


def canonical_encode(obj: Any) -> bytes:
    """Canonical encoding of an object for hashing.

    Uses JSON with sorted keys for dicts, and deterministic tensor
    serialization for tensors. Never uses Python hash().
    """
    return json.dumps(_to_serializable(obj), sort_keys=True).encode()


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, Tensor):
        return _tensor_to_serializable(obj)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if hasattr(obj, "state_hash") and callable(obj.state_hash):
        return obj.state_hash()
    if hasattr(obj, "to_log") and callable(obj.to_log):
        return _to_serializable(obj.to_log())
    # Fallback: repr is deterministic for most objects.
    return repr(obj)


def _tensor_to_serializable(t: Tensor) -> str:
    """Deterministic tensor serialization."""
    if t.numel() == 0:
        return "empty"
    return f"tensor:{t.dtype}:{tuple(t.shape)}:{t.detach().cpu().contiguous().numpy().tobytes().hex()}"


def state_hash(*components: Any) -> str:
    """Compute a deterministic SHA-256 hash from state components."""
    h = hashlib.sha256()
    for comp in components:
        if comp is None:
            h.update(b"none")
        elif isinstance(comp, str):
            h.update(comp.encode())
        elif isinstance(comp, (int, float)):
            h.update(str(comp).encode())
        elif isinstance(comp, Tensor):
            h.update(_tensor_to_serializable(comp).encode())
        elif hasattr(comp, "state_hash") and callable(comp.state_hash):
            h.update(comp.state_hash().encode())
        else:
            h.update(canonical_encode(comp))
    return h.hexdigest()


def graph_hash(graph: GraphBuffers) -> str:
    """Deterministic hash of a GraphBuffers."""
    return graph.state_hash()


def fiber_hash(fibers: Any) -> str:
    """Deterministic hash of fiber state."""
    if fibers is None:
        return "none"
    if hasattr(fibers, "state_hash") and callable(fibers.state_hash):
        return fibers.state_hash()
    raise TypeError(
        "Fiber state must implement deterministic state_hash(); "
        "Python hash() is not allowed in deterministic runtime paths"
    )


def gauge_hash(gauges: Any) -> str:
    """Deterministic hash of gauge state."""
    if gauges is None:
        return "none"
    if hasattr(gauges, "state_hash") and callable(gauges.state_hash):
        return gauges.state_hash()
    if hasattr(gauges, "raw_generators"):
        return state_hash(gauges.raw_generators)
    raise TypeError(
        "Gauge state must implement deterministic state_hash() or have raw_generators; "
        "Python hash() is not allowed in deterministic runtime paths"
    )
