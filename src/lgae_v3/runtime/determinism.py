"""Deterministic utilities for reproducible runtime behavior (v5.11 Phase 14).

Centralizes all determinism-critical operations:
- derive_seed: SHA-256 based seed derivation (replaces hash())
- canonical_sort: deterministic sorting
- canonical_json: deterministic JSON serialization
- canonical_float: deterministic float representation
- canonical_tensor_hash: deterministic tensor hashing

All decision-affecting code must use these utilities instead of:
- hash()
- uuid4()
- random.random()
- set/dict iteration (use canonical_sort first)
- time.time() (for decision-affecting values)
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor


def derive_seed(master_seed: int, namespace: str, *components: Any) -> int:
    """Derive a deterministic seed from a master seed and components.

    Uses SHA-256 for deterministic derivation. The same inputs always
    produce the same output, regardless of PYTHONHASHSEED.

    Args:
        master_seed: the master RNG seed
        namespace: a namespace string to avoid collisions
        *components: additional components (ints, strings, etc.)

    Returns:
        A deterministic uint64 seed.
    """
    parts = [str(master_seed), namespace]
    for c in components:
        if isinstance(c, (int, float, str, bool)):
            parts.append(str(c))
        else:
            parts.append(canonical_json(c))
    data = "|".join(parts).encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], "big")


def canonical_sort(items: Iterable[Any]) -> list[Any]:
    """Sort items deterministically.

    For non-comparable items, sorts by canonical JSON representation.
    """
    items = list(items)
    try:
        return sorted(items)
    except TypeError:
        return sorted(items, key=lambda x: canonical_json(x))


def canonical_json(obj: Any) -> str:
    """Canonical JSON serialization for deterministic hashing.

    Keys are sorted, floats are repr'd consistently, no whitespace.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=_default_serializer)


def _default_serializer(obj: Any) -> Any:
    if isinstance(obj, Tensor):
        return canonical_tensor_repr(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "value") and isinstance(obj.value, str):
        return obj.value
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)


def canonical_float(x: float) -> str:
    """Canonical float representation for deterministic comparison.

    Uses repr() which is consistent across Python versions for the same
    float value. Handles NaN and Inf consistently.
    """
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return repr(float(x))


def canonical_tensor_hash(tensor: Tensor) -> str:
    """SHA-256 hash of a tensor's canonical representation.

    The hash is deterministic across processes and Python versions.
    """
    return hashlib.sha256(canonical_tensor_repr(tensor).encode("utf-8")).hexdigest()


def canonical_tensor_repr(tensor: Tensor) -> str:
    """Canonical string representation of a tensor for hashing."""
    if tensor is None:
        return "None"
    t = tensor.detach().cpu()
    return f"shape={list(t.shape)}|dtype={t.dtype}|values={t.tolist()}"


def canonical_graph_hash(graph: Any) -> str:
    """Canonical hash of a GraphBuffers for state identity."""
    parts = []
    for name in ("src", "dst", "weight", "length", "valid", "role"):
        t = getattr(graph, name, None)
        if t is not None:
            parts.append(f"{name}={canonical_tensor_repr(t)}")
    parts.append(f"num_nodes={getattr(graph, 'num_nodes', 0)}")
    parts.append(f"version={getattr(graph, 'version', 0)}")
    data = "|".join(parts).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
