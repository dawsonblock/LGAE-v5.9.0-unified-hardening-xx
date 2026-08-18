"""Deterministic RNG context with domain-separated substreams (v5.3.3).

This module addresses the nondeterminism identified in Milestone 1 of the
v5.4.0 roadmap.  Previously, modules independently called global random
generators (``torch.manual_seed``, ``np.random.seed``, ``random.seed``),
which meant that adding one new random call could change every subsequent
result.

The ``DeterministicRNGContext`` provides domain-separated substreams:

    master_seed
       │
       ├── graph_generation
       ├── target_sampling
       ├── action_sampling
       ├── model_initialization
       ├── counterfactuals
       └── qualification

Each substream derives its seed via::

    s_i = SHA256(master_seed ∥ namespace)

This prevents one new random call from changing every subsequent result.
"""
from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch


def derive_seed(master_seed: int, namespace: str) -> int:
    """Derive a deterministic substream seed.

    Args:
        master_seed: The master seed.
        namespace: A domain-specific namespace string (e.g. "graph_generation").

    Returns:
        A deterministic 64-bit seed derived from the master seed and namespace.
        This is NOT dependent on PYTHONHASHSEED.
    """
    payload = f"{master_seed}:{namespace}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


# Standard namespace constants
NAMESPACES = (
    "graph_generation",
    "target_sampling",
    "action_sampling",
    "model_initialization",
    "counterfactuals",
    "qualification",
    "benchmark",
    "checkpoint",
)


@dataclass
class DeterministicRNGContext:
    """Owns all RNG state for a deterministic run.

    Provides domain-separated substreams for Python ``random``, NumPy, and
    PyTorch (CPU and CUDA).  Each substream is independently seeded from
    the master seed via SHA-256, so adding a new substream does not change
    existing substreams.

    Usage::

        ctx = DeterministicRNGContext(master_seed=42)
        g = ctx.graph_generation()       # numpy Generator
        torch_gen = ctx.torch_gen("model_initialization")
    """

    master_seed: int
    _generators: dict[str, np.random.Generator] = field(default_factory=dict)

    def substream_seed(self, namespace: str) -> int:
        """Get the deterministic seed for a namespace."""
        return derive_seed(self.master_seed, namespace)

    def numpy_gen(self, namespace: str) -> np.random.Generator:
        """Get a NumPy Generator for a namespace (cached)."""
        if namespace not in self._generators:
            self._generators[namespace] = np.random.default_rng(
                self.substream_seed(namespace)
            )
        return self._generators[namespace]

    def python_rng(self, namespace: str) -> random.Random:
        """Get a Python random.Random for a namespace."""
        return random.Random(self.substream_seed(namespace))

    def torch_gen(self, namespace: str, device: str = "cpu") -> torch.Generator:
        """Get a torch.Generator for a namespace."""
        gen = torch.Generator(device=device)
        gen.manual_seed(self.substream_seed(namespace))
        return gen

    def seed_torch(self, namespace: str, device: str = "cpu") -> None:
        """Seed the global torch RNG for a namespace.

        Use this when code calls ``torch.randn`` etc. without an explicit
        generator.  Prefer ``torch_gen`` for new code.
        """
        torch.manual_seed(self.substream_seed(namespace))

    # Convenience methods for standard namespaces

    def graph_generation(self) -> np.random.Generator:
        return self.numpy_gen("graph_generation")

    def target_sampling(self) -> np.random.Generator:
        return self.numpy_gen("target_sampling")

    def action_sampling(self) -> np.random.Generator:
        return self.numpy_gen("action_sampling")

    def model_initialization(self) -> torch.Generator:
        return self.torch_gen("model_initialization")

    def counterfactuals(self) -> np.random.Generator:
        return self.numpy_gen("counterfactuals")

    def qualification(self) -> np.random.Generator:
        return self.numpy_gen("qualification")


@contextmanager
def deterministic_mode(
    master_seed: int = 42,
    *,
    torch_deterministic: bool = True,
    warn_only: bool = False,
):
    """Context manager for fully deterministic execution.

    Sets:
    - PYTHONHASHSEED (via env, must be set before Python starts for full effect)
    - torch.use_deterministic_algorithms
    - torch.manual_seed
    - np.random.seed
    - random.seed

    Args:
        master_seed: The master seed for all RNG.
        torch_deterministic: Whether to enable torch deterministic algorithms.
        warn_only: If True, torch deterministic warnings instead of errors.
    """
    ctx = DeterministicRNGContext(master_seed=master_seed)

    # Seed all global RNGs with the qualification substream
    qseed = ctx.substream_seed("qualification")
    torch.manual_seed(qseed)
    np.random.seed(qseed & 0xFFFFFFFF)
    random.seed(qseed)

    prev_det = torch.are_deterministic_algorithms_enabled()

    # torch.are_deterministic_algorithms_warn_only_enabled() may not
    # exist in older torch versions; handle gracefully.
    try:
        prev_warn = torch.are_deterministic_algorithms_warn_only_enabled()
    except AttributeError:
        prev_warn = False

    if torch_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

    try:
        yield ctx
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)
