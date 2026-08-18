"""Authoritative state ownership (v5.11 Phase 1).

The AuthoritativeState is the single object that owns all mutable runtime
state. It is never returned directly to callers. All mutations go through
the CommitChannel, which requires an _AuthorityCapability token.

The state bundle contains:
  - graph (GraphBuffers)
  - fibers (FixedWidthFiberLatent)
  - gauges (GaugeConnections | None)
  - calibration (CalibrationState)
  - model_ref (ModelReference)
  - version (int)
  - state_hash (str)

This object is the authoritative S_t in the runtime invariant:

    S_{t+1} = Commit(S_t, T_t, A_t)

No other path may alter S_t.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor

from ...types import GraphBuffers
from ...fibers import FixedWidthFiberLatent


@dataclass
class ModelReference:
    """Reference to a model version in the registry."""
    model_id: str = ""
    version: int = 0
    checkpoint_hash: str = ""

    def to_log(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": int(self.version),
            "checkpoint_hash": self.checkpoint_hash,
        }


@dataclass
class CalibrationState:
    """Calibration state for prediction vs realized outcome."""
    n_samples: int = 0
    mean_predicted: float = 0.0
    mean_realized: float = 0.0
    bias: float = 0.0
    scale: float = 1.0

    def to_log(self) -> dict[str, Any]:
        return {
            "n_samples": int(self.n_samples),
            "mean_predicted": float(self.mean_predicted),
            "mean_realized": float(self.mean_realized),
            "bias": float(self.bias),
            "scale": float(self.scale),
        }

    def state_hash(self) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(str(self.n_samples).encode())
        h.update(str(self.mean_predicted).encode())
        h.update(str(self.mean_realized).encode())
        h.update(str(self.bias).encode())
        h.update(str(self.scale).encode())
        return h.hexdigest()


@dataclass
class AuthoritativeState:
    """The complete authoritative state of the runtime.

    This object is never returned directly to callers. It is owned by
    the _Authority layer and mutated only through CommitChannel.

    The state_hash is computed from all components:
      graph hash + fiber hash + gauge hash + calibration hash + model hash
    """
    graph: GraphBuffers
    fibers: FixedWidthFiberLatent
    gauges: Any = None  # GaugeConnections | None
    calibration: CalibrationState = field(default_factory=CalibrationState)
    model_ref: ModelReference = field(default_factory=ModelReference)
    version: int = 0

    @property
    def state_hash(self) -> str:
        """Deterministic hash of the complete state."""
        import hashlib
        h = hashlib.sha256()
        h.update(self.graph.state_hash().encode())
        h.update(self.fibers.state_hash().encode())
        if self.gauges is not None:
            gh = self.gauges.state_hash()
        else:
            gh = "none"
        h.update(gh.encode())
        h.update(self.calibration.state_hash().encode())
        h.update(self.model_ref.checkpoint_hash.encode())
        h.update(str(self.version).encode())
        return h.hexdigest()

    def to_log(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "state_hash": self.state_hash,
            "graph_hash": self.graph.state_hash(),
            "fiber_hash": self.fibers.state_hash(),
            "gauge_hash": self.gauges.state_hash() if self.gauges is not None else "none",
            "calibration": self.calibration.to_log(),
            "model_ref": self.model_ref.to_log(),
        }
