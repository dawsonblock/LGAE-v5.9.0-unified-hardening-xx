"""Typed configuration for the canonical v5.10 runtime.

This is a thin orchestration-level config. It does not duplicate the
subsystem-level ``LGAEConfig``; it composes around it. Subsystem behavior
remains governed by ``LGAEConfig`` and its ``ProductionConfig`` /
``ResearchConfig`` presets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from ..types import GraphBuffers


class RuntimeMode(str, Enum):
    """Research vs production execution mode (Phase 44 distinction)."""
    RESEARCH = "research"
    PRODUCTION = "production"


@dataclass(slots=True)
class RuntimeConfig:
    """Canonical runtime configuration.

    Production mode fails closed: signed receipts, strict authority, and
    deterministic ordering are mandatory and cannot be silently relaxed.
    """
    mode: RuntimeMode = RuntimeMode.RESEARCH
    # Evidence / receipt persistence. When None, in-memory ledgers are used.
    evidence_path: str | Path | None = None
    receipt_path: str | Path | None = None
    signing_key: str | None = None
    require_signed_receipts: bool = False
    # WAL path for crash-safe transactions (v5.11 Phase 12).
    # When set, every commit writes BEGIN/WRITE/COMMIT records to the WAL.
    # In production mode, wal_path is required.
    wal_path: str | Path | None = None
    # Optional structural MPC planning (Phase 14). When horizon > 1 the
    # runtime plans before committing, but only ever executes the first
    # action of the chosen plan (receding horizon).
    mpc_horizon: int = 1
    mpc_max_branching: int = 8
    mpc_max_sequences: int = 64
    # Structural-learning loop knobs (delegated to StructuralLearningLoop).
    ensemble_size: int = 5
    max_candidates: int = 5
    # Canonical multi-objective structural planning weights:
    # J(a)=E[U]+nu*IG-lambda*Cost-mu*Risk-rho*Homeostasis.
    information_gain_weight: float = 0.1
    cost_weight: float = 1.0
    risk_weight: float = 0.5
    homeostasis_weight: float = 0.5
    # Optional external utility function used for MPC planning and credit.
    utility_fn: Callable[[GraphBuffers, Tensor], float] | None = None
    # Deterministic ordering guard: never rely on set/dict iteration order.
    deterministic_ordering: bool = True
    # Maximum stale-read retries before raising (Phase 3 seqlock enforcement).
    max_stale_read_retries: int = 4

    def __post_init__(self) -> None:
        if self.mode == RuntimeMode.PRODUCTION:
            # v5.11 Phase 11: production truly fails closed.
            # Production mode requires ALL of:
            # - signed receipts with a signing key
            # - persistent evidence store
            # - persistent receipt store
            # - deterministic ordering
            # - strict authority (enforced by the runtime, not config)
            if not self.require_signed_receipts:
                raise ValueError(
                    "production mode requires require_signed_receipts=True"
                )
            if self.signing_key is None:
                raise ValueError(
                    "production mode requires a signing_key"
                )
            if self.evidence_path is None:
                raise ValueError(
                    "production mode requires evidence_path (persistent evidence store)"
                )
            if self.receipt_path is None:
                raise ValueError(
                    "production mode requires receipt_path (persistent receipt store)"
                )
            if not self.deterministic_ordering:
                raise ValueError(
                    "production mode requires deterministic_ordering=True"
                )
            if self.wal_path is None:
                raise ValueError(
                    "production mode requires wal_path (crash-safe transaction log)"
                )
        if int(self.mpc_horizon) < 1:
            raise ValueError("mpc_horizon must be >= 1")
        if int(self.max_stale_read_retries) < 0:
            raise ValueError("max_stale_read_retries must be >= 0")

    @property
    def is_production(self) -> bool:
        return self.mode == RuntimeMode.PRODUCTION

    def to_summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "mpc_horizon": int(self.mpc_horizon),
            "ensemble_size": int(self.ensemble_size),
            "max_candidates": int(self.max_candidates),
            "information_gain_weight": float(self.information_gain_weight),
            "cost_weight": float(self.cost_weight),
            "risk_weight": float(self.risk_weight),
            "require_signed_receipts": bool(self.require_signed_receipts),
            "deterministic_ordering": bool(self.deterministic_ordering),
            "evidence_path": None if self.evidence_path is None else str(self.evidence_path),
            "receipt_path": None if self.receipt_path is None else str(self.receipt_path),
        }


# ---------------------------------------------------------------------------
# Phase 43: typed configuration presets and loader.
# ---------------------------------------------------------------------------

def research_runtime_config() -> RuntimeConfig:
    """Research preset: in-memory evidence, no signing, relaxed ordering constraints."""
    return RuntimeConfig(
        mode=RuntimeMode.RESEARCH,
        mpc_horizon=1,
        ensemble_size=5,
        max_candidates=5,
        require_signed_receipts=False,
        deterministic_ordering=True,
        max_stale_read_retries=4,
    )


def production_runtime_config(
    *,
    evidence_path: str | Path,
    receipt_path: str | Path,
    signing_key: str,
    wal_path: str | Path | None = None,
) -> RuntimeConfig:
    """Production preset: signed receipts, persisted evidence, WAL, fail-closed."""
    return RuntimeConfig(
        mode=RuntimeMode.PRODUCTION,
        evidence_path=str(evidence_path),
        receipt_path=str(receipt_path),
        signing_key=str(signing_key),
        require_signed_receipts=True,
        deterministic_ordering=True,
        max_stale_read_retries=8,
        mpc_horizon=1,
        ensemble_size=5,
        max_candidates=5,
        wal_path=str(wal_path) if wal_path else None,
    )


def benchmark_runtime_config() -> RuntimeConfig:
    """Benchmark preset: research mode, larger candidate sets, no persistence."""
    return RuntimeConfig(
        mode=RuntimeMode.RESEARCH,
        mpc_horizon=1,
        ensemble_size=3,
        max_candidates=16,
        require_signed_receipts=False,
        deterministic_ordering=True,
        max_stale_read_retries=2,
    )


PRESETS: dict[str, Callable[[], RuntimeConfig]] = {
    "research": research_runtime_config,
    "production": production_runtime_config,
    "benchmark": benchmark_runtime_config,
}


def load_runtime_config(preset: str | None = None, **overrides: Any) -> RuntimeConfig:
    """Load a runtime config by preset name with optional overrides.

    ``preset`` selects a preset factory; ``overrides`` are applied on top of
    the preset's config. For ``production``, the required arguments
    (evidence_path, receipt_path, signing_key) must be provided via overrides.
    """
    if preset is None:
        return RuntimeConfig(**overrides)
    if preset not in PRESETS:
        raise ValueError(f"unknown runtime config preset: {preset!r}; choose from {sorted(PRESETS.keys())}")
    base = PRESETS[preset]()
    # Apply overrides by replacing fields on the dataclass.
    if not overrides:
        return base
    from dataclasses import replace
    return replace(base, **overrides)
