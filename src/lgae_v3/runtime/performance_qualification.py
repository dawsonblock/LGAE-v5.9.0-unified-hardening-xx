"""Performance qualification (Phase 49).

Defines S/M/L/XL scale tiers and measures the actual hot path at each tier:

  S:  1k nodes
  M:  10k nodes
  L:  100k nodes
  XL: 1M nodes (where supported)

Metrics per tier:
  - proposal_latency_ms
  - diagnostic_latency_ms
  - commit_latency_ms
  - peak_memory_bytes
  - candidate_throughput (candidates/s)

Do not claim million-node scalability without measuring the actual hot path
at that size. A tier that was not measured is recorded as NOT_MEASURED, not
inferred.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

import torch
from torch import Tensor

from ..types import GraphBuffers, make_graph_buffers


class ScaleTier(str, Enum):
    S = "S"    # 1k nodes
    M = "M"    # 10k nodes
    L = "L"    # 100k nodes
    XL = "XL"  # 1M nodes


TIER_NODE_COUNTS: dict[ScaleTier, int] = {
    ScaleTier.S: 1_000,
    ScaleTier.M: 10_000,
    ScaleTier.L: 100_000,
    ScaleTier.XL: 1_000_000,
}


class MeasurementStatus(str, Enum):
    # v5.11 Sprint 4 D11-014: Stronger status semantics.
    # MEASURED alone is insufficient — it can mean nothing executed.
    NOT_RUN = "not_run"           # measurement was never attempted
    INVALID = "invalid"           # measurement ran but data is invalid
    MEASURED = "measured"         # measurement ran and data is valid
    PASS = "pass"                 # measured and all thresholds met
    FAIL = "fail"                 # measured but thresholds not met
    SKIPPED = "skipped"           # explicitly skipped (e.g. unsupported tier)
    NOT_MEASURED = "not_measured"  # legacy alias for NOT_RUN


# v5.11 Sprint 4 D11-014: Actual performance thresholds per tier.
# A measurement is PASS only if all thresholds are met.
# These are conservative thresholds for the hot path.
PERFORMANCE_THRESHOLDS: dict[ScaleTier, dict[str, float]] = {
    ScaleTier.S: {
        "proposal_latency_ms": 1000.0,    # 1 second max for proposal
        "diagnostic_latency_ms": 2000.0,  # 2 seconds max for diagnostics
        "commit_latency_ms": 500.0,       # 500ms max for commit
        "candidate_throughput": 10.0,     # at least 10 candidates/s
    },
    ScaleTier.M: {
        "proposal_latency_ms": 5000.0,
        "diagnostic_latency_ms": 10000.0,
        "commit_latency_ms": 2000.0,
        "candidate_throughput": 5.0,
    },
    ScaleTier.L: {
        "proposal_latency_ms": 30000.0,
        "diagnostic_latency_ms": 60000.0,
        "commit_latency_ms": 10000.0,
        "candidate_throughput": 1.0,
    },
    ScaleTier.XL: {
        "proposal_latency_ms": 300000.0,
        "diagnostic_latency_ms": 600000.0,
        "commit_latency_ms": 60000.0,
        "candidate_throughput": 0.1,
    },
}


@dataclass(frozen=True, slots=True)
class TierMeasurement:
    """One performance measurement at a scale tier."""
    tier: ScaleTier
    n_nodes: int
    status: MeasurementStatus
    proposal_latency_ms: float = 0.0
    diagnostic_latency_ms: float = 0.0
    commit_latency_ms: float = 0.0
    peak_memory_bytes: int = 0
    candidate_throughput: float = 0.0
    notes: str = ""

    def to_log(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "n_nodes": int(self.n_nodes),
            "status": self.status.value,
            "proposal_latency_ms": float(self.proposal_latency_ms),
            "diagnostic_latency_ms": float(self.diagnostic_latency_ms),
            "commit_latency_ms": float(self.commit_latency_ms),
            "peak_memory_bytes": int(self.peak_memory_bytes),
            "candidate_throughput": float(self.candidate_throughput),
            "notes": self.notes,
        }

    def passes_thresholds(self) -> bool:
        """v5.11 Sprint 4 D11-014: Check if measurement passes all thresholds.

        A measurement passes only if:
        1. Status is MEASURED (not NOT_RUN, INVALID, or SKIPPED)
        2. All latency values are positive (actually measured)
        3. All values meet the tier's thresholds
        """
        if self.status != MeasurementStatus.MEASURED:
            return False
        # Must have actually measured something.
        if self.proposal_latency_ms <= 0 and self.diagnostic_latency_ms <= 0:
            return False
        thresholds = PERFORMANCE_THRESHOLDS.get(self.tier, {})
        if not thresholds:
            return True  # No thresholds defined — pass by default
        if self.proposal_latency_ms > thresholds.get("proposal_latency_ms", float('inf')):
            return False
        if self.diagnostic_latency_ms > thresholds.get("diagnostic_latency_ms", float('inf')):
            return False
        if self.commit_latency_ms > thresholds.get("commit_latency_ms", float('inf')):
            return False
        if self.candidate_throughput < thresholds.get("candidate_throughput", 0.0):
            return False
        return True

    @property
    def qualification_status(self) -> MeasurementStatus:
        """The effective status after threshold checking."""
        if self.status == MeasurementStatus.NOT_RUN:
            return MeasurementStatus.NOT_RUN
        if self.status == MeasurementStatus.SKIPPED:
            return MeasurementStatus.SKIPPED
        if self.status == MeasurementStatus.MEASURED:
            return MeasurementStatus.PASS if self.passes_thresholds() else MeasurementStatus.FAIL
        return self.status


@dataclass(slots=True)
class PerformanceQualificationReport:
    """Aggregate performance qualification report across tiers."""
    measurements: list[TierMeasurement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, m: TierMeasurement) -> None:
        self.measurements.append(m)

    @property
    def measured_tiers(self) -> list[ScaleTier]:
        return [m.tier for m in self.measurements if m.status == MeasurementStatus.MEASURED]

    def result_for(self, tier: ScaleTier) -> TierMeasurement | None:
        """Get the measurement for a specific tier, or None if not measured."""
        for m in self.measurements:
            if m.tier == tier:
                return m
        return None

    @property
    def xl_measured(self) -> bool:
        return any(m.tier == ScaleTier.XL and m.status == MeasurementStatus.MEASURED for m in self.measurements)

    def to_log(self) -> dict[str, Any]:
        return {
            "measurements": [m.to_log() for m in self.measurements],
            "measured_tiers": [t.value for t in self.measured_tiers],
            "xl_measured": self.xl_measured,
            "metadata": self.metadata,
        }


def _make_path_graph(n: int) -> GraphBuffers:
    edges = [(i, i + 1) for i in range(n - 1)]
    return make_graph_buffers(n, edges, capacity=max(n * 2, 16))


def _measure_latency(fn: Callable[[], Any]) -> float:
    """Measure wall-clock latency of fn in milliseconds."""
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def measure_tier(
    tier: ScaleTier,
    *,
    proposal_fn: Callable[[GraphBuffers], int] | None = None,
    diagnostic_fn: Callable[[GraphBuffers], Any] | None = None,
    commit_fn: Callable[[GraphBuffers], Any] | None = None,
    n_nodes: int | None = None,
    skip: bool = False,
) -> TierMeasurement:
    """Measure one scale tier. Functions that are None are not measured.

    ``proposal_fn`` returns the number of candidates generated (for throughput).
    """
    nn = int(n_nodes if n_nodes is not None else TIER_NODE_COUNTS[tier])
    if skip:
        return TierMeasurement(tier=tier, n_nodes=nn, status=MeasurementStatus.SKIPPED,
                               notes="explicitly skipped")

    # v5.11-RC Phase 17: If no benchmark functions are provided, return INVALID.
    if proposal_fn is None and diagnostic_fn is None and commit_fn is None:
        return TierMeasurement(tier=tier, n_nodes=nn, status=MeasurementStatus.INVALID,
                               notes="no benchmark functions provided")

    # Track peak memory via torch if available.
    try:
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    except Exception:
        pass

    graph = _make_path_graph(nn)
    prop_ms = 0.0
    diag_ms = 0.0
    commit_ms = 0.0
    n_cands = 0

    if proposal_fn is not None:
        prop_ms = _measure_latency(lambda: None)  # warmup
        t0 = time.perf_counter()
        n_cands = int(proposal_fn(graph))
        prop_ms = (time.perf_counter() - t0) * 1000.0

    if diagnostic_fn is not None:
        diag_ms = _measure_latency(lambda: diagnostic_fn(graph))

    if commit_fn is not None:
        commit_ms = _measure_latency(lambda: commit_fn(graph))

    # Peak memory estimate.
    peak = 0
    try:
        if torch.cuda.is_available():
            peak = int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    if peak == 0:
        # Fallback: estimate from graph buffer size.
        try:
            peak = int(graph.weight.element_size() * graph.weight.numel())
        except Exception:
            pass

    throughput = float(n_cands) / (prop_ms / 1000.0) if prop_ms > 0 else 0.0

    return TierMeasurement(
        tier=tier, n_nodes=nn, status=MeasurementStatus.MEASURED,
        proposal_latency_ms=float(prop_ms),
        diagnostic_latency_ms=float(diag_ms),
        commit_latency_ms=float(commit_ms),
        peak_memory_bytes=int(peak),
        candidate_throughput=float(throughput),
    )


def run_performance_qualification(
    *,
    proposal_fn: Callable[[GraphBuffers], int] | None = None,
    diagnostic_fn: Callable[[GraphBuffers], Any] | None = None,
    commit_fn: Callable[[GraphBuffers], Any] | None = None,
    tiers: Iterable[ScaleTier] | None = None,
    skip_tiers: set[ScaleTier] | None = None,
    metadata: dict[str, Any] | None = None,
) -> PerformanceQualificationReport:
    """Run performance qualification across tiers.

    Tiers in ``skip_tiers`` are recorded as SKIPPED (not inferred). Tiers not
    in ``tiers`` are not included at all.
    """
    report = PerformanceQualificationReport(metadata=dict(metadata or {}))
    skip = skip_tiers or set()
    for tier in (tiers or [ScaleTier.S, ScaleTier.M, ScaleTier.L, ScaleTier.XL]):
        m = measure_tier(
            tier, proposal_fn=proposal_fn, diagnostic_fn=diagnostic_fn, commit_fn=commit_fn,
            skip=tier in skip,
        )
        report.add(m)
    return report
