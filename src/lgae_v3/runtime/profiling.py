"""Performance profiling (Phase 32).

A profiling harness that measures wall-clock time and memory for runtime
phases. Unlike the performance qualification (Phase 49), which checks
scalability tiers, profiling identifies *which phase* dominates the runtime
cycle and tracks per-phase latency distributions.

The profiler is non-invasive: it wraps phase execution and records timings
without altering the runtime's behavior. Results are structured for the
JSONL metrics sink (Phase 40).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .runtime_events import RuntimePhase


@dataclass(slots=True)
class PhaseTiming:
    """Timing for one phase execution."""
    phase: str
    step: int
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "phase": str(self.phase),
            "step": int(self.step),
            "latency_ms": float(self.latency_ms),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ProfileReport:
    """Aggregate profiling report across steps."""
    timings: list[PhaseTiming] = field(default_factory=list)

    def add(self, timing: PhaseTiming) -> None:
        self.timings.append(timing)

    @property
    def phase_names(self) -> list[str]:
        return sorted(set(t.phase for t in self.timings))

    def latency_by_phase(self) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        for t in self.timings:
            result.setdefault(t.phase, []).append(t.latency_ms)
        return result

    def mean_latency_by_phase(self) -> dict[str, float]:
        by_phase = self.latency_by_phase()
        return {p: sum(v) / len(v) for p, v in by_phase.items() if v}

    def max_latency_by_phase(self) -> dict[str, float]:
        by_phase = self.latency_by_phase()
        return {p: max(v) for p, v in by_phase.items() if v}

    @property
    def total_latency_ms(self) -> float:
        return sum(t.latency_ms for t in self.timings)

    @property
    def step_count(self) -> int:
        return len(set(t.step for t in self.timings))

    @property
    def dominant_phase(self) -> str | None:
        means = self.mean_latency_by_phase()
        if not means:
            return None
        return max(means, key=means.get)

    def to_log(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "total_latency_ms": float(self.total_latency_ms),
            "dominant_phase": self.dominant_phase,
            "mean_latency_by_phase": {k: float(v) for k, v in self.mean_latency_by_phase().items()},
            "max_latency_by_phase": {k: float(v) for k, v in self.max_latency_by_phase().items()},
            "timing_count": len(self.timings),
        }


class RuntimeProfiler:
    """Non-invasive profiler that wraps phase execution."""

    def __init__(self) -> None:
        self.report = ProfileReport()

    def profile_phase(
        self,
        phase: RuntimePhase | str,
        step: int,
        fn: Callable[[], Any],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Execute ``fn`` and record its latency. Returns fn's result."""
        phase_name = phase.value if isinstance(phase, RuntimePhase) else str(phase)
        t0 = time.perf_counter()
        result = fn()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.report.add(PhaseTiming(
            phase=phase_name, step=int(step), latency_ms=float(latency_ms),
            metadata=dict(metadata or {}),
        ))
        return result

    def to_log(self) -> dict[str, Any]:
        return self.report.to_log()
