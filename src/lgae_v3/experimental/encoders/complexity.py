"""Complexity-adjusted representation score.

    R_eff = PredictiveGain / RepresentationCost

where cost can include:
- encoding latency
- memory
- feature extraction time
- model parameters

This stops exp3 from choosing a 5-million-parameter graph encoder that
improves ranking accuracy by 0.2%.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time
import math
import numpy as np

from .protocol import StateActionRepresentation
from .probes import EncoderProbeReport, ProbeResult


@dataclass(slots=True)
class ComplexityMetrics:
    """Complexity metrics for an encoder."""
    encoder_id: str
    dimension: int
    n_parameters: int = 0
    encoding_latency_ms: float = 0.0
    memory_bytes: int = 0

    @property
    def complexity_score(self) -> float:
        """A scalar complexity score (higher = more complex)."""
        return (
            float(self.dimension) * 1.0
            + float(self.n_parameters) * 0.001
            + float(self.encoding_latency_ms) * 10.0
            + float(self.memory_bytes) * 0.0001
        )

    def to_log(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "dimension": int(self.dimension),
            "n_parameters": int(self.n_parameters),
            "encoding_latency_ms": float(self.encoding_latency_ms),
            "memory_bytes": int(self.memory_bytes),
            "complexity_score": float(self.complexity_score),
        }


@dataclass(slots=True)
class RepresentationComparison:
    """Complexity-adjusted representation comparison."""
    encoder_id: str
    dimension: int
    predictive_gain: float  # average gain over baseline across tasks
    complexity_score: float
    effectiveness_score: float  # predictive_gain / complexity_score
    probe_results: list[dict[str, Any]] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "dimension": int(self.dimension),
            "predictive_gain": float(self.predictive_gain),
            "complexity_score": float(self.complexity_score),
            "effectiveness_score": float(self.effectiveness_score),
            "probe_results": list(self.probe_results),
        }


def measure_encoding_latency(
    encoder: Any,
    state: Any,
    global_features: list[float],
    action_type: str,
    action_target: dict[str, Any],
    local_features: list[float],
    n_runs: int = 10,
) -> float:
    """Measure encoding latency in milliseconds."""
    # Warmup.
    for _ in range(3):
        encoder.encode(state, global_features, action_type, action_target, local_features)
    # Measure.
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        encoder.encode(state, global_features, action_type, action_target, local_features)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.median(times))


def compute_effectiveness(
    probe_report: EncoderProbeReport,
    complexity: ComplexityMetrics,
) -> RepresentationComparison:
    """Compute complexity-adjusted effectiveness score.

    R_eff = PredictiveGain / RepresentationCost
    """
    # Average gain over baseline across all tasks.
    gains = [r.gain_over_baseline for r in probe_report.results]
    avg_gain = float(np.mean(gains)) if gains else 0.0

    # Complexity score (higher = more expensive).
    cost = max(complexity.complexity_score, 1e-10)

    # Effectiveness = gain / cost.
    effectiveness = avg_gain / cost

    return RepresentationComparison(
        encoder_id=probe_report.encoder_id,
        dimension=probe_report.encoder_dimension,
        predictive_gain=avg_gain,
        complexity_score=complexity.complexity_score,
        effectiveness_score=effectiveness,
        probe_results=[r.to_log() for r in probe_report.results],
    )


def compare_encoders(
    probe_reports: list[EncoderProbeReport],
    complexity_metrics: list[ComplexityMetrics],
) -> list[RepresentationComparison]:
    """Compare multiple encoders by effectiveness.

    Returns a list sorted by effectiveness score (descending).
    """
    comparisons = []
    for probe, complexity in zip(probe_reports, complexity_metrics):
        comp = compute_effectiveness(probe, complexity)
        comparisons.append(comp)
    # Sort by effectiveness (descending).
    comparisons.sort(key=lambda c: c.effectiveness_score, reverse=True)
    return comparisons
