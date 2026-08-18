"""Runtime observability (Phase 40).

A JSONL metrics sink that durably records runtime events and aggregates
counters, gauges, and histograms. The runtime emits structured events at
every cycle phase boundary; the ``MetricsSink`` writes them to a JSONL file
and keeps in-memory aggregates for quick reporting.

This builds on the existing ``RuntimeEvent`` infrastructure. It does not
replace the in-memory event list; it adds a durable sidecar.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, TextIO

from .runtime_events import RuntimeEvent, RuntimePhase


@dataclass(slots=True)
class Counter:
    name: str
    value: int = 0

    def inc(self, n: int = 1) -> None:
        self.value += int(n)

    def to_log(self) -> dict[str, Any]:
        return {"name": self.name, "value": int(self.value)}


@dataclass(slots=True)
class Gauge:
    name: str
    value: float = 0.0

    def set(self, v: float) -> None:
        self.value = float(v)

    def to_log(self) -> dict[str, Any]:
        return {"name": self.name, "value": float(self.value)}


_DEFAULT_BUCKETS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


@dataclass(slots=True)
class Histogram:
    """A bounded histogram with log-spaced buckets."""
    name: str
    buckets: tuple[float, ...] = field(default_factory=lambda: _DEFAULT_BUCKETS)
    counts: list[int] = field(default_factory=list)
    sum: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * (len(self.buckets) + 1)  # +1 for +Inf

    def observe(self, v: float) -> None:
        v = float(v)
        self.sum += v
        self.count += 1
        for i, b in enumerate(self.buckets):
            if v <= b:
                self.counts[i] += 1
                return
        self.counts[-1] += 1  # +Inf bucket

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def to_log(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": int(self.count),
            "sum": float(self.sum),
            "mean": float(self.mean),
            "buckets": list(self.buckets),
            "counts": list(self.counts),
        }


class MetricsSink:
    """Durable JSONL metrics sink + in-memory aggregates.

    Writes one JSON line per event to ``path`` (append-only). Maintains
    counters per phase, gauges for the latest values, and histograms for
    latency-like observations. Thread-safe via a single lock.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._events_written: int = 0
        self._fp: TextIO | None = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            self._fp = open(path, "a", encoding="utf-8")

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                self._fp.close()
                self._fp = None

    def __enter__(self) -> "MetricsSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def counter(self, name: str) -> Counter:
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name)
            return self._counters[name]

    def gauge(self, name: str) -> Gauge:
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name)
            return self._gauges[name]

    def histogram(self, name: str, buckets: tuple[float, ...] | None = None) -> Histogram:
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, buckets=buckets or _DEFAULT_BUCKETS)
            return self._histograms[name]

    def record_event(self, event: RuntimeEvent) -> None:
        """Write an event to the JSONL sink and update aggregates."""
        line = json.dumps(event.to_log(), sort_keys=True, default=str)
        with self._lock:
            if self._fp is not None:
                self._fp.write(line + "\n")
                self._fp.flush()
            self._events_written += 1
            # Per-phase counter.
            key = f"phase.{event.phase.value}"
            if key not in self._counters:
                self._counters[key] = Counter(key)
            self._counters[key].inc()

    def record_events(self, events: Iterable[RuntimeEvent]) -> None:
        for e in events:
            self.record_event(e)

    def observe_latency(self, name: str, seconds: float) -> None:
        self.histogram(name).observe(float(seconds) * 1000.0)  # ms

    def set_gauge(self, name: str, value: float) -> None:
        self.gauge(name).set(float(value))

    @property
    def events_written(self) -> int:
        with self._lock:
            return int(self._events_written)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "events_written": int(self._events_written),
                "counters": {k: c.to_log() for k, c in sorted(self._counters.items())},
                "gauges": {k: g.to_log() for k, g in sorted(self._gauges.items())},
                "histograms": {k: h.to_log() for k, h in sorted(self._histograms.items())},
            }


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file back into a list of dicts (for verification/tests)."""
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
