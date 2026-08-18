"""v5.10 Phase 40: runtime observability (JSONL metrics) tests."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from lgae_v3.runtime import (
    MetricsSink, Counter, Gauge, Histogram, read_jsonl,
)
from lgae_v3.runtime.runtime_events import RuntimeEvent, RuntimePhase


def test_counter_increments():
    c = Counter("x")
    c.inc(); c.inc(3)
    assert c.value == 4
    assert c.to_log()["value"] == 4


def test_gauge_sets_value():
    g = Gauge("g")
    g.set(3.14)
    assert g.value == 3.14


def test_histogram_observe_and_mean():
    h = Histogram("h")
    h.observe(0.5); h.observe(1.5)
    assert h.count == 2
    assert h.mean == 1.0
    log = h.to_log()
    assert log["count"] == 2 and log["sum"] == 2.0


def test_metrics_sink_writes_jsonl_events(tmp_path):
    path = str(tmp_path / "metrics.jsonl")
    sink = MetricsSink(path)
    sink.record_event(RuntimeEvent(RuntimePhase.OBSERVE, step=0, payload={"x": 1}))
    sink.record_event(RuntimeEvent(RuntimePhase.COMMIT, step=0, payload={"y": 2}))
    sink.close()
    lines = read_jsonl(path)
    assert len(lines) == 2
    assert lines[0]["phase"] == "observe"
    assert lines[1]["phase"] == "commit"


def test_metrics_sink_counters_per_phase(tmp_path):
    path = str(tmp_path / "metrics.jsonl")
    sink = MetricsSink(path)
    sink.record_event(RuntimeEvent(RuntimePhase.OBSERVE, step=0))
    sink.record_event(RuntimeEvent(RuntimePhase.OBSERVE, step=1))
    sink.record_event(RuntimeEvent(RuntimePhase.COMMIT, step=1))
    snap = sink.snapshot()
    assert snap["counters"]["phase.observe"]["value"] == 2
    assert snap["counters"]["phase.commit"]["value"] == 1
    assert sink.events_written == 3
    sink.close()


def test_metrics_sink_observe_latency_and_set_gauge(tmp_path):
    sink = MetricsSink(str(tmp_path / "m.jsonl"))
    try:
        sink.observe_latency("step_latency", 0.005)  # 5 ms
        sink.set_gauge("graph_generation", 42)
        snap = sink.snapshot()
        assert snap["histograms"]["step_latency"]["count"] == 1
        assert snap["gauges"]["graph_generation"]["value"] == 42.0
    finally:
        sink.close()


def test_metrics_sink_snapshot_is_sorted(tmp_path):
    sink = MetricsSink(str(tmp_path / "m.jsonl"))
    try:
        sink.set_gauge("z_gauge", 1)
        sink.set_gauge("a_gauge", 2)
        snap = sink.snapshot()
        keys = list(snap["gauges"].keys())
        assert keys == sorted(keys)
    finally:
        sink.close()


def test_metrics_sink_context_manager_closes_file(tmp_path):
    path = str(tmp_path / "m.jsonl")
    with MetricsSink(path) as sink:
        sink.record_event(RuntimeEvent(RuntimePhase.OBSERVE, step=0))
    # File should be closed and readable.
    assert len(read_jsonl(path)) == 1


def test_metrics_sink_no_file_works_in_memory():
    sink = MetricsSink(path=None)
    sink.record_event(RuntimeEvent(RuntimePhase.OBSERVE, step=0))
    assert sink.events_written == 1
    snap = sink.snapshot()
    assert snap["counters"]["phase.observe"]["value"] == 1
