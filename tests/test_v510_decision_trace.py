"""v5.10 Phase 41: runtime decision trace tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import TraceEntry, DecisionTrace
from lgae_v3.runtime.runtime_events import RuntimeEvent, RuntimePhase


def test_trace_entry_render():
    e = TraceEntry(step=0, phase="observe", summary="Observed graph state")
    line = e.render()
    assert "[step" in line and "observe" in line and "Observed graph state" in line


def test_decision_trace_add_event():
    trace = DecisionTrace()
    trace.add_event(RuntimeEvent(RuntimePhase.OBSERVE, step=0, payload={}))
    assert len(trace.entries) == 1
    assert trace.entries[0].phase == "observe"


def test_decision_trace_summarizes_payload():
    trace = DecisionTrace()
    trace.add_event(RuntimeEvent(RuntimePhase.COMMIT, step=0, payload={
        "authority_hash_after": "abcdef1234567890",
        "receipt_hash": "receipt123",
    }))
    rendered = trace.render()
    assert "hash=abcdef123456" in rendered
    assert "receipt=receipt123" in rendered


def test_decision_trace_summarizes_mutation_impact():
    trace = DecisionTrace()
    trace.add_event(RuntimeEvent(RuntimePhase.COMMIT, step=0, payload={
        "mutation_impact": {"topology": True, "weights": False, "gauges": False},
    }))
    rendered = trace.render()
    assert "impact=topology" in rendered


def test_decision_trace_summarizes_cache_invalidation():
    trace = DecisionTrace()
    trace.add_event(RuntimeEvent(RuntimePhase.CACHE_INVALIDATE, step=0, payload={
        "invalidated": ["topo_cache", "curvature_cache"],
        "spared": ["gauge_cache"],
    }))
    rendered = trace.render()
    assert "invalidated=topo_cache,curvature_cache" in rendered


def test_decision_trace_multiple_steps():
    trace = DecisionTrace()
    trace.add_event(RuntimeEvent(RuntimePhase.OBSERVE, step=0))
    trace.add_event(RuntimeEvent(RuntimePhase.COMMIT, step=0))
    trace.add_event(RuntimeEvent(RuntimePhase.OBSERVE, step=1))
    assert trace.step_count == 2
    assert len(trace.entries_for_step(0)) == 2
    assert len(trace.entries_for_step(1)) == 1


def test_decision_trace_render_full():
    trace = DecisionTrace()
    trace.add_event(RuntimeEvent(RuntimePhase.OBSERVE, step=0, payload={}))
    trace.add_event(RuntimeEvent(RuntimePhase.PROPOSE, step=0, payload={"n_candidates": 8}))
    rendered = trace.render()
    lines = rendered.strip().split("\n")
    assert len(lines) == 2
    assert "n_cand=8" in lines[1]


def test_decision_trace_to_log():
    trace = DecisionTrace()
    trace.add_event(RuntimeEvent(RuntimePhase.OBSERVE, step=0, payload={"x": 1}))
    log = trace.to_log()
    assert len(log) == 1
    assert log[0]["step"] == 0
    assert log[0]["phase"] == "observe"
    assert log[0]["payload"]["x"] == 1


def test_decision_trace_write_file(tmp_path):
    trace = DecisionTrace()
    trace.add_event(RuntimeEvent(RuntimePhase.OBSERVE, step=0))
    path = str(tmp_path / "trace.txt")
    trace.write_file(path)
    with open(path) as f:
        content = f.read()
    assert "observe" in content


def test_decision_trace_add_events_bulk():
    trace = DecisionTrace()
    events = [
        RuntimeEvent(RuntimePhase.OBSERVE, step=0),
        RuntimeEvent(RuntimePhase.COMMIT, step=0),
    ]
    trace.add_events(events)
    assert len(trace.entries) == 2
