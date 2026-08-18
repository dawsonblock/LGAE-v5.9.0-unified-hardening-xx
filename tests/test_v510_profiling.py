"""v5.10 Phase 32: performance profiling tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import (
    PhaseTiming, ProfileReport, RuntimeProfiler,
)
from lgae_v3.runtime.runtime_events import RuntimePhase


def test_phase_timing_to_log():
    t = PhaseTiming(phase="observe", step=0, latency_ms=1.5)
    log = t.to_log()
    assert log["phase"] == "observe"
    assert log["step"] == 0
    assert log["latency_ms"] == 1.5


def test_runtime_profiler_records_latency():
    p = RuntimeProfiler()
    result = p.profile_phase(RuntimePhase.OBSERVE, 0, lambda: 42)
    assert result == 42
    assert len(p.report.timings) == 1
    assert p.report.timings[0].phase == "observe"
    assert p.report.timings[0].latency_ms >= 0.0


def test_profile_report_mean_latency_by_phase():
    report = ProfileReport()
    report.add(PhaseTiming("observe", 0, 1.0))
    report.add(PhaseTiming("observe", 1, 3.0))
    report.add(PhaseTiming("commit", 0, 5.0))
    means = report.mean_latency_by_phase()
    assert means["observe"] == 2.0
    assert means["commit"] == 5.0


def test_profile_report_max_latency_by_phase():
    report = ProfileReport()
    report.add(PhaseTiming("observe", 0, 1.0))
    report.add(PhaseTiming("observe", 1, 3.0))
    maxs = report.max_latency_by_phase()
    assert maxs["observe"] == 3.0


def test_profile_report_dominant_phase():
    report = ProfileReport()
    report.add(PhaseTiming("observe", 0, 1.0))
    report.add(PhaseTiming("commit", 0, 5.0))
    assert report.dominant_phase == "commit"


def test_profile_report_step_count():
    report = ProfileReport()
    report.add(PhaseTiming("observe", 0, 1.0))
    report.add(PhaseTiming("commit", 0, 2.0))
    report.add(PhaseTiming("observe", 1, 1.0))
    assert report.step_count == 2


def test_profile_report_total_latency():
    report = ProfileReport()
    report.add(PhaseTiming("a", 0, 1.0))
    report.add(PhaseTiming("b", 0, 2.0))
    assert report.total_latency_ms == 3.0


def test_profile_report_to_log():
    report = ProfileReport()
    report.add(PhaseTiming("observe", 0, 1.0))
    log = report.to_log()
    assert "step_count" in log
    assert "total_latency_ms" in log
    assert "dominant_phase" in log
    assert "mean_latency_by_phase" in log


def test_profiler_with_string_phase_name():
    p = RuntimeProfiler()
    p.profile_phase("custom_phase", 0, lambda: None)
    assert p.report.timings[0].phase == "custom_phase"


def test_profiler_metadata_recorded():
    p = RuntimeProfiler()
    p.profile_phase(RuntimePhase.COMMIT, 0, lambda: None, metadata={"action": "add_edge"})
    assert p.report.timings[0].metadata["action"] == "add_edge"
