"""v5.10 Phase 37: formal invariant layer tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import make_graph_buffers
from lgae_v3.governance import (
    InvariantRegistry, InvariantResult, InvariantSeverity,
    invariant, register_invariant, DEFAULT_REGISTRY,
)


class _State:
    def __init__(self, graph=None, engine=None, receipt_path=None):
        self.graph = graph
        self.engine = engine
        self.receipt_path = receipt_path


def test_invariant_decorator_registers_on_default_registry():
    reg = InvariantRegistry()

    @invariant("always_true_test", severity=InvariantSeverity.BLOCKING)
    def _always_true(state):
        return True

    # Registered on DEFAULT_REGISTRY, not the local one.
    assert "always_true_test" in DEFAULT_REGISTRY.names
    results = DEFAULT_REGISTRY.check_all(_State())
    assert any(r.name == "always_true_test" and r.passed for r in results)


def test_invariant_violation_is_reported():
    reg = InvariantRegistry()
    reg.register("must_be_positive", lambda s: (s.x > 0, "x must be positive"))
    results = reg.check_all(type("S", (), {"x": -1})())
    r = next(r for r in results if r.name == "must_be_positive")
    assert not r.passed
    assert "x must be positive" in r.message


def test_invariant_raising_is_treated_as_failure():
    reg = InvariantRegistry()
    reg.register("raises", lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
    results = reg.check_all(_State())
    r = next(r for r in results if r.name == "raises")
    assert not r.passed
    assert "invariant raised" in r.message


def test_check_blocking_returns_all_passed_and_results():
    reg = InvariantRegistry()
    reg.register("ok", lambda s: (True, ""))
    reg.register("bad", lambda s: (False, "nope"))
    ok, results = reg.check_blocking(_State())
    assert not ok
    assert len(results) == 2


def test_check_blocking_ignores_warning_severity():
    reg = InvariantRegistry()
    reg.register("warn", lambda s: (False, "warn"), severity=InvariantSeverity.WARNING)
    ok, results = reg.check_blocking(_State())
    assert ok  # warnings do not block
    assert results == []


def test_invariant_names_are_sorted_deterministically():
    reg = InvariantRegistry()
    reg.register("zeta", lambda s: (True, ""))
    reg.register("alpha", lambda s: (True, ""))
    reg.register("mu", lambda s: (True, ""))
    assert reg.names == ["alpha", "mu", "zeta"]


def test_builtin_graph_buffers_finite_passes_on_clean_graph():
    g = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    results = DEFAULT_REGISTRY.check_all(_State(graph=g))
    r = next(r for r in results if r.name == "graph_buffers_finite")
    assert r.passed


def test_builtin_graph_buffers_finite_fails_on_nan():
    g = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    g.weight[0] = float("nan")
    results = DEFAULT_REGISTRY.check_all(_State(graph=g))
    r = next(r for r in results if r.name == "graph_buffers_finite")
    assert not r.passed


def test_builtin_positive_weights_fails_on_zero_weight():
    g = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    g.weight[0] = 0.0
    results = DEFAULT_REGISTRY.check_all(_State(graph=g))
    r = next(r for r in results if r.name == "graph_buffers_positive_weights")
    assert not r.passed


def test_builtin_laplacian_psd_passes_on_valid_graph():
    g = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    results = DEFAULT_REGISTRY.check_all(_State(graph=g))
    r = next(r for r in results if r.name == "laplacian_psd")
    assert r.passed


def test_register_invariant_helper():
    reg = InvariantRegistry()
    register_invariant("custom", lambda s: (True, ""), registry=reg)
    assert "custom" in reg.names


def test_invariant_result_to_log():
    r = InvariantResult(name="x", passed=True, severity=InvariantSeverity.BLOCKING, message="")
    log = r.to_log()
    assert log["name"] == "x" and log["passed"] and log["severity"] == "blocking"
