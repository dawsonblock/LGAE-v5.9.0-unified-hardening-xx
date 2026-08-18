"""v5.11 Phase 2: verify the canonical 8-phase path is real.

After Phase 2, step() calls all 8 phase methods in order.
This test replaces the v5.10 regression test that documented the defect.
"""
from __future__ import annotations

from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.types import make_graph_buffers


def test_step_calls_all_eight_phases():
    """All 8 phase methods are called during step() in canonical order."""
    graph = make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=32)
    runtime = LGAERuntime(graph, runtime_config=RuntimeConfig())

    # Track which phase methods are called.
    called_phases: list[str] = []
    for phase_name in ("observe", "reason", "propose", "plan",
                       "evaluate", "authorize", "commit", "learn"):
        orig = getattr(runtime, phase_name)
        def _wrap(name, orig_fn):
            def wrapper(*args, **kwargs):
                called_phases.append(name)
                return orig_fn(*args, **kwargs)
            return wrapper
        setattr(runtime, phase_name, _wrap(phase_name, orig))

    # Run one step.
    runtime.step()

    # All 8 phases must be called.
    all_phases = {"observe", "reason", "propose", "plan",
                  "evaluate", "authorize", "commit", "learn"}
    called_set = set(called_phases)
    assert called_set == all_phases, (
        f"Expected all 8 phases to be called, but got: {called_phases}. "
        f"Missing: {all_phases - called_set}"
    )

    # Phases must be called in canonical order.
    # (observe may be called twice — once directly, once from step.
    #  We check the first occurrence of each.)
    first_occurrence: dict[str, int] = {}
    for i, name in enumerate(called_phases):
        if name not in first_occurrence:
            first_occurrence[name] = i
    expected_order = ["observe", "reason", "propose", "plan",
                      "evaluate", "authorize", "commit", "learn"]
    actual_order = sorted(first_occurrence.keys(), key=lambda k: first_occurrence[k])
    assert actual_order == expected_order, (
        f"Expected canonical order {expected_order}, got {actual_order}"
    )


def test_step_emits_phase_order_in_metadata():
    """The step result metadata contains the phase execution order."""
    graph = make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=32)
    runtime = LGAERuntime(graph, runtime_config=RuntimeConfig())
    result = runtime.step()
    phase_order = result.metadata.get("phase_order", [])
    assert phase_order == ["observe", "reason", "propose", "plan",
                           "evaluate", "authorize", "commit", "learn"]
