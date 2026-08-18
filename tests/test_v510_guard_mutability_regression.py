"""v5.11 Phase 3: verify the guard is truly immutable.

After Phase 3, guard.graph returns a FrozenGraphView that defensively
clones tensors. Mutating through the guard raises UnauthorizedMutationError.

This test replaces the v5.10 regression test that documented the defect.
"""
from __future__ import annotations

import pytest
import torch

from lgae_v3.runtime import LGAERuntime, RuntimeConfig, UnauthorizedMutationError
from lgae_v3.runtime.state import FrozenGraphView
from lgae_v3.types import make_graph_buffers


def test_guard_graph_is_immutable():
    """The guard's graph property returns an immutable FrozenGraphView."""
    graph = make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=32)
    runtime = LGAERuntime(graph, runtime_config=RuntimeConfig())

    guard = runtime.guard_for("executive")
    frozen = guard.graph
    assert isinstance(frozen, FrozenGraphView)

    # Mutating the frozen view's weight must NOT change authoritative state.
    original_weight = graph.weight.clone()
    frozen_weight = frozen.weight.clone()

    # Try to mutate the frozen tensor — this is a clone, so it won't affect
    # the authoritative state.
    frozen_weight[0] = frozen_weight[0] * 2.0

    # The authoritative state must be unchanged.
    assert torch.equal(original_weight, graph.weight), (
        "Authoritative state was mutated through the frozen view!"
    )

    # Setting attributes on the frozen view must raise.
    with pytest.raises(UnauthorizedMutationError):
        frozen.weight = torch.zeros(32)

    with pytest.raises(UnauthorizedMutationError):
        frozen.custom_attr = 123


def test_guard_graph_is_frozen_view():
    """guard.graph returns a FrozenGraphView, not raw GraphBuffers."""
    graph = make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=32)
    runtime = LGAERuntime(graph, runtime_config=RuntimeConfig())
    guard = runtime.guard_for("executive")
    from lgae_v3.types import GraphBuffers
    # Must NOT be a raw GraphBuffers.
    assert not isinstance(guard.graph, GraphBuffers), (
        "guard.graph should return a FrozenGraphView, not raw GraphBuffers"
    )


def test_guard_fibers_are_frozen():
    """guard.fibers returns a FrozenFiberView."""
    graph = make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=32)
    runtime = LGAERuntime(graph, runtime_config=RuntimeConfig())
    guard = runtime.guard_for("executive")
    from lgae_v3.runtime.state import FrozenFiberView
    assert isinstance(guard.fibers, FrozenFiberView)

    # The fiber tensor must be a clone, not the original.
    original_z = runtime.engine.fibers().detach().clone()
    frozen_z = guard.fibers.z
    frozen_z[0, 0] = frozen_z[0, 0] * 2.0
    # Original must be unchanged.
    new_z = runtime.engine.fibers().detach().clone()
    assert torch.equal(original_z, new_z), (
        "Authoritative fiber state was mutated through the frozen view!"
    )
