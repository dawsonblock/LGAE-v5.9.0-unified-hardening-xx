"""v5.11 Phase 2: verify MPC is actually called during step().

After Phase 2, MPC.plan is called during plan() when horizon > 1.
This test replaces the v5.10 regression test that documented the defect.
"""
from __future__ import annotations

from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.types import make_graph_buffers


def test_mpc_is_called_in_step():
    """MPC.plan is called during step() when horizon > 1."""
    runtime_config = RuntimeConfig(mpc_horizon=3)
    runtime = LGAERuntime(
        make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=32),
        runtime_config=runtime_config,
    )

    # MPC is instantiated.
    assert runtime._mpc is not None

    # Track if MPC.plan is called.
    mpc_called = False
    orig_plan = runtime._mpc.plan

    def _tracking_plan(*args, **kwargs):
        nonlocal mpc_called
        mpc_called = True
        return orig_plan(*args, **kwargs)

    runtime._mpc.plan = _tracking_plan

    # Run step.
    runtime.step()

    # MPC.plan must be called during step().
    assert mpc_called, (
        "Expected MPC.plan to be called during step(). "
        "MPC is now wired into plan()."
    )
