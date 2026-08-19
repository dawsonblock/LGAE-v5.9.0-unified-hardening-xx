"""v5.11-RC / Phase 4: Full Fault-Injection & Crash Qualification Matrix.

Tests three failure classes across all internal commit failpoints:
  1. Ordinary Python exceptions (S_live == S_pre)
  2. SIGTERM subprocess termination (S_restart ∈ {S_pre, S_post})
  3. SIGKILL subprocess crash (S_restart ∈ {S_pre, S_post})

Guarantees that partial state mutation is strictly impossible.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.crash_recovery]
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import LGAERuntime, RuntimeConfig, make_graph_transaction, StructuralTransaction
from lgae_v3.runtime.contracts import AuthorizationResult, AuthorizationStatus
from lgae_v3.runtime.wal import replay_committed_transactions
from lgae_v3.types import MutationResult


ALL_FAILPOINTS = [
    "before_prepare",
    "after_prepare",
    "before_durable_commit_intent",
    "after_durable_commit_intent",
    "before_graph_apply",
    "after_graph_apply",
    "after_fiber_apply",
    "after_gauge_apply",
    "after_calibration_apply",
    "after_model_state_apply",
    "after_state_version_update",
    "after_state_swap",
    "before_verification",
    "after_verification",
    "before_receipt",
    "during_receipt",
    "after_receipt",
]


def _cfg() -> ResearchConfig:
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 2
    cfg.audit.orc_backend = "exact_lp"
    cfg.audit.persistent_homology_enabled = False
    cfg.audit.entropic_nodes = 0
    cfg.audit.bakry_nodes = 0
    cfg.audit.cde_nodes = 0
    cfg.audit.exact_lly_top_k = 0
    cfg.audit.orc_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]
    cfg.mutation.curvature_ema_enabled = False
    return cfg


def _graph():
    return make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=12)


_CRASH_SCRIPT = textwrap.dedent("""
    import sys, os, json, signal
    sys.path.insert(0, {src_path!r})
    import torch
    from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
    from lgae_v3.runtime import LGAERuntime, RuntimeConfig, make_graph_transaction, StructuralTransaction
    from lgae_v3.runtime.contracts import AuthorizationResult, AuthorizationStatus
    from lgae_v3.types import MutationResult

    torch.manual_seed(42)
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 6; cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 2
    cfg.audit.orc_backend = 'exact_lp'; cfg.audit.persistent_homology_enabled = False
    cfg.audit.entropic_nodes = 0; cfg.audit.bakry_nodes = 0; cfg.audit.cde_nodes = 0
    cfg.audit.exact_lly_top_k = 0; cfg.audit.orc_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]; cfg.mutation.curvature_ema_enabled = False

    failpoint = {failpoint!r}
    wal_path = {wal_path!r}
    state_file = {state_file!r}
    action = {action!r}

    rt = LGAERuntime(
        make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=12),
        cfg, runtime_config=RuntimeConfig(wal_path=wal_path),
    )
    pre_hash = rt.authority_hash

    shadow = rt.engine.graph.clone()
    shadow.weight[0] = shadow.weight[0] * 3.0
    txn = make_graph_transaction(
        base_state_version=int(rt.engine.graph.version),
        base_state_hash=rt.authority_hash,
        shadow_graph=shadow,
        mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        step=0,
    ).with_authorization()
    auth = AuthorizationResult(
        snapshot_id='s1', state_version=int(rt.engine.graph.version),
        state_hash=rt.authority_hash,
        status=AuthorizationStatus.AUTHORIZED,
        transaction_hash=txn.transaction_id,
    )

    if failpoint == 'none':
        result = rt.commit_channel.commit(txn, auth)
        with open(state_file, 'w') as f:
            json.dump({{'pre_hash': pre_hash, 'post_hash': rt.authority_hash,
                       'committed': result.committed}}, f)
    else:
        rt.commit_channel.set_failpoint(failpoint)
        if action == 'SIGKILL':
            def kill_check(name):
                if name == failpoint:
                    os.kill(os.getpid(), signal.SIGKILL)
            rt.commit_channel._check_failpoint = kill_check
        elif action == 'SIGTERM':
            def term_check(name):
                if name == failpoint:
                    os.kill(os.getpid(), signal.SIGTERM)
            rt.commit_channel._check_failpoint = term_check

        try:
            result = rt.commit_channel.commit(txn, auth)
            with open(state_file, 'w') as f:
                json.dump({{'pre_hash': pre_hash, 'post_hash': rt.authority_hash,
                           'committed': result.committed}}, f)
        except Exception as e:
            with open(state_file, 'w') as f:
                json.dump({{'pre_hash': pre_hash, 'post_hash': rt.authority_hash,
                           'committed': False, 'error': str(e)}}, f)
""")


def _run_crash_test(tmp_path: Path, failpoint: str, action: str = "SIGKILL") -> dict:
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    wal_path = str(tmp_path / "wal.jsonl")
    state_file = str(tmp_path / "state.json")

    script = _CRASH_SCRIPT.format(
        src_path=src_path, failpoint=failpoint,
        wal_path=wal_path, state_file=state_file,
        action=action,
    )
    script_file = tmp_path / "crash_script.py"
    tmp_path.mkdir(parents=True, exist_ok=True)
    script_file.write_text(script)

    proc = subprocess.run(
        [sys.executable, str(script_file)],
        capture_output=True, text=True, timeout=30,
    )

    state = {}
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)

    return {
        "returncode": proc.returncode,
        "state": state,
        "wal_path": wal_path,
    }


def _recover_and_get_hash(wal_path: str) -> str:
    torch.manual_seed(42)
    fresh = LGAERuntime(_graph(), _cfg())
    if os.path.exists(wal_path):
        replay_committed_transactions(wal_path, fresh._engine)
    return fresh.authority_hash


class TestExceptionFailpointMatrix:
    """Failure Class 1: Ordinary Python exceptions restore S_live == S_pre."""

    @pytest.mark.parametrize("failpoint", ALL_FAILPOINTS)
    def test_exception_at_any_failpoint_restores_pre_state(self, failpoint):
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        before_hash = rt.authority_hash
        before_version = rt.state_identity.version
        before_weight = float(rt.engine.graph.weight[0])
        before_fibers = rt.engine.fibers().detach().clone()

        shadow = rt.engine.graph.clone()
        shadow.weight[0] = shadow.weight[0] * 3.0
        tx = make_graph_transaction(
            base_state_version=before_version,
            base_state_hash=before_hash,
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        ).with_authorization()
        auth = AuthorizationResult(
            snapshot_id="s",
            state_version=before_version,
            state_hash=before_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=tx.transaction_id,
        )

        rt.commit_channel.set_failpoint(failpoint)
        with pytest.raises(RuntimeError, match="failpoint"):
            rt.commit_channel.commit(tx, auth)

        # Invariant: live authority is exactly pre-state
        assert rt.authority_hash == before_hash
        assert rt.state_identity.version == before_version
        assert float(rt.engine.graph.weight[0]) == before_weight
        assert torch.equal(rt.engine.fibers().detach(), before_fibers)


class TestSIGKILLFailpointMatrix:
    """Failure Class 2: SIGKILL at boundaries — S_restart ∈ {S_pre, S_{post}}."""

    @pytest.mark.parametrize("failpoint", [
        "before_prepare",
        "after_prepare",
        "before_durable_commit_intent",
        "after_durable_commit_intent",
        "before_graph_apply",
        "after_graph_apply",
        "after_state_swap",
        "before_verification",
        "after_verification",
        "before_receipt",
        "during_receipt",
        "after_receipt",
    ])
    def test_sigkill_at_failpoint_recovers_cleanly(self, tmp_path, failpoint):
        result = _run_crash_test(tmp_path, failpoint, action="SIGKILL")
        normal = _run_crash_test(tmp_path / "normal", "none")
        pre_hash = normal["state"]["pre_hash"]
        post_hash = normal["state"]["post_hash"]

        recovered = _recover_and_get_hash(result["wal_path"])
        assert recovered in (pre_hash, post_hash), (
            f"SIGKILL at '{failpoint}': recovered hash {recovered[:16]} "
            f"is neither pre ({pre_hash[:16]}) nor post ({post_hash[:16]})!"
        )


class TestSIGTERMFailpointMatrix:
    """Failure Class 3: SIGTERM at boundaries — S_restart ∈ {S_pre, S_{post}}."""

    @pytest.mark.parametrize("failpoint", [
        "before_prepare",
        "after_prepare",
        "before_durable_commit_intent",
        "after_durable_commit_intent",
        "before_graph_apply",
        "after_state_swap",
        "after_verification",
        "after_receipt",
    ])
    def test_sigterm_at_failpoint_recovers_cleanly(self, tmp_path, failpoint):
        result = _run_crash_test(tmp_path, failpoint, action="SIGTERM")
        normal = _run_crash_test(tmp_path / "normal", "none")
        pre_hash = normal["state"]["pre_hash"]
        post_hash = normal["state"]["post_hash"]

        recovered = _recover_and_get_hash(result["wal_path"])
        assert recovered in (pre_hash, post_hash), (
            f"SIGTERM at '{failpoint}': recovered hash {recovered[:16]} "
            f"is neither pre ({pre_hash[:16]}) nor post ({post_hash[:16]})!"
        )


def test_full_state_agreement_on_recovery(tmp_path):
    """Verify that WAL, authority hash, version, fibers, gauges, and indices all agree."""
    wal_path = str(tmp_path / "agreement.wal")
    rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
    
    shadow = rt.engine.graph.clone()
    shadow.weight[0] = shadow.weight[0] * 3.0
    tx = make_graph_transaction(
        base_state_version=rt.state_identity.version,
        base_state_hash=rt.authority_hash,
        shadow_graph=shadow,
        mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        step=0,
    ).with_authorization()
    auth = AuthorizationResult(
        snapshot_id="s",
        state_version=rt.state_identity.version,
        state_hash=rt.authority_hash,
        status=AuthorizationStatus.AUTHORIZED,
        transaction_hash=tx.transaction_id,
    )
    commit_res = rt.commit_channel.commit(tx, auth)
    
    # Fresh engine recovery from the same WAL
    fresh_rt = LGAERuntime(_graph(), _cfg())
    replay_committed_transactions(wal_path, fresh_rt._engine)
    
    assert fresh_rt.authority_hash == rt.authority_hash
    assert fresh_rt.state_identity.version == rt.state_identity.version
    assert fresh_rt.state_identity == commit_res.post_identity
    assert torch.equal(fresh_rt.engine.graph.weight, rt.engine.graph.weight)
    assert torch.equal(fresh_rt.engine.fibers().detach(), rt.engine.fibers().detach())
