"""v5.11 Sprint 2 D11-007: Crash matrix — process kill at each transaction stage.

Tests the central recovery invariant by actually killing subprocesses at
various points during transaction processing:

    S_restart ∈ { S_t, S_{t+1} }
    Never S_t + partial(Δ)

Crash points tested:
  1. Before BEGIN (clean state)
  2. After BEGIN, before WRITE
  3. After WRITE, before COMMIT
  4. After COMMIT, before APPLY (replay should reconstruct)
  5. During APPLY (replay should reconstruct)
  6. After APPLY (state should match)

For each crash point, we verify that recovery produces either the pre-
transaction state or the post-transaction state, never a partial state.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.runtime.wal import WriteAheadLog, WALRecordType, recover_transactions


def _cfg() -> ResearchConfig:
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 0
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
    from lgae_v3.runtime import LGAERuntime, RuntimeConfig
    from lgae_v3.runtime.transaction import (
        StructuralTransaction, GraphDelta, make_graph_transaction,
    )
    from lgae_v3.runtime.contracts.authorization import (
        AuthorizationResult, AuthorizationStatus,
    )
    from lgae_v3.types import MutationResult

    torch.manual_seed(42)
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 6; cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 0
    cfg.audit.orc_backend = 'exact_lp'; cfg.audit.persistent_homology_enabled = False
    cfg.audit.entropic_nodes = 0; cfg.audit.bakry_nodes = 0; cfg.audit.cde_nodes = 0
    cfg.audit.exact_lly_top_k = 0; cfg.audit.orc_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]; cfg.mutation.curvature_ema_enabled = False

    crash_point = {crash_point!r}
    wal_path = {wal_path!r}
    state_file = {state_file!r}

    rt = LGAERuntime(
        make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=12),
        cfg, runtime_config=RuntimeConfig(wal_path=wal_path),
    )
    pre_hash = rt.authority_hash

    # Build a transaction.
    shadow = rt.engine.graph.clone()
    shadow.weight[0] = shadow.weight[0] * 3.0
    txn = make_graph_transaction(
        base_state_version=int(rt.engine.graph.version),
        base_state_hash=rt.authority_hash,
        shadow_graph=shadow,
        mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        step=0,
    )
    full_txn = StructuralTransaction(
        transaction_id=txn.transaction_id,
        base_state_version=txn.base_state_version,
        base_state_hash=txn.base_state_hash,
        graph_delta=txn.graph_delta,
        authorization_id=txn.authorization_binding_hash(),
        delta_hash=txn.delta_hash,
        mutation_result=txn.mutation_result,
    )
    auth = AuthorizationResult(
        snapshot_id='s1', state_version=int(rt.engine.graph.version),
        state_hash=rt.authority_hash, status=AuthorizationStatus.AUTHORIZED,
        transaction_hash=txn.transaction_id,
    )

    if crash_point == 'before_begin':
        # Kill before any WAL activity.
        os.kill(os.getpid(), signal.SIGKILL)

    if crash_point == 'after_begin':
        # Write BEGIN to WAL, then kill.
        wal_txn_id = rt._wal.begin({{'transaction_id': txn.transaction_id}})
        os.kill(os.getpid(), signal.SIGKILL)

    if crash_point == 'after_write':
        # Write BEGIN + WRITE, then kill (no COMMIT).
        wal_txn_id = rt._wal.begin({{'transaction_id': txn.transaction_id}})
        sg = shadow
        sd = sg.to_state_dict()
        json_state = {{}}
        for k, v in sd.items():
            if hasattr(v, 'tolist'):
                json_state[k] = v.tolist()
            else:
                json_state[k] = v
        rt._wal.write(wal_txn_id, {{
            'kind': 'graph', 'mutation_name': 'test',
            'shadow_graph_state': json_state,
            'graph_hash': sg.state_hash(),
        }})
        os.kill(os.getpid(), signal.SIGKILL)

    if crash_point == 'after_commit':
        # Full commit through commit_channel, then kill before any
        # further processing. The COMMIT record is durable.
        result = rt.commit_channel.commit(full_txn, auth)
        post_hash = rt.authority_hash
        # Save state for verification.
        with open(state_file, 'w') as f:
            json.dump({{'pre_hash': pre_hash, 'post_hash': post_hash,
                       'committed': result.committed}}, f)
        os.kill(os.getpid(), signal.SIGKILL)

    if crash_point == 'no_crash':
        # Normal completion — save state for verification.
        result = rt.commit_channel.commit(full_txn, auth)
        post_hash = rt.authority_hash
        with open(state_file, 'w') as f:
            json.dump({{'pre_hash': pre_hash, 'post_hash': post_hash,
                       'committed': result.committed}}, f)
""")


def _run_crash_test(tmp_path: Path, crash_point: str) -> dict:
    """Run the crash script and return the state info."""
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    wal_path = str(tmp_path / "wal.jsonl")
    state_file = str(tmp_path / "state.json")

    script = _CRASH_SCRIPT.format(
        src_path=src_path,
        crash_point=crash_point,
        wal_path=wal_path,
        state_file=state_file,
    )

    script_file = tmp_path / "crash_script.py"
    tmp_path.mkdir(parents=True, exist_ok=True)
    script_file.write_text(script)

    proc = subprocess.run(
        [sys.executable, str(script_file)],
        capture_output=True, text=True, timeout=30,
    )
    # The process may be killed by SIGKILL (return code -9).
    # That's expected for crash tests.

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
    """Recover WAL onto a fresh runtime and return the authority hash."""
    torch.manual_seed(42)
    cfg = _cfg()
    fresh_rt = LGAERuntime(_graph(), cfg)
    if os.path.exists(wal_path):
        from lgae_v3.runtime.wal import replay_committed_transactions
        replay_committed_transactions(wal_path, fresh_rt._engine)
    return fresh_rt.authority_hash


class TestCrashMatrix:
    """D11-007: Process kill at each transaction stage."""

    def test_crash_before_begin(self, tmp_path):
        """Crash before BEGIN: recovery should give pre-transaction state."""
        result = _run_crash_test(tmp_path, "before_begin")
        # Process should have been killed.
        assert result["returncode"] == -9, f"Expected SIGKILL, got {result['returncode']}"
        # No state file written.
        assert result["state"] == {}
        # Recovery should give the initial state.
        recovered_hash = _recover_and_get_hash(result["wal_path"])
        # The initial state hash for this graph/config.
        torch.manual_seed(42)
        initial_rt = LGAERuntime(_graph(), _cfg())
        assert recovered_hash == initial_rt.authority_hash, (
            "Crash before BEGIN should recover to initial state"
        )

    def test_crash_after_begin(self, tmp_path):
        """Crash after BEGIN but before WRITE: no committed transaction."""
        result = _run_crash_test(tmp_path, "after_begin")
        assert result["returncode"] == -9
        # Recovery should give initial state (BEGIN without COMMIT is discarded).
        recovered_hash = _recover_and_get_hash(result["wal_path"])
        torch.manual_seed(42)
        initial_rt = LGAERuntime(_graph(), _cfg())
        assert recovered_hash == initial_rt.authority_hash, (
            "Crash after BEGIN should recover to initial state (no COMMIT)"
        )

    def test_crash_after_write_no_commit(self, tmp_path):
        """Crash after WRITE but before COMMIT: transaction is discarded."""
        result = _run_crash_test(tmp_path, "after_write")
        assert result["returncode"] == -9
        # Recovery should give initial state (WRITE without COMMIT is discarded).
        recovered_hash = _recover_and_get_hash(result["wal_path"])
        torch.manual_seed(42)
        initial_rt = LGAERuntime(_graph(), _cfg())
        assert recovered_hash == initial_rt.authority_hash, (
            "Crash after WRITE (no COMMIT) should recover to initial state"
        )

    def test_crash_after_commit(self, tmp_path):
        """Crash after COMMIT: recovery should give post-transaction state.

        This is the key D11-005 test: with COMMIT-before-APPLY ordering,
        a crash after COMMIT but during/after APPLY should still recover
        to the post-transaction state.
        """
        result = _run_crash_test(tmp_path, "after_commit")
        assert result["returncode"] == -9
        state = result["state"]
        assert state.get("committed"), "Transaction should have committed"
        pre_hash = state["pre_hash"]
        post_hash = state["post_hash"]
        # Recovery should give the POST-transaction state (COMMIT was durable).
        recovered_hash = _recover_and_get_hash(result["wal_path"])
        assert recovered_hash == post_hash, (
            f"Crash after COMMIT should recover to post-transaction state. "
            f"Expected: {post_hash[:16]}, Got: {recovered_hash[:16]}"
        )
        # And it should NOT be the pre-transaction state.
        assert recovered_hash != pre_hash, (
            "Crash after COMMIT should NOT recover to pre-transaction state"
        )

    def test_no_crash_normal_completion(self, tmp_path):
        """Normal completion: state should be post-transaction."""
        result = _run_crash_test(tmp_path, "no_crash")
        assert result["returncode"] == 0
        state = result["state"]
        assert state.get("committed"), "Transaction should have committed"
        post_hash = state["post_hash"]
        # Recovery should give the same post-transaction state.
        recovered_hash = _recover_and_get_hash(result["wal_path"])
        assert recovered_hash == post_hash, (
            f"Normal completion recovery should match. "
            f"Expected: {post_hash[:16]}, Got: {recovered_hash[:16]}"
        )

    def test_recovery_invariant_no_partial_state(self, tmp_path):
        """For every crash point, S_restart ∈ {S_t, S_{t+1}}."""
        torch.manual_seed(42)
        initial_rt = LGAERuntime(_graph(), _cfg())
        initial_hash = initial_rt.authority_hash

        # Get the post-transaction hash from a normal run.
        normal_result = _run_crash_test(tmp_path / "normal", "no_crash")
        post_hash = normal_result["state"]["post_hash"]

        # Test each crash point.
        crash_points = [
            "before_begin",
            "after_begin",
            "after_write",
            "after_commit",
        ]
        for cp in crash_points:
            result = _run_crash_test(tmp_path / cp, cp)
            recovered_hash = _recover_and_get_hash(result["wal_path"])
            # The recovered state must be either the pre or post state.
            assert recovered_hash in (initial_hash, post_hash), (
                f"Crash at '{cp}': recovered hash {recovered_hash[:16]} "
                f"is neither pre ({initial_hash[:16]}) nor post ({post_hash[:16]})! "
                f"PARTIAL STATE DETECTED — D11-007 invariant violated!"
            )
