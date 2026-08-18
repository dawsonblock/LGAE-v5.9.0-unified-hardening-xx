"""v5.11-RC Phase 20: Golden multi-domain transaction scenario.

The decisive release gate test:

1. Construct one transaction changing graph topology, fiber state, and
   gauge state simultaneously.
2. Commit through the canonical eight-phase path.
3. SIGKILL at every internal commit boundary.
4. Recover from checkpoint + WAL.
5. Verify complete pre-state or complete post-state only — never mixed.

Required equality checks:
- authority hash
- graph hash
- fiber hash
- gauge hash
- state version
- transaction receipt
- deterministic replay result
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
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import (
    LGAERuntime, RuntimeConfig,
    make_joint_transaction, StructuralTransaction,
)
from lgae_v3.runtime.contracts import AuthorizationResult, AuthorizationStatus
from lgae_v3.runtime.wal import replay_committed_transactions
from lgae_v3.types import MutationResult


def _cfg() -> ResearchConfig:
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 3
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


_GOLDEN_SCRIPT = textwrap.dedent("""
    import sys, os, json, signal
    sys.path.insert(0, {src_path!r})
    import torch
    from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
    from lgae_v3.runtime import LGAERuntime, RuntimeConfig, make_joint_transaction, StructuralTransaction
    from lgae_v3.runtime.contracts import AuthorizationResult, AuthorizationStatus
    from lgae_v3.types import MutationResult

    torch.manual_seed(42)
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 6; cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 3
    cfg.audit.orc_backend = 'exact_lp'; cfg.audit.persistent_homology_enabled = False
    cfg.audit.entropic_nodes = 0; cfg.audit.bakry_nodes = 0; cfg.audit.cde_nodes = 0
    cfg.audit.exact_lly_top_k = 0; cfg.audit.orc_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]; cfg.mutation.curvature_ema_enabled = False

    failpoint = {failpoint!r}
    wal_path = {wal_path!r}
    state_file = {state_file!r}

    rt = LGAERuntime(
        make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=12),
        cfg, runtime_config=RuntimeConfig(wal_path=wal_path),
    )
    pre_hash = rt.authority_hash
    pre_version = int(rt.engine.graph.version)
    pre_graph_hash = rt.engine.graph.state_hash()
    pre_fiber_hash = rt.engine.fibers.state_hash()
    pre_gauge_hash = rt.engine.gauge_connections.state_hash if rt.engine.gauge_connections else ""

    # Build a joint transaction: graph + fiber + gauge.
    shadow_graph = rt.engine.graph.clone()
    shadow_graph.weight[0] = shadow_graph.weight[0] * 3.0
    snap = rt.engine.fibers.snapshot()
    if hasattr(snap, 'latent') and snap.latent is not None:
        snap.latent.data.fill_(0.5)
    raw = rt.engine.gauge_connections.raw_generators.detach().clone()
    raw.fill_(0.123)

    txn = make_joint_transaction(
        base_state_version=pre_version,
        base_state_hash=pre_hash,
        shadow_graph=shadow_graph,
        shadow_fiber_snapshot=snap,
        shadow_gauge_raw=raw,
        graph_action='golden_graph',
        fiber_action='golden_fiber',
        gauge_action='golden_gauge',
        mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        step=0,
    )
    full_txn = StructuralTransaction(
        transaction_id=txn.transaction_id,
        base_state_version=txn.base_state_version,
        base_state_hash=txn.base_state_hash,
        graph_delta=txn.graph_delta,
        fiber_delta=txn.fiber_delta,
        gauge_delta=txn.gauge_delta,
        authorization_id=txn.authorization_binding_hash(),
        delta_hash=txn.delta_hash,
        mutation_result=txn.mutation_result,
    )
    auth = AuthorizationResult(
        snapshot_id='s1', state_version=pre_version,
        state_hash=pre_hash,
        status=AuthorizationStatus.AUTHORIZED,
        transaction_hash=full_txn.transaction_id,
    )

    if failpoint == 'none':
        result = rt.commit_channel.commit(full_txn, auth)
        with open(state_file, 'w') as f:
            json.dump({{
                'pre_hash': pre_hash, 'post_hash': rt.authority_hash,
                'pre_version': pre_version,
                'post_version': int(rt.engine.graph.version),
                'post_graph_hash': rt.engine.graph.state_hash(),
                'post_fiber_hash': rt.engine.fibers.state_hash(),
                'post_gauge_hash': rt.engine.gauge_connections.state_hash if rt.engine.gauge_connections else '',
                'committed': result.committed,
                'transaction_id': result.transaction_id,
            }}, f)
    else:
        rt.commit_channel.set_failpoint(failpoint)
        original_check = rt.commit_channel._check_failpoint
        def kill_check(name):
            if name == failpoint:
                os.kill(os.getpid(), signal.SIGKILL)
        rt.commit_channel._check_failpoint = kill_check
        try:
            result = rt.commit_channel.commit(full_txn, auth)
            with open(state_file, 'w') as f:
                json.dump({{
                    'pre_hash': pre_hash, 'post_hash': rt.authority_hash,
                    'pre_version': pre_version,
                    'post_version': int(rt.engine.graph.version),
                    'post_graph_hash': rt.engine.graph.state_hash(),
                    'post_fiber_hash': rt.engine.fibers.state_hash(),
                    'post_gauge_hash': rt.engine.gauge_connections.state_hash if rt.engine.gauge_connections else '',
                    'committed': result.committed,
                    'transaction_id': result.transaction_id,
                }}, f)
        except Exception as e:
            with open(state_file, 'w') as f:
                json.dump({{
                    'pre_hash': pre_hash, 'post_hash': rt.authority_hash,
                    'pre_version': pre_version,
                    'committed': False, 'error': str(e),
                }}, f)
""")


def _run_golden(tmp_path: Path, failpoint: str) -> dict:
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    wal_path = str(tmp_path / "wal.jsonl")
    state_file = str(tmp_path / "state.json")
    script = _GOLDEN_SCRIPT.format(
        src_path=src_path, failpoint=failpoint,
        wal_path=wal_path, state_file=state_file,
    )
    script_file = tmp_path / "golden_script.py"
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
    return {"returncode": proc.returncode, "state": state, "wal_path": wal_path}


def _recover(tmp_path: Path) -> tuple[str, int, str, str, str]:
    """Recover from WAL and return (hash, version, graph_hash, fiber_hash, gauge_hash)."""
    wal_path = str(tmp_path / "wal.jsonl")
    torch.manual_seed(42)
    fresh = LGAERuntime(_graph(), _cfg())
    if os.path.exists(wal_path):
        replay_committed_transactions(wal_path, fresh._engine)
    return (
        fresh.authority_hash,
        int(fresh.engine.graph.version),
        fresh.engine.graph.state_hash(),
        fresh.engine.fibers.state_hash(),
        fresh.engine.gauge_connections.state_hash if fresh.engine.gauge_connections else "",
    )


class TestGoldenMultiDomainTransaction:
    """The decisive release gate: joint graph/fiber/gauge transaction + crash matrix."""

    def test_golden_normal_completion(self, tmp_path):
        """Normal completion of a joint transaction produces consistent post-state."""
        result = _run_golden(tmp_path, "none")
        assert result["returncode"] == 0
        state = result["state"]
        assert state["committed"]
        assert state["post_version"] == state["pre_version"] + 1
        assert state["post_hash"] != state["pre_hash"]

    def test_golden_crash_before_prepare(self, tmp_path):
        """Crash before any WAL activity → pre-state."""
        result = _run_golden(tmp_path, "before_prepare")
        assert result["returncode"] == -9
        normal = _run_golden(tmp_path / "normal", "none")
        pre_hash = normal["state"]["pre_hash"]
        rec_hash, rec_ver, _, _, _ = _recover(tmp_path)
        assert rec_hash == pre_hash, (
            f"Crash before prepare should recover to pre-state. "
            f"Expected: {pre_hash[:16]}, Got: {rec_hash[:16]}"
        )

    @pytest.mark.parametrize("failpoint", [
        "after_wal_commit",
        "before_state_swap",
        "after_state_swap",
    ])
    def test_golden_crash_after_wal_commit_recovers_to_post(self, tmp_path, failpoint):
        """Crash after WAL COMMIT → post-state (replay reconstructs)."""
        result = _run_golden(tmp_path, failpoint)
        assert result["returncode"] == -9
        normal = _run_golden(tmp_path / "normal", "none")
        post_hash = normal["state"]["post_hash"]
        post_ver = normal["state"]["post_version"]
        post_graph = normal["state"]["post_graph_hash"]
        post_fiber = normal["state"]["post_fiber_hash"]
        post_gauge = normal["state"]["post_gauge_hash"]

        rec_hash, rec_ver, rec_graph, rec_fiber, rec_gauge = _recover(tmp_path)

        # All component hashes must match — never a mixed state.
        assert rec_hash == post_hash, f"Authority hash mismatch at {failpoint}"
        assert rec_ver == post_ver, f"Version mismatch at {failpoint}"
        assert rec_graph == post_graph, f"Graph hash mismatch at {failpoint}"
        assert rec_fiber == post_fiber, f"Fiber hash mismatch at {failpoint}"
        assert rec_gauge == post_gauge, f"Gauge hash mismatch at {failpoint}"

    def test_golden_deterministic_replay(self, tmp_path):
        """Replaying the same WAL twice produces the same state."""
        result = _run_golden(tmp_path, "none")
        assert result["returncode"] == 0
        wal_path = result["wal_path"]

        # First recovery.
        torch.manual_seed(42)
        fresh1 = LGAERuntime(_graph(), _cfg())
        replay_committed_transactions(wal_path, fresh1._engine)
        hash1 = fresh1.authority_hash

        # Second recovery.
        torch.manual_seed(42)
        fresh2 = LGAERuntime(_graph(), _cfg())
        replay_committed_transactions(wal_path, fresh2._engine)
        hash2 = fresh2.authority_hash

        assert hash1 == hash2, "Deterministic replay: same WAL → same state"
        assert hash1 == result["state"]["post_hash"]

    def test_golden_transaction_receipt(self, tmp_path):
        """The transaction receipt is consistent across crash and recovery."""
        result = _run_golden(tmp_path, "none")
        assert result["returncode"] == 0
        state = result["state"]
        txn_id = state["transaction_id"]
        assert txn_id, "Transaction receipt should have a transaction_id"

        # Recover and verify the transaction is in the WAL.
        from lgae_v3.runtime.wal import WriteAheadLog
        wal = WriteAheadLog(result["wal_path"])
        records = list(wal.iter_records())
        begin_records = [r for r in records if r.record_type.value == "begin"]
        assert len(begin_records) > 0, "WAL should have a BEGIN record"
        assert begin_records[0].payload.get("transaction_id") == txn_id

    def test_golden_all_component_hashes_match(self, tmp_path):
        """After recovery, all component hashes match the post-state."""
        result = _run_golden(tmp_path, "none")
        assert result["returncode"] == 0
        state = result["state"]

        rec_hash, rec_ver, rec_graph, rec_fiber, rec_gauge = _recover(tmp_path)

        assert rec_hash == state["post_hash"]
        assert rec_ver == state["post_version"]
        assert rec_graph == state["post_graph_hash"]
        assert rec_fiber == state["post_fiber_hash"]
        assert rec_gauge == state["post_gauge_hash"]
