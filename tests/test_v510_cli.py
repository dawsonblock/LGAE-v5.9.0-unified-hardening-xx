"""v5.10 Phase 42: CLI tests."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest


def _run_cli(*args: str) -> tuple[int, str, str]:
    """Run the lgae-v3 CLI and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, "-m", "lgae_v3.cli", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return proc.returncode, proc.stdout, proc.stderr


def test_cli_inspect():
    rc, out, err = _run_cli("inspect", "--nodes", "6")
    assert rc == 0, f"stderr: {err}"
    data = json.loads(out)
    assert data["phase"] == "v5.10-canonical-runtime"
    assert data["graph"]["num_nodes"] == 6
    assert "authority_hash" in data
    assert "snapshot" in data
    assert "boundary" in data


def test_cli_diagnose():
    rc, out, err = _run_cli("diagnose", "--risk", "0.7", "--authority", "structural")
    assert rc == 0, f"stderr: {err}"
    data = json.loads(out)
    assert data["level_name"] == "L2_STRUCTURAL"
    assert data["authority"] == "structural"


def test_cli_diagnose_irreversible_forces_l3():
    rc, out, err = _run_cli("diagnose", "--authority", "irreversible")
    assert rc == 0, f"stderr: {err}"
    data = json.loads(out)
    assert data["level_name"] == "L3_EXACT"


def test_cli_propose():
    rc, out, err = _run_cli("propose", "--nodes", "6", "--top_k", "4")
    assert rc == 0, f"stderr: {err}"
    data = json.loads(out)
    assert data["candidate_count"] >= 1
    assert "candidates" in data
    assert "channel_counts" in data


def test_cli_step():
    rc, out, err = _run_cli("step", "--nodes", "6")
    assert rc == 0, f"stderr: {err}"
    data = json.loads(out)
    assert "step" in data
    assert "committed" in data
    assert "authority_hash_after" in data


def test_cli_run():
    rc, out, err = _run_cli("run", "--nodes", "6", "--steps", "2")
    assert rc == 0, f"stderr: {err}"
    data = json.loads(out)
    assert data["steps"] == 2
    assert len(data["results"]) == 2
    assert "final_authority_hash" in data


def test_cli_qualify():
    rc, out, err = _run_cli("qualify", "--nodes", "6")
    assert rc == 0, f"stderr: {err}"
    data = json.loads(out)
    assert "invariants" in data
    assert "all_blocking_passed" in data
    assert data["all_blocking_passed"] is True


def test_cli_legacy_demo_still_works():
    rc, out, err = _run_cli("demo", "--nodes", "6", "--steps", "1")
    assert rc == 0, f"stderr: {err}"
    data = json.loads(out)
    assert data["version"] is not None
    assert data["nodes"] == 6


def test_cli_legacy_qualify_lly_still_works():
    rc, out, err = _run_cli("qualify-lly", "--graph", "cycle", "--nodes", "4")
    assert rc == 0, f"stderr: {err}"
    data = json.loads(out)
    assert isinstance(data, dict)
