"""Phase 37 tests: Deterministic Replay Across PYTHONHASHSEED=0, 1, 2, 42, 123456."""
from __future__ import annotations

import os
import subprocess
import sys
import pytest

pytestmark = [pytest.mark.crash_recovery]
from pathlib import Path

SEEDS = [0, 1, 2, 42, 123456]

_REPLAY_SCRIPT = """
import sys, os, json, hashlib
sys.path.insert(0, {src_path!r})
import torch
from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime, RuntimeConfig

torch.manual_seed(42)
cfg = ResearchConfig()
cfg.fiber.d_base = 2; cfg.fiber.d_max = 6; cfg.fiber.spawn_width = 1; cfg.fiber.gauge_dim = 0
cfg.audit.orc_backend = "exact_lp"; cfg.audit.persistent_homology_enabled = False
cfg.audit.entropic_nodes = 0; cfg.audit.bakry_nodes = 0; cfg.audit.cde_nodes = 0
cfg.audit.exact_lly_top_k = 0; cfg.audit.orc_top_k = 0
cfg.mutation.shadow_horizons = [1, 2]; cfg.mutation.curvature_ema_enabled = False

graph = make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=12)
rt = LGAERuntime(graph, cfg, runtime_config=RuntimeConfig())

steps = []
for _ in range(3):
    res = rt.step()
    candidate_ids = [c.candidate_id for c in res.candidates.candidates] if res.candidates else []
    steps.append({{
        "step": res.step,
        "chosen_action": res.chosen_action,
        "candidate_ids": candidate_ids,
        "governance_decision": res.governance_decision,
        "committed": res.committed,
        "authority_hash_after": res.snapshot_after.authority_hash,
    }})

digest = json.dumps(steps, sort_keys=True, separators=(",", ":"))
print(hashlib.sha256(digest.encode()).hexdigest())
"""


@pytest.mark.parametrize("seed", SEEDS)
def test_cross_seed_identical_execution(seed):
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    script = _REPLAY_SCRIPT.format(src_path=src_path)
    
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert res.returncode == 0, f"Failed under PYTHONHASHSEED={seed}: {res.stderr}"
    hash_val = res.stdout.strip()
    assert len(hash_val) == 64
    
    # Store reference or compare
    ref_env = os.environ.copy()
    ref_env["PYTHONHASHSEED"] = "0"
    ref_res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=ref_env,
        timeout=30,
    )
    assert hash_val == ref_res.stdout.strip(), (
        f"Hash mismatch under PYTHONHASHSEED={seed} vs 0: {hash_val} != {ref_res.stdout.strip()}"
    )
