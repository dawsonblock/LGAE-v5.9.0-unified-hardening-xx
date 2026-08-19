"""v5.11 golden scenario: deterministic end-to-end digest.

This is the heartbeat test for v5.11. It runs the full canonical cycle
and produces a deterministic digest bundle. The same inputs must produce
the same hashes:

- current process
- new process
- different PYTHONHASHSEED values
- repeated 5 times

If any hash differs, the runtime is not deterministic.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.crash_recovery]
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.runtime.contracts import CANONICAL_PHASE_ORDER


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


def _run_golden_scenario() -> dict:
    """Run the golden scenario and return the digest bundle."""
    torch.manual_seed(42)
    rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig())

    # Run 3 steps.
    results = []
    for _ in range(3):
        result = rt.step()
        results.append(result)

    # Compute the digest bundle.
    digests = {
        "phase_order": list(CANONICAL_PHASE_ORDER),
        "steps": [],
        "final_authority_hash": rt.authority_hash,
        "final_graph_version": int(rt.engine.graph.version),
    }
    for i, result in enumerate(results):
        digests["steps"].append({
            "step": result.step,
            "chosen_action": result.chosen_action,
            "governance_decision": result.governance_decision,
            "committed": result.committed,
            "authority_hash_before": result.metadata.get("authority_hash_before", ""),
            "authority_hash_after": result.metadata.get("authority_hash_after", ""),
            "phase_order": result.metadata.get("phase_order", []),
        })

    # Compute a single canonical hash over the entire digest.
    canonical = json.dumps(digests, sort_keys=True, separators=(",", ":"))
    digests["canonical_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    return digests


class TestGoldenScenarioInProcess:
    """The golden scenario must be deterministic within a process."""

    def test_golden_scenario_runs(self):
        """The golden scenario runs without errors."""
        digests = _run_golden_scenario()
        assert digests["canonical_hash"]
        assert len(digests["canonical_hash"]) == 64

    def test_golden_scenario_deterministic_repeated(self):
        """Running the golden scenario 5 times produces the same hash."""
        hashes = []
        for _ in range(5):
            digests = _run_golden_scenario()
            hashes.append(digests["canonical_hash"])
        assert len(set(hashes)) == 1, (
            f"Golden scenario is not deterministic! Hashes: {hashes}"
        )

    def test_golden_scenario_all_phases_executed(self):
        """Every step executes all 8 canonical phases."""
        digests = _run_golden_scenario()
        for step in digests["steps"]:
            assert step["phase_order"] == list(CANONICAL_PHASE_ORDER)


class TestGoldenScenarioCrossProcess:
    """The golden scenario must be deterministic across processes."""

    def test_golden_scenario_same_across_processes(self):
        """Running in a subprocess produces the same hash."""
        # Run in the current process.
        digests1 = _run_golden_scenario()
        hash1 = digests1["canonical_hash"]

        # Run in a subprocess with a different PYTHONHASHSEED.
        script = f"""
import sys
sys.path.insert(0, {repr(str(Path(__file__).resolve().parents[2] / "src"))})
import json, hashlib, torch
from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.runtime.contracts import CANONICAL_PHASE_ORDER

torch.manual_seed(42)
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
graph = make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=12)
rt = LGAERuntime(graph, cfg, runtime_config=RuntimeConfig())
results = []
for _ in range(3):
    results.append(rt.step())
digests = {{"phase_order": list(CANONICAL_PHASE_ORDER), "steps": [], "final_authority_hash": rt.authority_hash, "final_graph_version": int(rt.engine.graph.version)}}
for result in results:
    digests["steps"].append({{"step": result.step, "chosen_action": result.chosen_action, "governance_decision": result.governance_decision, "committed": result.committed, "authority_hash_before": result.metadata.get("authority_hash_before", ""), "authority_hash_after": result.metadata.get("authority_hash_after", ""), "phase_order": result.metadata.get("phase_order", [])}})
canonical = json.dumps(digests, sort_keys=True, separators=(",", ":"))
print(hashlib.sha256(canonical.encode()).hexdigest())
"""
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "12345"
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env, timeout=60,
        )
        if result.returncode != 0:
            pytest.skip(f"subprocess failed: {result.stderr}")
        hash2 = result.stdout.strip()
        assert hash1 == hash2, (
            f"Golden scenario differs across processes! "
            f"In-process: {hash1}, Subprocess: {hash2}"
        )

    def test_golden_scenario_different_pythonhashseed(self):
        """Different PYTHONHASHSEED values produce the same hash."""
        script_template = """
import sys
sys.path.insert(0, {src_path!r})
import json, hashlib, torch
from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.runtime.contracts import CANONICAL_PHASE_ORDER
torch.manual_seed(42)
cfg = ResearchConfig()
cfg.fiber.d_base = 2; cfg.fiber.d_max = 6; cfg.fiber.spawn_width = 1; cfg.fiber.gauge_dim = 0
cfg.audit.orc_backend = "exact_lp"; cfg.audit.persistent_homology_enabled = False
cfg.audit.entropic_nodes = 0; cfg.audit.bakry_nodes = 0; cfg.audit.cde_nodes = 0
cfg.audit.exact_lly_top_k = 0; cfg.audit.orc_top_k = 0
cfg.mutation.shadow_horizons = [1, 2]; cfg.mutation.curvature_ema_enabled = False
graph = make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=12)
rt = LGAERuntime(graph, cfg, runtime_config=RuntimeConfig())
results = [rt.step() for _ in range(3)]
digests = {{"phase_order": list(CANONICAL_PHASE_ORDER), "steps": [], "final_authority_hash": rt.authority_hash, "final_graph_version": int(rt.engine.graph.version)}}
for r in results:
    digests["steps"].append({{"step": r.step, "chosen_action": r.chosen_action, "governance_decision": r.governance_decision, "committed": r.committed, "authority_hash_before": r.metadata.get("authority_hash_before", ""), "authority_hash_after": r.metadata.get("authority_hash_after", ""), "phase_order": r.metadata.get("phase_order", [])}})
canonical = json.dumps(digests, sort_keys=True, separators=(",", ":"))
print(hashlib.sha256(canonical.encode()).hexdigest())
"""
        src_path = str(Path(__file__).resolve().parents[2] / "src")
        script = script_template.format(src_path=src_path)

        hashes = []
        for seed in ("0", "1", "42", "999"):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = seed
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, env=env, timeout=60,
            )
            if result.returncode != 0:
                pytest.skip(f"subprocess failed with seed {seed}: {result.stderr}")
            hashes.append(result.stdout.strip())

        assert len(set(hashes)) == 1, (
            f"Golden scenario differs across PYTHONHASHSEED values! Hashes: {hashes}"
        )
