"""Experiment snapshot for exp7.5.

An immutable JSON containing all configuration that could affect
the experiment outcome. Hashed to prevent configuration drift.

This is saved BEFORE the live run and referenced in the final report.
"""
from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime


@dataclass
class ExperimentSnapshot:
    """Immutable experiment configuration snapshot."""

    # Source
    source_commit: str = ""
    snapshot_timestamp: str = ""

    # Backend
    provider: str = ""
    model_id: str = ""
    backend_config_hash: str = ""

    # Prompts
    prompt_hashes: dict = field(default_factory=dict)

    # Data splits
    train_task_ids: list = field(default_factory=list)
    calibration_task_ids: list = field(default_factory=list)
    test_task_ids: list = field(default_factory=list)

    # Objective
    objective_weights: dict = field(default_factory=dict)

    # Routing
    routing_config: dict = field(default_factory=dict)

    # Seeds
    benchmark_seed: int = 42
    split_seed: int = 43

    # Budget
    budget_ceilings: dict = field(default_factory=dict)

    # Gates
    gate_definitions: list = field(default_factory=list)

    # Snapshot hash (computed after all fields set)
    snapshot_hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA256 hash of the snapshot (excluding the hash field)."""
        data = self.to_dict()
        data.pop("snapshot_hash", None)
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def to_dict(self) -> dict:
        return {
            "source_commit": self.source_commit,
            "snapshot_timestamp": self.snapshot_timestamp,
            "provider": self.provider,
            "model_id": self.model_id,
            "backend_config_hash": self.backend_config_hash,
            "prompt_hashes": self.prompt_hashes,
            "train_task_ids": self.train_task_ids,
            "calibration_task_ids": self.calibration_task_ids,
            "test_task_ids": self.test_task_ids,
            "objective_weights": self.objective_weights,
            "routing_config": self.routing_config,
            "benchmark_seed": self.benchmark_seed,
            "split_seed": self.split_seed,
            "budget_ceilings": self.budget_ceilings,
            "gate_definitions": self.gate_definitions,
            "snapshot_hash": self.snapshot_hash,
        }

    def save(self, path: str) -> str:
        """Save snapshot to file and return the hash."""
        self.snapshot_hash = self.compute_hash()
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return self.snapshot_hash

    @classmethod
    def load(cls, path: str) -> "ExperimentSnapshot":
        """Load snapshot from file."""
        with open(path) as f:
            data = json.load(f)
        snap = cls()
        for key, val in data.items():
            if hasattr(snap, key):
                setattr(snap, key, val)
        return snap


def create_snapshot(
    *,
    source_commit: str,
    provider: str,
    model_id: str,
    backend_config_hash: str,
    prompt_hashes: dict,
    train_ids: list,
    calibration_ids: list,
    test_ids: list,
    objective_weights: dict,
    routing_config: dict,
    budget_ceilings: dict,
    gate_definitions: list,
    benchmark_seed: int = 42,
    split_seed: int = 43,
) -> ExperimentSnapshot:
    """Create an experiment snapshot."""
    snap = ExperimentSnapshot(
        source_commit=source_commit,
        snapshot_timestamp=datetime.utcnow().isoformat() + "Z",
        provider=provider,
        model_id=model_id,
        backend_config_hash=backend_config_hash,
        prompt_hashes=prompt_hashes,
        train_task_ids=train_ids,
        calibration_task_ids=calibration_ids,
        test_task_ids=test_ids,
        objective_weights=objective_weights,
        routing_config=routing_config,
        benchmark_seed=benchmark_seed,
        split_seed=split_seed,
        budget_ceilings=budget_ceilings,
        gate_definitions=gate_definitions,
    )
    snap.snapshot_hash = snap.compute_hash()
    return snap


# The 15 predeclared gate definitions (frozen).
GATE_DEFINITIONS = [
    {"id": "A", "name": "real_backend_executes_every_role", "criterion": "smoke test passes for all 6 roles"},
    {"id": "B", "name": "topology_changes_context_output", "criterion": "Var(Q_full - Q_minimal) > 0 on 20 tasks"},
    {"id": "C", "name": "identical_model_and_prompts", "criterion": "all conditions use same backend and prompt hashes"},
    {"id": "D", "name": "deterministic_graders", "criterion": "quality evaluators use task-specific deterministic grading"},
    {"id": "E", "name": "lgae_quality_gte_fixed_minus_tol", "criterion": "LGAE quality >= Fixed quality - 0.05"},
    {"id": "F", "name": "lgae_token_cost_lt_fixed", "criterion": "LGAE tokens < Fixed tokens"},
    {"id": "G", "name": "lgae_j_gt_fixed", "criterion": "LGAE J > Fixed J"},
    {"id": "H", "name": "lgae_quality_approx_dynamic", "criterion": "LGAE quality >= Dynamic quality - 0.02"},
    {"id": "I", "name": "lgae_tokens_lte_dynamic", "criterion": "LGAE tokens <= Dynamic tokens"},
    {"id": "J", "name": "nonzero_adaptive_routing", "criterion": "n_calibrations > 0"},
    {"id": "K", "name": "no_failure_regression", "criterion": "LGAE failures <= Fixed failures + 1.0"},
    {"id": "L", "name": "rollback_works", "criterion": "KNOWN_GOOD_TOPOLOGY preserved, rollback implemented"},
    {"id": "M", "name": "test_untouched", "criterion": "LGAE adapts on TRAIN/CALIBRATION only"},
    {"id": "N", "name": "authority_preserved", "criterion": "routing through NodeNecessityRouter + governance"},
    {"id": "O", "name": "release_qualification", "criterion": "full release qualification passes"},
]
