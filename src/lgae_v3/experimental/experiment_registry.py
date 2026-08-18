"""Experiment registry for v6 experiments.

Tracks every experiment with full provenance: configuration, code hash,
dataset hash, results, and status. This ensures experiments are
reproducible and comparable.

Each experiment record contains:
- Unique experiment ID.
- Name and description.
- Configuration (reproducibility config, hyperparameters).
- Code hash (from the manifest).
- Dataset hash (if using a structural dataset).
- Results (metrics, benchmark summaries).
- Status (running, completed, failed, superseded).
- Timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SUPERSEDED = "superseded"


@dataclass(slots=True)
class ExperimentRecord:
    """A single experiment record."""
    experiment_id: str
    name: str
    description: str
    config: dict[str, Any]
    code_hash: str = ""
    dataset_hash: str = ""
    results: dict[str, Any] = field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.PENDING
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    error: str = ""
    tags: list[str] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "code_hash": self.code_hash,
            "dataset_hash": self.dataset_hash,
            "results": self.results,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "tags": list(self.tags),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_log(), sort_keys=True, indent=2)


class ExperimentRegistry:
    """Registry of v6 experiments.

    Experiments can be stored in-memory or persisted to a JSON file.

    Usage::

        registry = ExperimentRegistry()
        exp = registry.create(
            name="v6.0-exp1-benchmark-baseline-competition",
            description="Run all baselines on frozen graph families",
            config={"n_steps": 5, "seed": 42},
        )
        registry.start(exp.experiment_id)
        # ... run experiment ...
        registry.complete(exp.experiment_id, results={...})
        registry.save("experiments.json")
    """

    def __init__(self) -> None:
        self._experiments: dict[str, ExperimentRecord] = {}

    def create(
        self,
        name: str,
        description: str,
        config: dict[str, Any],
        *,
        code_hash: str = "",
        dataset_hash: str = "",
        tags: list[str] | None = None,
    ) -> ExperimentRecord:
        """Create a new experiment record."""
        # Deterministic experiment ID.
        content = f"{name}:{description}:{json.dumps(config, sort_keys=True)}"
        eid = hashlib.sha256(content.encode()).hexdigest()[:16]
        record = ExperimentRecord(
            experiment_id=eid,
            name=name,
            description=description,
            config=dict(config),
            code_hash=code_hash,
            dataset_hash=dataset_hash,
            tags=list(tags) if tags else [],
            created_at=_utc_now(),
        )
        self._experiments[eid] = record
        return record

    def start(self, experiment_id: str) -> None:
        """Mark an experiment as running."""
        exp = self._experiments[experiment_id]
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = _utc_now()

    def complete(
        self,
        experiment_id: str,
        results: dict[str, Any],
    ) -> None:
        """Mark an experiment as completed with results."""
        exp = self._experiments[experiment_id]
        exp.status = ExperimentStatus.COMPLETED
        exp.results = dict(results)
        exp.completed_at = _utc_now()

    def fail(self, experiment_id: str, error: str) -> None:
        """Mark an experiment as failed."""
        exp = self._experiments[experiment_id]
        exp.status = ExperimentStatus.FAILED
        exp.error = error
        exp.completed_at = _utc_now()

    def supersede(self, experiment_id: str) -> None:
        """Mark an experiment as superseded."""
        exp = self._experiments[experiment_id]
        exp.status = ExperimentStatus.SUPERSEDED

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        return self._experiments.get(experiment_id)

    def all_experiments(self) -> list[ExperimentRecord]:
        return list(self._experiments.values())

    def by_status(self, status: ExperimentStatus) -> list[ExperimentRecord]:
        return [e for e in self._experiments.values() if e.status == status]

    def by_tag(self, tag: str) -> list[ExperimentRecord]:
        return [e for e in self._experiments.values() if tag in e.tags]

    def to_json(self) -> str:
        """Serialize all experiments to JSON."""
        return json.dumps(
            [e.to_log() for e in self._experiments.values()],
            sort_keys=True, indent=2,
        )

    def save(self, path: str | Path) -> None:
        """Save to a JSON file."""
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentRegistry":
        """Load from a JSON file."""
        registry = cls()
        data = json.loads(Path(path).read_text())
        for e_data in data:
            record = ExperimentRecord(
                experiment_id=e_data["experiment_id"],
                name=e_data["name"],
                description=e_data["description"],
                config=e_data["config"],
                code_hash=e_data.get("code_hash", ""),
                dataset_hash=e_data.get("dataset_hash", ""),
                results=e_data.get("results", {}),
                status=ExperimentStatus(e_data.get("status", "pending")),
                created_at=e_data.get("created_at", ""),
                started_at=e_data.get("started_at", ""),
                completed_at=e_data.get("completed_at", ""),
                error=e_data.get("error", ""),
                tags=e_data.get("tags", []),
            )
            registry._experiments[record.experiment_id] = record
        return registry

    def __len__(self) -> int:
        return len(self._experiments)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
