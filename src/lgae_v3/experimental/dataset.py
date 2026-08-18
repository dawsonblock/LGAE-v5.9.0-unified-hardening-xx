"""Structural dataset schema and serialization.

Defines the schema for datasets of structural transitions produced by the
transition recorder. Datasets are split into train/validation/held-out
following the frozen graph family splits.

Schema version: ``LGAE_STRUCTURAL_DATASET_V6_0_EXP1``

A dataset contains:
- Metadata: schema version, creation timestamp, source runtime version,
  graph family split, seed, transition count.
- Transitions: list of StructuralTransition records.
- Split assignment: each transition is tagged with its split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import hashlib
from pathlib import Path

from .transition_recorder import StructuralTransition
from .graph_families import GraphFamilySplit, FROZEN_SPLIT


DATASET_SCHEMA_VERSION = "LGAE_STRUCTURAL_DATASET_V6_0_EXP1"


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Metadata for a structural dataset."""
    schema_version: str
    created_at: str
    runtime_version: str
    split: str  # "train", "validation", "held_out", or "all"
    n_transitions: int
    n_committed: int
    n_rejected: int
    seed: int
    graph_family_split: dict[str, Any]
    description: str = ""

    def to_log(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "runtime_version": self.runtime_version,
            "split": self.split,
            "n_transitions": int(self.n_transitions),
            "n_committed": int(self.n_committed),
            "n_rejected": int(self.n_rejected),
            "seed": int(self.seed),
            "graph_family_split": self.graph_family_split,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """A split of the dataset."""
    name: str  # "train", "validation", "held_out"
    transition_ids: tuple[str, ...]

    def to_log(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_transitions": len(self.transition_ids),
            "transition_ids": list(self.transition_ids),
        }


class StructuralDataset:
    """A dataset of structural transitions.

    Datasets are immutable after construction. They can be serialized to
    JSON and deserialized back.

    The dataset is split-aware: transitions are tagged with their graph
    family split (train/validation/held_out).
    """

    def __init__(
        self,
        transitions: list[StructuralTransition],
        *,
        split: str = "all",
        seed: int = 42,
        description: str = "",
        graph_family_split: GraphFamilySplit | None = None,
        runtime_version: str = "",
        created_at: str = "",
    ) -> None:
        self._transitions = list(transitions)
        self._split = split
        self._seed = int(seed)
        self._description = description
        self._graph_family_split = graph_family_split or FROZEN_SPLIT
        self._runtime_version = runtime_version
        self._created_at = created_at or _utc_now()

    @property
    def transitions(self) -> list[StructuralTransition]:
        return list(self._transitions)

    @property
    def n_transitions(self) -> int:
        return len(self._transitions)

    @property
    def n_committed(self) -> int:
        return sum(1 for t in self._transitions if t.executed)

    @property
    def n_rejected(self) -> int:
        return sum(1 for t in self._transitions if not t.executed)

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            schema_version=DATASET_SCHEMA_VERSION,
            created_at=self._created_at,
            runtime_version=self._runtime_version,
            split=self._split,
            n_transitions=self.n_transitions,
            n_committed=self.n_committed,
            n_rejected=self.n_rejected,
            seed=self._seed,
            graph_family_split=self._graph_family_split.to_log(),
            description=self._description,
        )

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the dataset content (transitions + metadata)."""
        content = json.dumps(
            [t.to_log() for t in self._transitions],
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            "metadata": self.metadata.to_log(),
            "transitions": [t.to_log() for t in self._transitions],
            "content_hash": self.content_hash,
        }, sort_keys=True, indent=2)

    def save(self, path: str | Path) -> None:
        """Save to a JSON file."""
        Path(path).write_text(self.to_json())

    @classmethod
    def from_json(cls, json_str: str) -> "StructuralDataset":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        meta = data["metadata"]
        transitions = []
        for t_data in data["transitions"]:
            transitions.append(StructuralTransition(
                transition_id=t_data["transition_id"],
                step=t_data["step"],
                seed=t_data["seed"],
                state_before_hash=t_data["state_before_hash"],
                state_before_version=t_data["state_before_version"],
                state_before_summary=t_data["state_before_summary"],
                chosen_action=t_data["chosen_action"],
                governance_decision=t_data["governance_decision"],
                executed=t_data["executed"],
                action_metadata=t_data["action_metadata"],
                state_after_hash=t_data["state_after_hash"],
                state_after_version=t_data["state_after_version"],
                state_after_summary=t_data["state_after_summary"],
                utility_before=t_data["utility_before"],
                utility_after=t_data["utility_after"],
                delta_utility=t_data["delta_utility"],
                compute_cost=t_data["compute_cost"],
                reward=t_data["reward"],
                runtime_version=t_data["runtime_version"],
                timestamp=t_data["timestamp"],
            ))
        return cls(
            transitions=transitions,
            split=meta["split"],
            seed=meta["seed"],
            description=meta.get("description", ""),
            runtime_version=meta["runtime_version"],
            created_at=meta["created_at"],
        )

    @classmethod
    def load(cls, path: str | Path) -> "StructuralDataset":
        """Load from a JSON file."""
        return cls.from_json(Path(path).read_text())

    def to_log(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_log(),
            "content_hash": self.content_hash,
        }


class StructuralDatasetSchema:
    """Schema validation for structural datasets."""

    SCHEMA_VERSION = DATASET_SCHEMA_VERSION

    @staticmethod
    def validate(data: dict[str, Any]) -> bool:
        """Validate that a dictionary conforms to the dataset schema."""
        required_keys = {"metadata", "transitions", "content_hash"}
        if not required_keys.issubset(data.keys()):
            return False
        meta = data["metadata"]
        if meta.get("schema_version") != DATASET_SCHEMA_VERSION:
            return False
        if not isinstance(data["transitions"], list):
            return False
        # Validate each transition has required fields.
        required_t_fields = {
            "transition_id", "step", "seed",
            "state_before_hash", "state_after_hash",
            "chosen_action", "governance_decision", "executed",
            "utility_before", "utility_after", "delta_utility",
            "compute_cost", "reward",
        }
        for t in data["transitions"]:
            if not required_t_fields.issubset(t.keys()):
                return False
        return True


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
