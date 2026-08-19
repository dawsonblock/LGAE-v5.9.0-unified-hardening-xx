"""Dataset freeze for the exp4.2 study.

The exp2 dataset becomes immutable for this study. This module:
- Records split hashes, record counts, provenance distributions
- Persists freeze manifests to disk
- Provides a load function to verify freeze integrity

The held-out partition must not influence normalization, feature
selection, encoder selection, hyperparameters, model selection,
threshold selection, or architecture selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import json
import hashlib
import time


@dataclass(frozen=True, slots=True)
class SplitFreeze:
    """Immutable freeze record for one dataset split."""
    split: str
    content_hash: str
    n_records: int
    n_realized: int
    n_counterfactual: int
    n_shadow: int
    mutation_distribution: dict[str, int] = field(default_factory=dict)
    graph_family_distribution: dict[str, int] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "content_hash": self.content_hash,
            "n_records": int(self.n_records),
            "n_realized": int(self.n_realized),
            "n_counterfactual": int(self.n_counterfactual),
            "n_shadow": int(self.n_shadow),
            "mutation_distribution": dict(self.mutation_distribution),
            "graph_family_distribution": dict(self.graph_family_distribution),
        }


@dataclass
class DatasetFreeze:
    """Full dataset freeze record for exp4.2."""
    dataset_schema_hash: str
    train: SplitFreeze
    validation: SplitFreeze
    heldout: SplitFreeze
    feature_schema_hash: str
    graph_family_registry_hash: str
    seed: int
    frozen_at: str = ""
    freeze_hash: str = ""

    def __post_init__(self) -> None:
        if not self.frozen_at:
            self.frozen_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not self.freeze_hash:
            self.freeze_hash = self._compute_freeze_hash()

    def _compute_freeze_hash(self) -> str:
        content = json.dumps({
            "dataset_schema_hash": self.dataset_schema_hash,
            "train": self.train.to_log(),
            "validation": self.validation.to_log(),
            "heldout": self.heldout.to_log(),
            "feature_schema_hash": self.feature_schema_hash,
            "graph_family_registry_hash": self.graph_family_registry_hash,
            "seed": int(self.seed),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def train_split_hash(self) -> str:
        return self.train.content_hash

    @property
    def validation_split_hash(self) -> str:
        return self.validation.content_hash

    @property
    def heldout_split_hash(self) -> str:
        return self.heldout.content_hash

    def to_log(self) -> dict[str, Any]:
        return {
            "dataset_schema_hash": self.dataset_schema_hash,
            "train": self.train.to_log(),
            "validation": self.validation.to_log(),
            "heldout": self.heldout.to_log(),
            "feature_schema_hash": self.feature_schema_hash,
            "graph_family_registry_hash": self.graph_family_registry_hash,
            "seed": int(self.seed),
            "frozen_at": self.frozen_at,
            "freeze_hash": self.freeze_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_log(), sort_keys=True, indent=2)

    def save(self, directory: str | Path) -> None:
        """Save freeze manifests to a directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        # Full freeze.
        (dir_path / "DATASET_FREEZE.json").write_text(self.to_json())
        # Per-split manifests.
        for split_name, split in [("train", self.train), ("validation", self.validation), ("heldout", self.heldout)]:
            (dir_path / f"{split_name}.manifest.json").write_text(
                json.dumps(split.to_log(), sort_keys=True, indent=2)
            )
        # README.
        (dir_path / "DATASET_FREEZE.md").write_text(
            f"# Dataset Freeze — exp4.2\n\n"
            f"Freeze hash: `{self.freeze_hash}`\n"
            f"Frozen at: {self.frozen_at}\n\n"
            f"| Split | Records | Realized | CF | Shadow | Hash |\n"
            f"|-------|---------|----------|----|--------|------|\n"
            f"| train | {self.train.n_records} | {self.train.n_realized} | "
            f"{self.train.n_counterfactual} | {self.train.n_shadow} | "
            f"`{self.train.content_hash[:16]}` |\n"
            f"| validation | {self.validation.n_records} | {self.validation.n_realized} | "
            f"{self.validation.n_counterfactual} | {self.validation.n_shadow} | "
            f"`{self.validation.content_hash[:16]}` |\n"
            f"| heldout | {self.heldout.n_records} | {self.heldout.n_realized} | "
            f"{self.heldout.n_counterfactual} | {self.heldout.n_shadow} | "
            f"`{self.heldout.content_hash[:16]}` |\n\n"
            f"**Held-out must not influence any selection decision.**\n"
        )


def _provenance_value(r: Any) -> str:
    p = getattr(r, "provenance", None)
    if p is None:
        return "unknown"
    if hasattr(p, "value"):
        return str(p.value).lower()
    return str(p).lower()


def freeze_dataset(
    datasets: dict[str, Any],
    *,
    dataset_schema_hash: str,
    feature_schema_hash: str,
    graph_family_registry_hash: str,
    seed: int,
) -> DatasetFreeze:
    """Create a DatasetFreeze from generated split datasets.

    Args:
        datasets: Dict mapping split name to SplitDataset (from exp2 generator).
        dataset_schema_hash: Hash of the dataset schema.
        feature_schema_hash: Hash of the feature schema.
        graph_family_registry_hash: Hash of the graph family registry.
        seed: The generation seed.

    Returns:
        DatasetFreeze with all split metadata.
    """
    def _make_split_freeze(split_name: str, ds: Any) -> SplitFreeze:
        records = ds.records
        n = len(records)
        prov_counts: dict[str, int] = {}
        mut_counts: dict[str, int] = {}
        gf_counts: dict[str, int] = {}
        for r in records:
            p = _provenance_value(r)
            prov_counts[p] = prov_counts.get(p, 0) + 1
            action = getattr(r, "action", "unknown")
            mut_counts[action] = mut_counts.get(action, 0) + 1
            gf = getattr(r, "graph_family", "unknown")
            gf_counts[gf] = gf_counts.get(gf, 0) + 1
        return SplitFreeze(
            split=split_name,
            content_hash=ds.content_hash,
            n_records=n,
            n_realized=prov_counts.get("realized", 0),
            n_counterfactual=prov_counts.get("counterfactual", 0),
            n_shadow=prov_counts.get("shadow", 0),
            mutation_distribution=mut_counts,
            graph_family_distribution=gf_counts,
        )

    return DatasetFreeze(
        dataset_schema_hash=dataset_schema_hash,
        train=_make_split_freeze("train", datasets["train"]),
        validation=_make_split_freeze("validation", datasets["validation"]),
        heldout=_make_split_freeze("held_out", datasets["held_out"]),
        feature_schema_hash=feature_schema_hash,
        graph_family_registry_hash=graph_family_registry_hash,
        seed=seed,
    )


def load_dataset_freeze(directory: str | Path) -> DatasetFreeze:
    """Load a DatasetFreeze from a directory."""
    dir_path = Path(directory)
    data = json.loads((dir_path / "DATASET_FREEZE.json").read_text())
    return DatasetFreeze(
        dataset_schema_hash=data["dataset_schema_hash"],
        train=SplitFreeze(**data["train"]),
        validation=SplitFreeze(**data["validation"]),
        heldout=SplitFreeze(**data["heldout"]),
        feature_schema_hash=data["feature_schema_hash"],
        graph_family_registry_hash=data["graph_family_registry_hash"],
        seed=data["seed"],
        frozen_at=data.get("frozen_at", ""),
        freeze_hash=data.get("freeze_hash", ""),
    )
