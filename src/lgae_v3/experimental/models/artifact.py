"""Model artifact with provenance binding.

Every trained model artifact binds to:
- model type
- model version
- encoder ID
- encoder schema hash
- dataset schema hash
- train split hash
- normalization hash
- hyperparameter hash
- seed
- training code version

No model should be loadable against an incompatible encoder or dataset
without an explicit compatibility failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json
import time


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A trained model artifact with full provenance.

    This is the canonical artifact format for v6.0-exp4. It captures
    everything needed to reproduce a model and verify compatibility.
    """
    model_id: str
    predictor_type: str
    predictor_version: str
    encoder_id: str
    encoder_schema_hash: str
    dataset_schema_hash: str
    train_split_hash: str
    normalization_hash: str
    hyperparameter_hash: str
    seed: int
    training_code_version: str
    n_train_samples: int
    n_features: int
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    description: str = ""

    @property
    def artifact_hash(self) -> str:
        """Deterministic hash of the artifact."""
        content = json.dumps({
            "model_id": self.model_id,
            "predictor_type": self.predictor_type,
            "predictor_version": self.predictor_version,
            "encoder_id": self.encoder_id,
            "encoder_schema_hash": self.encoder_schema_hash,
            "dataset_schema_hash": self.dataset_schema_hash,
            "train_split_hash": self.train_split_hash,
            "normalization_hash": self.normalization_hash,
            "hyperparameter_hash": self.hyperparameter_hash,
            "seed": int(self.seed),
            "training_code_version": self.training_code_version,
            "n_train_samples": int(self.n_train_samples),
            "n_features": int(self.n_features),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def is_compatible_with(
        self,
        encoder_schema_hash: str,
        dataset_schema_hash: str,
    ) -> bool:
        """Check if this artifact is compatible with a given encoder and dataset."""
        return (
            self.encoder_schema_hash == encoder_schema_hash
            and self.dataset_schema_hash == dataset_schema_hash
        )

    def to_log(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "predictor_type": self.predictor_type,
            "predictor_version": self.predictor_version,
            "encoder_id": self.encoder_id,
            "encoder_schema_hash": self.encoder_schema_hash,
            "dataset_schema_hash": self.dataset_schema_hash,
            "train_split_hash": self.train_split_hash,
            "normalization_hash": self.normalization_hash,
            "hyperparameter_hash": self.hyperparameter_hash,
            "seed": int(self.seed),
            "training_code_version": self.training_code_version,
            "n_train_samples": int(self.n_train_samples),
            "n_features": int(self.n_features),
            "metrics": dict(self.metrics),
            "created_at": self.created_at,
            "description": self.description,
            "artifact_hash": self.artifact_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_log(), sort_keys=True, indent=2)


class CompatibilityError(Exception):
    """Raised when a model is loaded against an incompatible encoder or dataset."""


def create_artifact(
    model: Any,
    *,
    encoder_id: str,
    encoder_schema_hash: str,
    dataset_schema_hash: str,
    train_split_hash: str = "",
    normalization_hash: str = "",
    training_code_version: str = "v6.0-exp4",
    metrics: dict[str, Any] | None = None,
    description: str = "",
) -> ModelArtifact:
    """Create a ModelArtifact from a fitted model."""
    # Extract hyperparameter hash from model config.
    hp = {
        "model_type": getattr(model, "model_type", "unknown"),
        "version": getattr(model, "version", "unknown"),
        "seed": getattr(model, "seed", 42),
    }
    hp_content = json.dumps(hp, sort_keys=True)
    hp_hash = hashlib.sha256(hp_content.encode()).hexdigest()[:16]

    return ModelArtifact(
        model_id=getattr(model, "model_id", "unknown"),
        predictor_type=getattr(model, "model_type", "unknown"),
        predictor_version=getattr(model, "version", "unknown"),
        encoder_id=encoder_id,
        encoder_schema_hash=encoder_schema_hash,
        dataset_schema_hash=dataset_schema_hash,
        train_split_hash=train_split_hash,
        normalization_hash=normalization_hash,
        hyperparameter_hash=hp_hash,
        seed=getattr(model, "seed", 42),
        training_code_version=training_code_version,
        n_train_samples=getattr(model, "_n_samples", 0),
        n_features=getattr(model, "_n_features", 0) or getattr(model, "_in_dim", 0),
        metrics=metrics or {},
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        description=description,
    )
