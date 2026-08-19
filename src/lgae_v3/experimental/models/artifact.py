"""Model artifact with full provenance binding and state serialization.

Every trained model artifact binds to:
- model type, version, complete hyperparameter configuration
- encoder ID, encoder schema hash
- dataset schema hash, train split hash, normalization hash
- feature schema hash, target schema hash
- seed, training code version
- actual model state (weights, stumps, ensemble parameters)

No model should be loadable against an incompatible encoder or dataset
without an explicit compatibility failure.

Fix 1: Serializes actual model state, not just metadata.
Fix 2: Hashes complete hyperparameter configuration.
Fix 3: Binds compatibility to dataset/split/normalization identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json
import time
import base64
import pickle


class CompatibilityError(Exception):
    """Raised when a model is loaded against an incompatible encoder or dataset."""


def _serialize_state(state: Any) -> str:
    """Serialize model state to a base64-encoded string."""
    raw = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(raw).decode("ascii")


def _deserialize_state(encoded: str) -> Any:
    """Deserialize model state from a base64-encoded string."""
    raw = base64.b64decode(encoded.encode("ascii"))
    return pickle.loads(raw)


def _state_hash(state: Any) -> str:
    """Compute a deterministic hash over model state."""
    raw = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A trained model artifact with full provenance and state.

    This is the canonical artifact format for v6.0-exp4.1. It captures
    everything needed to reproduce a model, verify compatibility, and
    reload the trained state.

    Fix 1: ``model_state`` serializes actual weights/stumps/ensemble params.
    Fix 2: ``hyperparameter_hash`` covers the complete training config.
    Fix 3: Compatibility checks include split/normalization/feature/target identity.
    """
    model_id: str
    predictor_type: str
    predictor_version: str
    encoder_id: str
    encoder_schema_hash: str
    dataset_schema_hash: str
    train_split_hash: str
    normalization_hash: str
    feature_schema_hash: str
    target_schema_hash: str
    hyperparameter_hash: str
    model_state_hash: str
    seed: int
    training_code_version: str
    n_train_samples: int
    n_features: int
    # Fix 1: Actual serialized model state.
    model_state: str = ""  # base64-encoded pickle
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    description: str = ""

    @property
    def artifact_hash(self) -> str:
        """Deterministic hash of the artifact (excluding volatile fields)."""
        content = json.dumps({
            "model_id": self.model_id,
            "predictor_type": self.predictor_type,
            "predictor_version": self.predictor_version,
            "encoder_id": self.encoder_id,
            "encoder_schema_hash": self.encoder_schema_hash,
            "dataset_schema_hash": self.dataset_schema_hash,
            "train_split_hash": self.train_split_hash,
            "normalization_hash": self.normalization_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "target_schema_hash": self.target_schema_hash,
            "hyperparameter_hash": self.hyperparameter_hash,
            "model_state_hash": self.model_state_hash,
            "seed": int(self.seed),
            "training_code_version": self.training_code_version,
            "n_train_samples": int(self.n_train_samples),
            "n_features": int(self.n_features),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def is_compatible_with(
        self,
        *,
        encoder_schema_hash: str,
        dataset_schema_hash: str,
        train_split_hash: str = "",
        normalization_hash: str = "",
        feature_schema_hash: str = "",
        target_schema_hash: str = "",
        strict: bool = False,
    ) -> bool:
        """Check full compatibility with encoder, dataset, split, and normalization.

        Fix 3: Requires identity match on all binding fields, not just schema.

        Args:
            strict: If True, ALL identity fields must exist (non-empty) on
                both the artifact and the query side, and must match exactly.
                Wildcards are forbidden. This is the only mode permitted
                in scientific runs (exp4.2+). If False (default), empty
                fields are treated as wildcards for backward compatibility
                with legacy artifacts.
        """
        if strict:
            # In strict mode, every field must be present and match exactly.
            # No wildcards allowed.
            field_pairs = [
                (self.encoder_schema_hash, encoder_schema_hash, "encoder_schema_hash"),
                (self.dataset_schema_hash, dataset_schema_hash, "dataset_schema_hash"),
                (self.train_split_hash, train_split_hash, "train_split_hash"),
                (self.normalization_hash, normalization_hash, "normalization_hash"),
                (self.feature_schema_hash, feature_schema_hash, "feature_schema_hash"),
                (self.target_schema_hash, target_schema_hash, "target_schema_hash"),
            ]
            for artifact_val, query_val, name in field_pairs:
                if not artifact_val or not query_val:
                    return False
                if artifact_val != query_val:
                    return False
            return True

        # Non-strict: empty fields are wildcards (backward compatible).
        checks = [
            self.encoder_schema_hash == encoder_schema_hash,
            self.dataset_schema_hash == dataset_schema_hash,
        ]
        # Identity-level checks (only enforced if both sides provide a value).
        if self.train_split_hash and train_split_hash:
            checks.append(self.train_split_hash == train_split_hash)
        if self.normalization_hash and normalization_hash:
            checks.append(self.normalization_hash == normalization_hash)
        if self.feature_schema_hash and feature_schema_hash:
            checks.append(self.feature_schema_hash == feature_schema_hash)
        if self.target_schema_hash and target_schema_hash:
            checks.append(self.target_schema_hash == target_schema_hash)
        return all(checks)

    def load_state(self) -> Any:
        """Deserialize and return the model state.

        Fix 1: Returns the actual learned parameters (weights, stumps, etc.).
        """
        if not self.model_state:
            return None
        return _deserialize_state(self.model_state)

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
            "feature_schema_hash": self.feature_schema_hash,
            "target_schema_hash": self.target_schema_hash,
            "hyperparameter_hash": self.hyperparameter_hash,
            "model_state_hash": self.model_state_hash,
            "seed": int(self.seed),
            "training_code_version": self.training_code_version,
            "n_train_samples": int(self.n_train_samples),
            "n_features": int(self.n_features),
            "has_model_state": bool(self.model_state),
            "metrics": dict(self.metrics),
            "created_at": self.created_at,
            "description": self.description,
            "artifact_hash": self.artifact_hash,
        }

    def to_json(self) -> str:
        """Serialize to JSON (without model_state to keep it readable)."""
        return json.dumps(self.to_log(), sort_keys=True, indent=2)

    def to_full_json(self) -> str:
        """Full serialization including model state."""
        log = self.to_log()
        log["model_state"] = self.model_state
        return json.dumps(log, sort_keys=True, indent=2)


def create_artifact(
    model: Any,
    *,
    encoder_id: str,
    encoder_schema_hash: str,
    dataset_schema_hash: str,
    train_split_hash: str = "",
    normalization_hash: str = "",
    feature_schema_hash: str = "",
    target_schema_hash: str = "",
    training_code_version: str = "v6.0-exp4.1",
    metrics: dict[str, Any] | None = None,
    description: str = "",
) -> ModelArtifact:
    """Create a ModelArtifact from a fitted model.

    Fix 1: Extracts and serializes actual model state via ``model.get_state()``.
    Fix 2: Uses ``model.hyperparameters()`` for complete hyperparameter hashing.
    Fix 3: Binds to all identity fields, not just schema.
    """
    # Fix 2: Complete hyperparameter configuration.
    hp = model.hyperparameters() if hasattr(model, "hyperparameters") else {
        "model_type": getattr(model, "model_type", "unknown"),
        "version": getattr(model, "version", "unknown"),
        "seed": getattr(model, "seed", 42),
    }
    hp_content = json.dumps(hp, sort_keys=True, default=str)
    hp_hash = hashlib.sha256(hp_content.encode()).hexdigest()[:16]

    # Fix 1: Serialize actual model state.
    model_state = ""
    state_hash = ""
    if hasattr(model, "get_state"):
        state = model.get_state()
        if state is not None:
            model_state = _serialize_state(state)
            state_hash = _state_hash(state)

    return ModelArtifact(
        model_id=getattr(model, "model_id", "unknown"),
        predictor_type=getattr(model, "model_type", "unknown"),
        predictor_version=getattr(model, "version", "unknown"),
        encoder_id=encoder_id,
        encoder_schema_hash=encoder_schema_hash,
        dataset_schema_hash=dataset_schema_hash,
        train_split_hash=train_split_hash,
        normalization_hash=normalization_hash,
        feature_schema_hash=feature_schema_hash,
        target_schema_hash=target_schema_hash,
        hyperparameter_hash=hp_hash,
        model_state_hash=state_hash,
        seed=getattr(model, "seed", 42),
        training_code_version=training_code_version,
        n_train_samples=getattr(model, "_n_samples", 0),
        n_features=getattr(model, "_n_features", 0) or getattr(model, "_in_dim", 0),
        model_state=model_state,
        metrics=metrics or {},
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        description=description,
    )
