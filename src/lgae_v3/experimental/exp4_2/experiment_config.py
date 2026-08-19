"""Experiment configuration for exp4.2.

Freezes encoder and predictor matrices, selection weights, and finalist
configurations before held-out access begins.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json
import time


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Frozen encoder configuration."""
    encoder_id: str
    version: str = ""
    dimension: int = 0
    schema_hash: str = ""
    requires_fit: bool = True
    normalization_hash: str = ""

    def to_log(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "version": self.version,
            "dimension": int(self.dimension),
            "schema_hash": self.schema_hash,
            "requires_fit": bool(self.requires_fit),
            "normalization_hash": self.normalization_hash,
        }


@dataclass(frozen=True, slots=True)
class PredictorConfig:
    """Frozen predictor configuration."""
    predictor_id: str
    model_type: str = ""
    version: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    deterministic: bool = True

    def to_log(self) -> dict[str, Any]:
        return {
            "predictor_id": self.predictor_id,
            "model_type": self.model_type,
            "version": self.version,
            "hyperparameters": dict(self.hyperparameters),
            "deterministic": bool(self.deterministic),
        }


@dataclass(frozen=True, slots=True)
class SelectionWeights:
    """Weights for the validation-based model selection score.

    Frozen BEFORE held-out access. Do not optimize primarily for RMSE.
    """
    w_spearman: float = 0.25
    w_ndcg: float = 0.15
    w_regret: float = 0.20
    w_sign_accuracy: float = 0.10
    w_ece: float = 0.10
    w_latency_penalty: float = 0.05
    w_ood_proxy: float = 0.15

    def to_log(self) -> dict[str, Any]:
        return {
            "w_spearman": float(self.w_spearman),
            "w_ndcg": float(self.w_ndcg),
            "w_regret": float(self.w_regret),
            "w_sign_accuracy": float(self.w_sign_accuracy),
            "w_ece": float(self.w_ece),
            "w_latency_penalty": float(self.w_latency_penalty),
            "w_ood_proxy": float(self.w_ood_proxy),
        }

    def compute_score(
        self,
        *,
        spearman: float,
        ndcg: float,
        regret: float,
        sign_accuracy: float,
        ece: float,
        latency_ms: float,
        ood_proxy: float = 0.0,
    ) -> float:
        """Compute the weighted validation score."""
        return (
            self.w_spearman * spearman
            + self.w_ndcg * ndcg
            - self.w_regret * regret
            + self.w_sign_accuracy * sign_accuracy
            - self.w_ece * ece
            - self.w_latency_penalty * (latency_ms / 1000.0)
            + self.w_ood_proxy * ood_proxy
        )


@dataclass
class FinalistLock:
    """Locked finalist configuration.

    Generated after validation, before held-out access.
    Hashed and immutable once created.
    """
    finalists: list[dict[str, Any]] = field(default_factory=list)
    selection_weights: dict[str, float] = field(default_factory=dict)
    locked_at: str = ""
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not self.locked_at:
            self.locked_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not self.config_hash:
            self.config_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "finalists": list(self.finalists),
            "selection_weights": dict(self.selection_weights),
            "locked_at": self.locked_at,
        }, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_log(self) -> dict[str, Any]:
        return {
            "finalists": list(self.finalists),
            "selection_weights": dict(self.selection_weights),
            "locked_at": self.locked_at,
            "config_hash": self.config_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_log(), sort_keys=True, indent=2)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())


@dataclass
class ExperimentConfig:
    """Full experiment configuration for exp4.2.

    All matrices and parameters are frozen at creation time.
    """
    experiment_id: str = "LGAE_V6_EXP4_2_STRUCTURAL_PREDICTION_STUDY_001"
    encoders: list[EncoderConfig] = field(default_factory=list)
    predictors: list[PredictorConfig] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    selection_weights: SelectionWeights = field(default_factory=SelectionWeights)
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    n_epochs: int = 50
    n_ensemble: int = 3
    catastrophic_regret_threshold: float = 0.1
    coverage_levels: list[float] = field(
        default_factory=lambda: [1.0, 0.9, 0.75, 0.5, 0.25]
    )
    bootstrap_samples: int = 1000
    bootstrap_confidence: float = 0.95
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_log(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "encoders": [e.to_log() for e in self.encoders],
            "predictors": [p.to_log() for p in self.predictors],
            "targets": list(self.targets),
            "selection_weights": self.selection_weights.to_log(),
            "seeds": list(self.seeds),
            "n_epochs": int(self.n_epochs),
            "n_ensemble": int(self.n_ensemble),
            "catastrophic_regret_threshold": float(self.catastrophic_regret_threshold),
            "coverage_levels": list(self.coverage_levels),
            "bootstrap_samples": int(self.bootstrap_samples),
            "bootstrap_confidence": float(self.bootstrap_confidence),
            "created_at": self.created_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_log(), sort_keys=True, indent=2)


def default_experiment_config() -> ExperimentConfig:
    """Create the default experiment configuration with frozen matrices."""
    from ..encoders import EncoderRegistry
    from ..models.model_registry import ModelRegistry

    encoders = []
    for enc_id in EncoderRegistry.available_encoders():
        info = EncoderRegistry.encoder_info(enc_id)
        encoders.append(EncoderConfig(
            encoder_id=enc_id,
            version=info.get("version", ""),
            dimension=info.get("dimension", 0),
            schema_hash=info.get("schema_hash", ""),
            requires_fit=info.get("requires_fit", True),
        ))

    predictors = []
    for pred_id in ModelRegistry.available_models():
        info = ModelRegistry.model_info(pred_id)
        predictors.append(PredictorConfig(
            predictor_id=pred_id,
            model_type=info.get("model_type", ""),
            version=info.get("version", ""),
            deterministic=info.get("deterministic", True),
        ))

    return ExperimentConfig(
        encoders=encoders,
        predictors=predictors,
        targets=["realized_delta", "sign_delta", "risk", "cost"],
    )
