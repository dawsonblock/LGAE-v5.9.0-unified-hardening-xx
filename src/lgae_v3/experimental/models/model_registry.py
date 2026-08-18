"""Model registry with compatibility checking and artifact binding.

The registry allows:
    model = ModelRegistry.create("linear")
    artifact = ModelRegistry.register(model, ...)

No model should be loadable against an incompatible encoder or dataset
without an explicit compatibility failure.
"""
from __future__ import annotations

from typing import Any
import hashlib
import json

from .protocol import ModelLifecycle
from .baselines import GlobalMeanPredictor, MutationTypeMeanPredictor, NearestExperiencePredictor
from .linear import LinearRegressionPredictor, RidgeRegressionPredictor, LogisticRegressionPredictor
from .tree import GradientBoostedTreePredictor
from .mlp import MLPRegressor, MLPClassifier
from .ranking import PointwiseRankingModel, PairwiseRankingModel
from .artifact import ModelArtifact, CompatibilityError, create_artifact


class ModelRegistry:
    """Registry for creating and managing outcome/risk/cost models."""

    _registry: dict[str, type] = {
        # Baselines.
        "global_mean": GlobalMeanPredictor,
        "mutation_type_mean": MutationTypeMeanPredictor,
        "nearest_experience": NearestExperiencePredictor,
        # Linear models.
        "linear": LinearRegressionPredictor,
        "ridge": RidgeRegressionPredictor,
        "logistic": LogisticRegressionPredictor,
        # Tree.
        "tree": GradientBoostedTreePredictor,
        # MLP.
        "mlp": MLPRegressor,
        "mlp_clf": MLPClassifier,
        # Ranking.
        "pointwise_rank": PointwiseRankingModel,
        "pairwise_rank": PairwiseRankingModel,
    }

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Any:
        """Create a model by name."""
        if name not in cls._registry:
            raise KeyError(
                f"Unknown model: '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_models(cls) -> list[str]:
        """List all available model names."""
        return list(cls._registry.keys())

    @classmethod
    def model_info(cls, name: str) -> dict[str, Any]:
        """Get metadata about a model without creating it."""
        if name not in cls._registry:
            raise KeyError(f"Unknown model: '{name}'")
        model = cls._registry[name]()
        return {
            "model_type": getattr(model, "model_type", "unknown"),
            "version": getattr(model, "version", "unknown"),
            "requires_fit": getattr(model, "requires_fit", True),
            "deterministic": getattr(model, "deterministic", True),
            "model_id": getattr(model, "model_id", "unknown"),
        }

    @classmethod
    def all_model_info(cls) -> list[dict[str, Any]]:
        """Get metadata for all available models."""
        return [cls.model_info(name) for name in cls._registry]

    @classmethod
    def register(
        cls,
        model: Any,
        *,
        encoder_id: str,
        encoder_schema_hash: str,
        dataset_schema_hash: str,
        train_split_hash: str = "",
        normalization_hash: str = "",
        metrics: dict[str, Any] | None = None,
        description: str = "",
    ) -> ModelArtifact:
        """Register a fitted model as an artifact."""
        return create_artifact(
            model,
            encoder_id=encoder_id,
            encoder_schema_hash=encoder_schema_hash,
            dataset_schema_hash=dataset_schema_hash,
            train_split_hash=train_split_hash,
            normalization_hash=normalization_hash,
            metrics=metrics,
            description=description,
        )

    @classmethod
    def verify_compatibility(
        cls,
        artifact: ModelArtifact,
        *,
        encoder_schema_hash: str,
        dataset_schema_hash: str,
    ) -> None:
        """Verify that an artifact is compatible with the given encoder and dataset.

        Raises:
            CompatibilityError: If the artifact is not compatible.
        """
        if not artifact.is_compatible_with(encoder_schema_hash, dataset_schema_hash):
            raise CompatibilityError(
                f"Model artifact {artifact.model_id} is not compatible. "
                f"Expected encoder_schema={encoder_schema_hash}, "
                f"got {artifact.encoder_schema_hash}. "
                f"Expected dataset_schema={dataset_schema_hash}, "
                f"got {artifact.dataset_schema_hash}."
            )
