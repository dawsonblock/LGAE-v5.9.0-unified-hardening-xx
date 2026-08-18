"""Encoder registry with provenance and dataset binding.

Allows:
    encoder = EncoderRegistry.create("global-local-v1")

Every encoder exposes:
    name, version, dimension, schema_hash, requires_fit,
    deterministic, supported action types

The registry preserves configuration/provenance and binds encoders
to exact dataset versions.
"""
from __future__ import annotations

from typing import Any
import hashlib
import json

from .protocol import ActionEncodingSchema, DEFAULT_ACTION_SCHEMA
from .minimal import MinimalControlEncoder
from .global_features import GlobalStateEncoder
from .local_action import LocalActionEncoder
from .semantic_action import SemanticActionEncoder
from .local_subgraph import LocalSubgraphEncoder
from .geometric import GeometricEncoder
from .spectral import SpectralEncoder
from .learned_graph import SmallLearnedGraphEncoder
from .hybrid import HybridEncoder


class EncoderRegistry:
    """Registry for creating and managing structural encoders.

    Encoders are created by name and can be bound to specific dataset
    versions for provenance tracking.
    """

    _registry: dict[str, type] = {
        "minimal-control": MinimalControlEncoder,
        "global": GlobalStateEncoder,
        "global-local": LocalActionEncoder,
        "semantic-action": SemanticActionEncoder,
        "local-subgraph": LocalSubgraphEncoder,
        "geometric": GeometricEncoder,
        "spectral": SpectralEncoder,
        "learned-graph": SmallLearnedGraphEncoder,
        "hybrid": HybridEncoder,
    }

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Any:
        """Create an encoder by name.

        Args:
            name: Encoder name (e.g., "global-local").
            **kwargs: Encoder-specific configuration.

        Returns:
            A new encoder instance.

        Raises:
            KeyError: If the encoder name is not registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Unknown encoder: '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_encoders(cls) -> list[str]:
        """List all available encoder names."""
        return list(cls._registry.keys())

    @classmethod
    def encoder_info(cls, name: str) -> dict[str, Any]:
        """Get metadata about an encoder without creating it."""
        if name not in cls._registry:
            raise KeyError(f"Unknown encoder: '{name}'")
        # Create a temporary instance to get metadata.
        enc = cls._registry[name]()
        return {
            "name": enc.name,
            "version": enc.version,
            "dimension": enc.dimension,
            "schema_hash": enc.schema_hash,
            "requires_fit": enc.requires_fit,
            "deterministic": enc.deterministic,
            "lifecycle": enc.lifecycle,
        }

    @classmethod
    def all_encoder_info(cls) -> list[dict[str, Any]]:
        """Get metadata for all available encoders."""
        return [cls.encoder_info(name) for name in cls._registry]


class EncoderProvenance:
    """Provenance binding for a fitted encoder.

    Binds an encoder to the exact dataset it was fitted on, including:
    - dataset schema hash
    - train split hash
    - validation split hash
    - held-out split hash
    - feature schema hash
    """

    def __init__(
        self,
        encoder_id: str,
        encoder_version: str,
        encoder_schema_hash: str,
        dataset_schema_hash: str = "",
        train_split_hash: str = "",
        validation_split_hash: str = "",
        heldout_split_hash: str = "",
        feature_schema_hash: str = "",
        normalization_hash: str = "",
    ) -> None:
        self.encoder_id = encoder_id
        self.encoder_version = encoder_version
        self.encoder_schema_hash = encoder_schema_hash
        self.dataset_schema_hash = dataset_schema_hash
        self.train_split_hash = train_split_hash
        self.validation_split_hash = validation_split_hash
        self.heldout_split_hash = heldout_split_hash
        self.feature_schema_hash = feature_schema_hash
        self.normalization_hash = normalization_hash

    @property
    def provenance_hash(self) -> str:
        content = json.dumps({
            "encoder_id": self.encoder_id,
            "encoder_version": self.encoder_version,
            "encoder_schema_hash": self.encoder_schema_hash,
            "dataset_schema_hash": self.dataset_schema_hash,
            "train_split_hash": self.train_split_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "normalization_hash": self.normalization_hash,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_log(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "encoder_version": self.encoder_version,
            "encoder_schema_hash": self.encoder_schema_hash,
            "dataset_schema_hash": self.dataset_schema_hash,
            "train_split_hash": self.train_split_hash,
            "validation_split_hash": self.validation_split_hash,
            "heldout_split_hash": self.heldout_split_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "normalization_hash": self.normalization_hash,
            "provenance_hash": self.provenance_hash,
        }
