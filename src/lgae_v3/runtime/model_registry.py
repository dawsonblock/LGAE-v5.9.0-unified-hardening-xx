"""Model registry (Phase 45).

A registry of learned models/policies with metadata, maturity level, and
content hash. The registry is append-only: a model is registered once and
its metadata is immutable. Promotion updates the maturity level via an
auditable transition.

This is not a model server; it is a governance artifact that ties model
identity to promotion gates and qualification reports.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .promotion import PromotionLevel


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """Immutable record of one registered model."""
    model_id: str
    name: str
    version: str
    content_hash: str
    registered_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    maturity: PromotionLevel = PromotionLevel.EXPERIMENTAL

    def to_log(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "content_hash": self.content_hash,
            "registered_at": float(self.registered_at),
            "maturity": self.maturity.name,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class PromotionTransition:
    """Auditable record of a maturity transition."""
    model_id: str
    from_level: PromotionLevel
    to_level: PromotionLevel
    timestamp: float
    gate_report: dict[str, Any] = field(default_factory=dict)
    approved: bool = False

    def to_log(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "from_level": self.from_level.name,
            "to_level": self.to_level.name,
            "timestamp": float(self.timestamp),
            "approved": bool(self.approved),
            "gate_report": self.gate_report,
        }


class ModelRegistry:
    """Append-only model registry with auditable promotion transitions."""

    def __init__(self) -> None:
        self._models: dict[str, ModelRecord] = {}
        self._transitions: list[PromotionTransition] = []

    def register(
        self,
        *,
        name: str,
        version: str,
        content: bytes | str,
        metadata: dict[str, Any] | None = None,
    ) -> ModelRecord:
        """Register a new model. Returns the immutable record."""
        content_hash = hashlib.sha256(
            content.encode("utf-8") if isinstance(content, str) else content
        ).hexdigest()
        model_id = f"{name}:{version}:{content_hash[:12]}"
        if model_id in self._models:
            raise ValueError(f"model {model_id!r} is already registered")
        record = ModelRecord(
            model_id=model_id,
            name=str(name),
            version=str(version),
            content_hash=content_hash,
            registered_at=time.time(),
            metadata=dict(metadata or {}),
            maturity=PromotionLevel.EXPERIMENTAL,
        )
        self._models[model_id] = record
        return record

    def get(self, model_id: str) -> ModelRecord:
        return self._models[str(model_id)]

    @property
    def models(self) -> list[ModelRecord]:
        return [self._models[k] for k in sorted(self._models.keys())]

    @property
    def transitions(self) -> list[PromotionTransition]:
        return list(self._transitions)

    def promote(
        self,
        model_id: str,
        to_level: PromotionLevel,
        *,
        gate_report: dict[str, Any] | None = None,
        approved: bool = True,
    ) -> PromotionTransition:
        """Record a maturity promotion transition for a model."""
        record = self.get(model_id)
        from_level = record.maturity
        if to_level <= from_level:
            raise ValueError(
                f"promotion must increase maturity: {from_level.name} -> {to_level.name}"
            )
        transition = PromotionTransition(
            model_id=str(model_id),
            from_level=from_level,
            to_level=to_level,
            timestamp=time.time(),
            gate_report=dict(gate_report or {}),
            approved=bool(approved),
        )
        self._transitions.append(transition)
        # Update the model's maturity (replace the frozen record).
        from dataclasses import replace
        self._models[model_id] = replace(record, maturity=to_level)
        return transition

    def models_at_maturity(self, level: PromotionLevel) -> list[ModelRecord]:
        return [m for m in self.models if m.maturity == level]

    def to_log(self) -> dict[str, Any]:
        return {
            "model_count": len(self._models),
            "models": [m.to_log() for m in self.models],
            "transition_count": len(self._transitions),
            "transitions": [t.to_log() for t in self._transitions],
        }
