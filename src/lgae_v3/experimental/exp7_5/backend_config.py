"""Backend configuration for exp7.5.

Frozen, hashable configuration that gets recorded in every
experiment artifact. No secrets here — only model identifiers
and operational parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import hashlib
import json


@dataclass(frozen=True)
class BackendConfig:
    """Frozen backend configuration. No secrets."""

    provider: str                          # "openai", "mock", "local"
    model_id: str                          # e.g. "gpt-4o-mini" or "ft:gpt-4o-mini:org:run:abc123"
    temperature: float = 0.0               # deterministic initially
    max_output_tokens: int = 1024
    timeout_seconds: float = 30.0
    max_retries: int = 3
    # Pricing per 1M tokens (for dollar cost tracking)
    input_price_per_mtok: float = 0.0      # $/1M input tokens
    output_price_per_mtok: float = 0.0     # $/1M output tokens
    cached_input_price_per_mtok: float = 0.0  # $/1M cached input tokens
    # SDK version (for provenance)
    sdk_version: str = ""
    # Backend version
    backend_version: str = "exp7.5-v1"

    @property
    def config_hash(self) -> str:
        """Stable hash of the config (excluding secrets)."""
        data = {
            "provider": self.provider,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "input_price_per_mtok": self.input_price_per_mtok,
            "output_price_per_mtok": self.output_price_per_mtok,
            "cached_input_price_per_mtok": self.cached_input_price_per_mtok,
            "sdk_version": self.sdk_version,
            "backend_version": self.backend_version,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "input_price_per_mtok": self.input_price_per_mtok,
            "output_price_per_mtok": self.output_price_per_mtok,
            "cached_input_price_per_mtok": self.cached_input_price_per_mtok,
            "sdk_version": self.sdk_version,
            "backend_version": self.backend_version,
            "config_hash": self.config_hash,
        }

    def compute_dollar_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Compute dollar cost for a single request."""
        cost = 0.0
        cost += (input_tokens / 1_000_000) * self.input_price_per_mtok
        cost += (output_tokens / 1_000_000) * self.output_price_per_mtok
        # Cached tokens are charged at cached rate (usually cheaper).
        # If cached_tokens > 0, subtract the difference.
        if cached_tokens > 0:
            # Assume input_tokens includes cached; adjust.
            non_cached = max(0, input_tokens - cached_tokens)
            cost = (non_cached / 1_000_000) * self.input_price_per_mtok
            cost += (cached_tokens / 1_000_000) * self.cached_input_price_per_mtok
            cost += (output_tokens / 1_000_000) * self.output_price_per_mtok
        return cost


# Predefined configs for common models.
MOCK_CONFIG = BackendConfig(
    provider="mock",
    model_id="mock-v1",
    temperature=0.0,
    max_output_tokens=512,
    backend_version="exp7.5-mock",
)


def make_openai_config(
    model_id: str,
    *,
    input_price: float = 0.0,
    output_price: float = 0.0,
    cached_price: float = 0.0,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
) -> BackendConfig:
    """Create an OpenAI backend config.

    Pricing should be set to the actual model pricing.
    For fine-tuned models, use the fine-tuned model's pricing.
    """
    import openai
    sdk_ver = ""
    try:
        sdk_ver = openai.__version__
    except Exception:
        pass
    return BackendConfig(
        provider="openai",
        model_id=model_id,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout_seconds=30.0,
        max_retries=3,
        input_price_per_mtok=input_price,
        output_price_per_mtok=output_price,
        cached_input_price_per_mtok=cached_price,
        sdk_version=sdk_ver,
        backend_version="exp7.5-openai-v1",
    )
