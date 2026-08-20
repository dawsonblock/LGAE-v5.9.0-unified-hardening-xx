"""OpenAI backend for exp7.5.

Uses the OpenAI Responses API (client.responses.create).
Returns normalized telemetry via the existing ModelResponse —
no OpenAI-specific objects leak.

Security:
  - API key is read from OPENAI_API_KEY environment variable
  - Never logged, never written to artifacts
  - Only OPENAI_API_KEY_PRESENT=true is recorded

Error handling:
  - SUCCESS: normal completion
  - TIMEOUT: request timed out
  - RATE_LIMIT: 429 from API
  - API_ERROR: other API errors
  - INVALID_RESPONSE: response parsing failed
  - BUDGET_EXCEEDED: budget guard triggered

Budget guards:
  - max_api_calls, max_tokens, max_dollar_cost
  - Fails closed if exceeded
"""
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from ...exp7_2.model_backend import ModelBackend, ModelResponse, Message
from ..backend_config import BackendConfig

logger = logging.getLogger(__name__)


class BackendStatus(str, Enum):
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    API_ERROR = "API_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


@dataclass
class BudgetGuard:
    """Tracks cumulative API usage and fails closed if exceeded."""
    max_api_calls: int = 10000
    max_tokens: int = 10_000_000
    max_dollar_cost: float = 100.0

    n_calls: int = 0
    n_tokens: int = 0
    dollar_cost: float = 0.0

    def check(self) -> bool:
        if self.n_calls >= self.max_api_calls:
            return True
        if self.n_tokens >= self.max_tokens:
            return True
        if self.dollar_cost >= self.max_dollar_cost:
            return True
        return False

    def record(self, tokens: int, cost: float) -> None:
        self.n_calls += 1
        self.n_tokens += tokens
        self.dollar_cost += cost

    def summary(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_tokens": self.n_tokens,
            "dollar_cost": round(self.dollar_cost, 4),
            "max_api_calls": self.max_api_calls,
            "max_tokens": self.max_tokens,
            "max_dollar_cost": self.max_dollar_cost,
            "budget_exceeded": self.check(),
        }


class OpenAIBackend:
    """OpenAI backend using the Responses API.

    Reads API key from OPENAI_API_KEY environment variable.
    Never logs or exposes the key.

    Synchronous — matches the existing ModelBackend protocol.
    """

    def __init__(
        self,
        config: BackendConfig,
        *,
        budget: Optional[BudgetGuard] = None,
    ) -> None:
        self.config = config
        self.budget = budget or BudgetGuard()

        api_key = os.environ.get("OPENAI_API_KEY", "")
        self._api_key_present = bool(api_key)
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable not set. "
                "Set it before creating OpenAIBackend."
            )

        try:
            import openai
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            ) from e

        self._openai = openai
        self._client = openai.OpenAI(api_key=api_key, timeout=config.timeout_seconds)

    @property
    def api_key_present(self) -> bool:
        return self._api_key_present

    def generate(
        self,
        *,
        role: str,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """Generate a response using the OpenAI Responses API."""
        t0 = time.time()

        # Check budget before making the call.
        if self.budget.check():
            return ModelResponse(
                text="",
                tokens_in=0, tokens_out=0,
                latency_ms=0.0,
                model_id=self.config.model_id,
                status=BackendStatus.BUDGET_EXCEEDED.value,
                error="Budget exceeded",
            )

        # Build input text from messages.
        input_text = self._build_input(messages)

        try:
            response = self._client.responses.create(
                model=self.config.model_id,
                input=input_text,
                instructions=system_prompt,
                max_output_tokens=min(max_tokens, self.config.max_output_tokens),
                temperature=temperature if temperature >= 0 else self.config.temperature,
            )
        except Exception as e:
            err_str = str(e)
            status = BackendStatus.API_ERROR.value
            if "429" in err_str or "rate_limit" in err_str.lower():
                status = BackendStatus.RATE_LIMIT.value
            if "timeout" in err_str.lower() or "timed out" in err_str.lower():
                status = BackendStatus.TIMEOUT.value
            logger.warning(f"OpenAI API error: {status}: {err_str[:200]}")
            return ModelResponse(
                text="",
                tokens_in=0, tokens_out=0,
                latency_ms=(time.time() - t0) * 1000,
                model_id=self.config.model_id,
                status=status,
                error=err_str[:500],
            )

        latency_ms = (time.time() - t0) * 1000

        # Extract normalized telemetry.
        try:
            output_text = response.output_text or ""
            usage = response.usage

            input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
            cached_tokens = 0
            if usage:
                input_details = getattr(usage, "input_tokens_details", None)
                if input_details:
                    cached_tokens = getattr(input_details, "cached_tokens", 0)

            request_id = getattr(response, "id", "")

            # Compute dollar cost.
            dollar_cost = self.config.compute_dollar_cost(
                input_tokens, output_tokens, cached_tokens
            )

            # Record in budget.
            self.budget.record(input_tokens + output_tokens, dollar_cost)

            # Determine status.
            if not output_text.strip():
                status = BackendStatus.INVALID_RESPONSE.value
                error_msg = "Empty response"
            else:
                status = BackendStatus.SUCCESS.value
                error_msg = None

            return ModelResponse(
                text=output_text,
                tokens_in=input_tokens,
                tokens_out=output_tokens,
                latency_ms=latency_ms,
                confidence=0.8,
                finish_reason="stop",
                error=error_msg,
                cached_tokens=cached_tokens,
                model_id=self.config.model_id,
                request_id=request_id,
                status=status,
                dollar_cost=dollar_cost,
            )

        except Exception as e:
            logger.warning(f"Failed to parse OpenAI response: {e}")
            return ModelResponse(
                text="",
                tokens_in=0, tokens_out=0,
                latency_ms=latency_ms,
                model_id=self.config.model_id,
                status=BackendStatus.INVALID_RESPONSE.value,
                error=f"Response parsing failed: {str(e)[:200]}",
            )

    def _build_input(self, messages: list[Message]) -> str:
        """Build input text from messages."""
        parts = []
        for msg in messages:
            parts.append(f"{msg.role}: {msg.content}")
        return "\n\n".join(parts)

    def get_provenance(self) -> dict:
        """Get provenance info (no secrets)."""
        return {
            "provider": self.config.provider,
            "model_id": self.config.model_id,
            "api_key_present": self._api_key_present,
            "config_hash": self.config.config_hash,
            "sdk_version": self.config.sdk_version,
            "backend_version": self.config.backend_version,
            "budget": self.budget.summary(),
        }

    def verify_model(self) -> tuple[bool, str]:
        """Verify that the model ID is valid via the Models API."""
        try:
            model = self._client.models.retrieve(self.config.model_id)
            return True, f"Model {model.id} verified"
        except Exception as e:
            return False, f"Model verification failed: {str(e)[:200]}"
