"""DeepSeek backend for exp7.5.

DeepSeek uses an OpenAI-compatible API. This backend wraps the
OpenAI client with a different base_url.

API key is read from DEEPSEEK_API_KEY environment variable.
Never logged, never committed.
"""
from __future__ import annotations

import os
import time
import logging
from typing import Optional

from ...exp7_2.model_backend import ModelResponse, Message
from ..backend_config import BackendConfig
from .openai_backend import BackendStatus, BudgetGuard

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekBackend:
    """DeepSeek backend using OpenAI-compatible Chat Completions API.

    DeepSeek supports the standard chat.completions.create endpoint.
    Reads API key from DEEPSEEK_API_KEY environment variable.
    """

    def __init__(
        self,
        config: BackendConfig,
        *,
        budget: Optional[BudgetGuard] = None,
    ) -> None:
        self.config = config
        self.budget = budget or BudgetGuard()

        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self._api_key_present = bool(api_key)
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY environment variable not set. "
                "Set it before creating DeepSeekBackend."
            )

        try:
            import openai
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            ) from e

        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=config.timeout_seconds,
        )

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
        """Generate a response using DeepSeek's OpenAI-compatible API."""
        t0 = time.time()

        # Check budget.
        if self.budget.check():
            return ModelResponse(
                text="",
                tokens_in=0, tokens_out=0,
                latency_ms=0.0,
                model_id=self.config.model_id,
                status=BackendStatus.BUDGET_EXCEEDED.value,
                error="Budget exceeded",
            )

        # Build messages for chat completions.
        full_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            full_messages.append({"role": msg.role, "content": msg.content})

        try:
            response = self._client.chat.completions.create(
                model=self.config.model_id,
                messages=full_messages,
                max_tokens=min(max_tokens, self.config.max_output_tokens),
                temperature=temperature if temperature >= 0 else self.config.temperature,
            )
        except Exception as e:
            err_str = str(e)
            status = BackendStatus.API_ERROR.value
            if "429" in err_str or "rate_limit" in err_str.lower():
                status = BackendStatus.RATE_LIMIT.value
            if "timeout" in err_str.lower() or "timed out" in err_str.lower():
                status = BackendStatus.TIMEOUT.value
            logger.warning(f"DeepSeek API error: {status}: {err_str[:200]}")
            return ModelResponse(
                text="",
                tokens_in=0, tokens_out=0,
                latency_ms=(time.time() - t0) * 1000,
                model_id=self.config.model_id,
                status=status,
                error=err_str[:500],
            )

        latency_ms = (time.time() - t0) * 1000

        try:
            output_text = response.choices[0].message.content or ""
            usage = response.usage

            input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            cached_tokens = 0
            # DeepSeek may report cached tokens in prompt_tokens_details.
            if usage:
                prompt_details = getattr(usage, "prompt_tokens_details", None)
                if prompt_details:
                    cached_tokens = getattr(prompt_details, "cached_tokens", 0)

            request_id = getattr(response, "id", "")

            dollar_cost = self.config.compute_dollar_cost(
                input_tokens, output_tokens, cached_tokens
            )

            self.budget.record(input_tokens + output_tokens, dollar_cost)

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
                finish_reason=response.choices[0].finish_reason or "stop",
                error=error_msg,
                cached_tokens=cached_tokens,
                model_id=self.config.model_id,
                request_id=request_id,
                status=status,
                dollar_cost=dollar_cost,
            )

        except Exception as e:
            logger.warning(f"Failed to parse DeepSeek response: {e}")
            return ModelResponse(
                text="",
                tokens_in=0, tokens_out=0,
                latency_ms=latency_ms,
                model_id=self.config.model_id,
                status=BackendStatus.INVALID_RESPONSE.value,
                error=f"Response parsing failed: {str(e)[:200]}",
            )

    def get_provenance(self) -> dict:
        return {
            "provider": self.config.provider,
            "model_id": self.config.model_id,
            "api_key_present": self._api_key_present,
            "config_hash": self.config.config_hash,
            "sdk_version": self.config.sdk_version,
            "backend_version": self.config.backend_version,
            "base_url": DEEPSEEK_BASE_URL,
            "budget": self.budget.summary(),
        }

    def verify_model(self) -> tuple[bool, str]:
        """Verify model ID via the models list."""
        try:
            models = self._client.models.list()
            model_ids = [m.id for m in models.data]
            if self.config.model_id in model_ids:
                return True, f"Model {self.config.model_id} verified"
            return False, f"Model {self.config.model_id} not found. Available: {model_ids}"
        except Exception as e:
            return False, f"Model verification failed: {str(e)[:200]}"
