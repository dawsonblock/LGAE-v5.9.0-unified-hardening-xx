"""Response cache for deterministic API calls.

Caches by Hash(model, prompt, context, temperature, max_tokens).
Avoids paying for identical calls twice across analysis scripts.

Does NOT cache calls whose context intentionally changes due to
topology — those have different upstream context and thus different
cache keys naturally.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from ...exp7_2.model_backend import ModelResponse, Message


@dataclass
class CacheEntry:
    response: ModelResponse
    timestamp: float
    hit_count: int = 0


class ResponseCache:
    """Disk-backed response cache for deterministic API calls."""

    def __init__(self, cache_dir: str = ".api_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_hits = 0
        self.n_misses = 0
        self.tokens_saved = 0
        self.dollars_saved = 0.0

    def _make_key(
        self,
        model_id: str,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        role: str,
    ) -> str:
        """Create a deterministic cache key."""
        msg_str = json.dumps(
            [{"role": m.role, "content": m.content} for m in messages],
            sort_keys=True,
        )
        key_data = f"{model_id}|{role}|{system_prompt}|{msg_str}|{max_tokens}|{temperature}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(
        self,
        model_id: str,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        role: str,
    ) -> Optional[ModelResponse]:
        """Get a cached response if available."""
        key = self._make_key(model_id, system_prompt, messages, max_tokens, temperature, role)
        path = self.cache_dir / f"{key}.json"

        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self.n_hits += 1
                self.tokens_saved += data.get("total_tokens", 0)
                self.dollars_saved += data.get("dollar_cost", 0.0)
                return ModelResponse(
                    text=data["text"],
                    tokens_in=data.get("tokens_in", 0),
                    tokens_out=data.get("tokens_out", 0),
                    latency_ms=data.get("latency_ms", 0.0),
                    confidence=data.get("confidence", 0.8),
                    finish_reason=data.get("finish_reason", "stop"),
                    error=data.get("error"),
                    cached_tokens=data.get("cached_tokens", 0),
                    model_id=data.get("model_id", model_id),
                    request_id=data.get("request_id", ""),
                    status=data.get("status", "SUCCESS"),
                    dollar_cost=data.get("dollar_cost", 0.0),
                )
            except Exception:
                pass  # Corrupt cache entry — treat as miss.
        self.n_misses += 1
        return None

    def put(
        self,
        model_id: str,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        role: str,
        response: ModelResponse,
    ) -> None:
        """Cache a response."""
        if response.status != "SUCCESS":
            return  # Don't cache errors.
        key = self._make_key(model_id, system_prompt, messages, max_tokens, temperature, role)
        path = self.cache_dir / f"{key}.json"
        data = {
            "text": response.text,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "latency_ms": response.latency_ms,
            "confidence": response.confidence,
            "finish_reason": response.finish_reason,
            "error": response.error,
            "cached_tokens": response.cached_tokens,
            "model_id": response.model_id,
            "request_id": response.request_id,
            "status": response.status,
            "dollar_cost": response.dollar_cost,
            "total_tokens": response.total_tokens,
            "cached_at": time.time(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def summary(self) -> dict:
        return {
            "n_hits": self.n_hits,
            "n_misses": self.n_misses,
            "tokens_saved": self.tokens_saved,
            "dollars_saved": round(self.dollars_saved, 4),
            "hit_rate": round(self.n_hits / max(self.n_hits + self.n_misses, 1), 4),
        }

    def clear(self) -> None:
        """Clear the cache."""
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
        self.n_hits = 0
        self.n_misses = 0
        self.tokens_saved = 0
        self.dollars_saved = 0.0


class CachedBackend:
    """Wraps any backend with response caching.

    Only caches when temperature == 0 (deterministic).
    Topology-dependent calls have different context → different keys.
    """

    def __init__(self, backend, cache: ResponseCache) -> None:
        self._backend = backend
        self._cache = cache

    @property
    def config(self):
        return self._backend.config if hasattr(self._backend, "config") else None

    @property
    def budget(self):
        return self._backend.budget if hasattr(self._backend, "budget") else None

    @property
    def api_key_present(self):
        return getattr(self._backend, "api_key_present", False)

    def generate(
        self,
        *,
        role: str,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        # Only cache deterministic calls.
        if temperature == 0.0:
            model_id = getattr(self._backend, "config", None)
            model_id = model_id.model_id if model_id else "unknown"
            cached = self._cache.get(model_id, system_prompt, messages, max_tokens, temperature, role)
            if cached is not None:
                return cached

        response = self._backend.generate(
            role=role,
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if temperature == 0.0 and response.status == "SUCCESS":
            model_id = getattr(self._backend, "config", None)
            model_id = model_id.model_id if model_id else "unknown"
            self._cache.put(model_id, system_prompt, messages, max_tokens, temperature, role, response)

        return response

    def get_provenance(self) -> dict:
        if hasattr(self._backend, "get_provenance"):
            prov = self._backend.get_provenance()
            prov["cache_summary"] = self._cache.summary()
            return prov
        return {"cache_summary": self._cache.summary()}

    def verify_model(self):
        if hasattr(self._backend, "verify_model"):
            return self._backend.verify_model()
        return True, "cached"
