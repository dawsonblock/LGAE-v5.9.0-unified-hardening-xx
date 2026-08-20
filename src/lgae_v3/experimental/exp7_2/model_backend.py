"""ModelBackend protocol for exp7.2.

Pluggable LLM backend interface. Any backend (real API, local model,
mock) that implements this protocol can be used by the AI topology.

The key design principle: topology changes what context each node
receives, which changes the prompt, which changes the model's output.
The backend is just the execution engine — the topology controls
cognition by controlling information flow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional, runtime_checkable
import time
import random
import hashlib


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ModelResponse:
    """Response from a model backend."""
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    confidence: float = 0.0
    finish_reason: str = "stop"
    error: Optional[str] = None
    # Extended fields for exp7.5 (optional, backward-compatible)
    cached_tokens: int = 0
    model_id: str = ""
    request_id: str = ""
    status: str = "SUCCESS"
    dollar_cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def success(self) -> bool:
        return self.error is None and self.status == "SUCCESS"


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol for LLM backends.

    Any backend that implements this can be plugged into the topology.
    The topology controls what context reaches the model; the backend
    just generates text from that context.
    """

    def generate(
        self,
        *,
        role: str,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """Generate a response from the model.

        Args:
            role: The node role (planner, worker, researcher, critic, verifier, memory)
            system_prompt: System prompt for this node
            messages: Conversation history (what the node "sees" from upstream)
            max_tokens: Maximum output tokens
            temperature: Sampling temperature

        Returns:
            ModelResponse with text, token counts, latency, confidence
        """
        ...


class MockModelBackend:
    """Topology-sensitive mock backend.

    Unlike exp7.1's mock, this backend produces different outputs
    depending on:
      - The role (different output formats)
      - The messages (different context → different output)
      - Whether research/criticism/verification context is present

    This makes topology matter: routing through Researcher adds
    research context that changes Worker output; routing through
    Critic adds critique that changes the final quality.
    """

    def __init__(self, seed: int = 42, failure_rate: float = 0.03) -> None:
        self.seed = seed
        self.failure_rate = failure_rate
        self._call_count = 0

    def generate(
        self,
        *,
        role: str,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        self._call_count += 1
        t0 = time.time()

        # Build the full context from messages.
        context = "\n".join(m.content for m in messages if m.role == "user")
        context_len = len(context)
        has_research = "research" in context.lower() or "RESEARCH" in context
        has_critique = "critique" in context.lower() or "CRITIC" in context or "REVIEW" in context.upper()
        has_memory = "memory" in context.lower() or "MEMORY" in context
        has_plan = "plan" in context.lower() or "PLAN" in context

        # Token counts based on input + output.
        tokens_in = max(20, len(system_prompt) // 4 + context_len // 4)

        # Generate role-appropriate output that depends on context.
        rng = random.Random(hash((context, role, self.seed)) % 2**31)

        # Check for simulated failure.
        if rng.random() < self.failure_rate:
            latency = max(50, tokens_in * 5)
            return ModelResponse(
                text=f"ERROR: simulated failure in {role}",
                tokens_in=tokens_in,
                tokens_out=10,
                latency_ms=latency,
                confidence=0.0,
                finish_reason="error",
                error="simulated_failure",
            )

        if role == "planner":
            n_subtasks = 3 if has_memory else 2
            output = f"PLAN: decompose into {n_subtasks} subtasks\n"
            for i in range(n_subtasks):
                output += f"{i+1}. {'analyze context' if has_research else 'analyze problem'}\n"
            output += f"{n_subtasks}. execute solution\n"
            output += f"{n_subtasks+1}. verify result"
            tokens_out = max(50, len(output) // 4)
            confidence = 0.85

        elif role == "researcher":
            # Researcher adds factual context that improves worker quality.
            research_quality = rng.random()  # 0-1 quality of research
            findings = []
            for i in range(3):
                findings.append(f"Finding {i+1}: relevant fact about {context[:50]}")
            output = f"RESEARCH: {len(findings)} findings (quality={research_quality:.2f})\n"
            output += "\n".join(findings)
            output += f"\nRESEARCH_QUALITY_SCORE: {research_quality:.4f}"
            tokens_out = max(100, len(output) // 4)
            confidence = 0.5 + research_quality * 0.4

        elif role == "worker":
            # Worker quality depends on what context it received.
            base_quality = 0.5
            if has_research:
                base_quality += 0.2  # research improves worker output
            if has_plan:
                base_quality += 0.1  # planning improves worker output
            if has_memory:
                base_quality += 0.1  # memory context helps

            # Add some noise.
            base_quality += rng.gauss(0, 0.05)
            base_quality = max(0.1, min(0.95, base_quality))

            output = f"RESULT: processed input (quality={base_quality:.4f})\n"
            output += f"WORKER_QUALITY_SCORE: {base_quality:.4f}\n"
            output += f"Solution based on {'research-informed' if has_research else 'direct'} approach"
            tokens_out = max(80, len(output) // 4)
            confidence = base_quality

        elif role == "critic":
            # Critic evaluates the worker output.
            # If worker quality is in the context, critic can catch low quality.
            worker_quality = 0.5
            for line in context.split("\n"):
                if "WORKER_QUALITY_SCORE:" in line:
                    try:
                        worker_quality = float(line.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass

            # Critic is more likely to flag low quality.
            if worker_quality < 0.6:
                verdict = "BAD"
                output = f"REVIEW: BAD - quality score {worker_quality:.2f} is below threshold\nREJECT"
            else:
                verdict = "GOOD"
                output = f"REVIEW: GOOD - quality score {worker_quality:.2f} is acceptable\nACCEPT"

            tokens_out = max(40, len(output) // 4)
            confidence = 0.8

        elif role == "verifier":
            # Verifier checks correctness.
            # If worker quality is high, pass; if low, fail.
            worker_quality = 0.5
            has_critic_review = has_critique

            for line in context.split("\n"):
                if "WORKER_QUALITY_SCORE:" in line:
                    try:
                        worker_quality = float(line.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass

            # Verifier is stricter if critic reviewed (catches more errors).
            threshold = 0.5 if has_critic_review else 0.3

            if worker_quality >= threshold:
                output = f"VERIFICATION: PASS - quality {worker_quality:.2f} >= {threshold}"
                confidence = 0.9
            else:
                output = f"VERIFICATION: FAIL - quality {worker_quality:.2f} < {threshold}"
                confidence = 0.85

            tokens_out = max(30, len(output) // 4)

        elif role == "memory":
            # Memory provides stored context.
            n_items = rng.randint(1, 5)
            output = f"MEMORY: retrieved {n_items} relevant items\n"
            for i in range(n_items):
                output += f"  item {i+1}: stored context related to {context[:40]}\n"
            output += "MEMORY_RELEVANCE: high"
            tokens_out = max(60, len(output) // 4)
            confidence = 0.7

        else:
            output = f"OUTPUT: {context[:100]}"
            tokens_out = max(30, len(output) // 4)
            confidence = 0.5

        latency = max(50, (tokens_in + tokens_out) * 8)

        return ModelResponse(
            text=output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency,
            confidence=confidence,
            finish_reason="stop",
        )


class OpenAIBackend:
    """OpenAI API backend.

    Requires OPENAI_API_KEY environment variable.
    Falls back to mock if not available.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or __import__("os").environ.get("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("openai package not installed")
        return self._client

    def generate(
        self,
        *,
        role: str,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        if not self.api_key:
            # Fall back to mock.
            return MockModelBackend().generate(
                role=role, system_prompt=system_prompt,
                messages=messages, max_tokens=max_tokens, temperature=temperature,
            )

        t0 = time.time()
        try:
            client = self._get_client()
            full_messages = [{"role": "system", "content": system_prompt}]
            full_messages.extend({"role": m.role, "content": m.content} for m in messages)

            response = client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            latency = (time.time() - t0) * 1000
            return ModelResponse(
                text=response.choices[0].message.content,
                tokens_in=response.usage.prompt_tokens,
                tokens_out=response.usage.completion_tokens,
                latency_ms=latency,
                confidence=0.8,  # real models don't report confidence
                finish_reason=response.choices[0].finish_reason,
            )
        except Exception as e:
            return ModelResponse(
                text="",
                tokens_in=0, tokens_out=0,
                latency_ms=(time.time() - t0) * 1000,
                error=str(e),
            )


def create_backend(backend_type: str = "mock", **kwargs) -> ModelBackend:
    """Create a model backend by type."""
    if backend_type == "mock":
        return MockModelBackend(**kwargs)
    elif backend_type == "openai":
        return OpenAIBackend(**kwargs)
    else:
        return MockModelBackend(**kwargs)
