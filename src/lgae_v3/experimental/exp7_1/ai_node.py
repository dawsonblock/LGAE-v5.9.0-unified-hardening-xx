"""AI Node abstraction for exp7.

Each node has a fixed role, prompt template, and model config.
LGAE controls only the routing graph — not the nodes themselves.

Roles:
  PLANNER: decomposes tasks into subtasks
  WORKER: executes subtasks
  CRITIC: evaluates outputs
  VERIFIER: validates correctness
  MEMORY: stores/retrieves context

Telemetry per node:
  tokens, latency, success/failure, confidence, tool_calls, verification_outcome
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
import time
import hashlib


class NodeRole(str, Enum):
    PLANNER = "planner"
    WORKER = "worker"
    CRITIC = "critic"
    VERIFIER = "verifier"
    MEMORY = "memory"


@dataclass
class NodeTelemetry:
    """Telemetry emitted by a node invocation."""
    node_id: str
    role: NodeRole
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    success: bool = True
    confidence: float = 0.0
    tool_calls: int = 0
    verification_outcome: Optional[str] = None  # "pass", "fail", None
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


@dataclass
class AINode:
    """A role-based AI node in the execution topology.

    The node has a fixed role and prompt template. LGAE does not
    modify the node's behavior — only the routing between nodes.
    """
    node_id: str
    role: NodeRole
    prompt_template: str
    model_config: dict = field(default_factory=lambda: {
        "provider": "mock",
        "model": "mock-v1",
        "temperature": 0.0,
        "max_tokens": 1024,
    })

    # The LLM backend — pluggable. Mock by default.
    _llm_call: Optional[Callable] = None

    def invoke(
        self,
        task_input: str,
        context: str = "",
        tools: list = None,
    ) -> tuple[str, NodeTelemetry]:
        """Invoke this node on the given input.

        Returns (output_text, telemetry).
        """
        t0 = time.time()

        # Build the prompt from template.
        prompt = self.prompt_template.format(
            input=task_input,
            context=context,
            role=self.role.value,
        )

        # Call the LLM (or mock).
        if self._llm_call is not None:
            output, tokens_in, tokens_out, confidence = self._llm_call(
                prompt, self.model_config,
            )
        else:
            output, tokens_in, tokens_out, confidence = _mock_llm_call(
                prompt, self.model_config, self.role,
            )

        latency_ms = (time.time() - t0) * 1000

        # Determine success and verification outcome based on role.
        success = True
        verification_outcome = None
        if self.role == NodeRole.VERIFIER:
            verification_outcome = "pass" if "PASS" in output.upper() else "fail"
            success = verification_outcome == "pass"
        elif self.role == NodeRole.CRITIC:
            success = "GOOD" in output.upper() or "ACCEPT" in output.upper()

        telemetry = NodeTelemetry(
            node_id=self.node_id,
            role=self.role,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            success=success,
            confidence=confidence,
            tool_calls=len(tools) if tools else 0,
            verification_outcome=verification_outcome,
        )

        return output, telemetry


def _mock_llm_call(
    prompt: str,
    model_config: dict,
    role: NodeRole,
) -> tuple[str, int, int, float]:
    """Mock LLM call for testing without API access.

    Simulates realistic token counts and latency based on role.
    The output quality depends on the role and input.
    """
    # Token counts based on prompt length.
    tokens_in = max(10, len(prompt) // 4)
    tokens_out = max(20, min(512, tokens_in // 2))

    # Simulate latency based on output size.
    # Mock: 10ms per token, minimum 50ms.
    latency = max(50, tokens_out * 10)

    # Generate role-appropriate output.
    if role == NodeRole.PLANNER:
        output = f"PLAN: decompose into 3 subtasks\n1. analyze\n2. execute\n3. verify"
        confidence = 0.85
    elif role == NodeRole.WORKER:
        output = f"RESULT: processed input successfully\noutput: {prompt[:100]}..."
        confidence = 0.75
    elif role == NodeRole.CRITIC:
        output = f"REVIEW: GOOD - output meets quality standards\nACCEPT"
        confidence = 0.80
    elif role == NodeRole.VERIFIER:
        output = f"VERIFICATION: PASS - all checks succeeded"
        confidence = 0.90
    elif role == NodeRole.MEMORY:
        output = f"MEMORY: retrieved 3 relevant items\ncontext: {prompt[:80]}..."
        confidence = 0.70
    else:
        output = f"OUTPUT: {prompt[:100]}..."
        confidence = 0.50

    # Simulate occasional failures (5% rate).
    import random
    rng = random.Random(hash(prompt) % 2**31)
    if rng.random() < 0.05:
        output = f"ERROR: simulated failure for {role.value}"
        confidence = 0.0

    return output, tokens_in, tokens_out, confidence


def create_default_nodes() -> dict[str, AINode]:
    """Create the default 5-node topology."""
    return {
        "planner": AINode(
            node_id="planner",
            role=NodeRole.PLANNER,
            prompt_template=(
                "You are a Planner. Decompose the following task into subtasks.\n"
                "Task: {input}\n"
                "Context: {context}\n"
                "Provide a clear plan with 2-4 subtasks."
            ),
        ),
        "worker": AINode(
            node_id="worker",
            role=NodeRole.WORKER,
            prompt_template=(
                "You are a Worker. Execute the following subtask.\n"
                "Subtask: {input}\n"
                "Context: {context}\n"
                "Provide a complete solution."
            ),
        ),
        "critic": AINode(
            node_id="critic",
            role=NodeRole.CRITIC,
            prompt_template=(
                "You are a Critic. Evaluate the following output.\n"
                "Output: {input}\n"
                "Context: {context}\n"
                "Rate as GOOD or BAD with explanation."
            ),
        ),
        "verifier": AINode(
            node_id="verifier",
            role=NodeRole.VERIFIER,
            prompt_template=(
                "You are a Verifier. Check the following output for correctness.\n"
                "Output: {input}\n"
                "Context: {context}\n"
                "Return PASS or FAIL with reasoning."
            ),
        ),
        "memory": AINode(
            node_id="memory",
            role=NodeRole.MEMORY,
            prompt_template=(
                "You are a Memory node. Retrieve relevant context for the task.\n"
                "Task: {input}\n"
                "Provide relevant stored context."
            ),
        ),
    }
