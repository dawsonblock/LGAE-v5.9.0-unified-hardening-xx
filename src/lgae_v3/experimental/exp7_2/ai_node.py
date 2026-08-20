"""Topology-dependent AI nodes for exp7.2.

Key change from exp7.1: topology genuinely changes what context
each node receives. The runtime accumulates context from visited
nodes, so routing through Researcher before Worker gives Worker
different context than routing directly from Planner.

Nodes:
  Planner: decomposes tasks
  Worker: executes subtasks
  Researcher: gathers additional information (expensive optional path)
  Critic: evaluates outputs
  Verifier: validates correctness
  Memory: stores/retrieves context
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
import time

from .model_backend import ModelBackend, ModelResponse, Message, MockModelBackend


class NodeRole(str, Enum):
    PLANNER = "planner"
    WORKER = "worker"
    RESEARCHER = "researcher"
    CRITIC = "critic"
    VERIFIER = "verifier"
    MEMORY = "memory"


# System prompts per role — these are FIXED. LGAE does not change them.
SYSTEM_PROMPTS = {
    NodeRole.PLANNER: (
        "You are a Planner. Decompose the task into subtasks. "
        "Consider what information is needed and what can be executed directly. "
        "Output a clear plan with 2-4 steps."
    ),
    NodeRole.WORKER: (
        "You are a Worker. Execute the task using all available context. "
        "If research findings are provided, use them. If a plan is provided, follow it. "
        "Output a complete solution with a quality self-assessment."
    ),
    NodeRole.RESEARCHER: (
        "You are a Researcher. Gather relevant information for the task. "
        "Provide findings that will help the Worker produce a better solution. "
        "Output structured research findings with a quality score."
    ),
    NodeRole.CRITIC: (
        "You are a Critic. Evaluate the Worker's output for quality and completeness. "
        "If the quality is below standard, reject and explain why. "
        "Output GOOD or BAD with reasoning."
    ),
    NodeRole.VERIFIER: (
        "You are a Verifier. Check the final output for correctness. "
        "Apply strict validation criteria. Output PASS or FAIL."
    ),
    NodeRole.MEMORY: (
        "You are a Memory node. Retrieve relevant stored context for the task. "
        "Output the most relevant items from memory."
    ),
}


@dataclass
class NodeTelemetry:
    """Telemetry from a single node invocation."""
    node_id: str
    role: NodeRole
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    success: bool = True
    confidence: float = 0.0
    tool_calls: int = 0
    verification_outcome: Optional[str] = None
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


@dataclass
class AINode:
    """A role-based AI node with a fixed system prompt.

    The node's behavior is fixed. LGAE controls only whether this
    node is reached (via topology edges), not what it does.
    """
    node_id: str
    role: NodeRole
    system_prompt: str = ""
    max_tokens: int = 1024
    temperature: float = 0.0

    def __post_init__(self):
        if not self.system_prompt:
            self.system_prompt = SYSTEM_PROMPTS.get(self.role, "You are an AI assistant.")

    def invoke(
        self,
        task_input: str,
        accumulated_context: str,
        backend: ModelBackend,
    ) -> tuple[str, NodeTelemetry]:
        """Invoke this node.

        The accumulated_context contains outputs from all upstream
        nodes that were visited before this one. This is what makes
        topology matter: different routes produce different context.
        """
        # Build messages from accumulated context.
        messages = [Message(role="user", content=task_input)]
        if accumulated_context:
            messages.append(Message(role="assistant", content=accumulated_context))
            messages.append(Message(
                role="user",
                content=f"Based on the above context, perform your role as {self.role.value}."
            ))

        response = backend.generate(
            role=self.role.value,
            system_prompt=self.system_prompt,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        # Determine success and verification outcome.
        success = response.success
        verification_outcome = None
        if self.role == NodeRole.VERIFIER:
            if "PASS" in response.text.upper():
                verification_outcome = "pass"
            elif "FAIL" in response.text.upper():
                verification_outcome = "fail"
                success = False
        elif self.role == NodeRole.CRITIC:
            if "BAD" in response.text.upper() or "REJECT" in response.text.upper():
                success = False

        telemetry = NodeTelemetry(
            node_id=self.node_id,
            role=self.role,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            latency_ms=response.latency_ms,
            success=success,
            confidence=response.confidence,
            verification_outcome=verification_outcome,
            error=response.error,
        )

        return response.text, telemetry


def create_default_nodes() -> dict[str, AINode]:
    """Create the default 6-node topology."""
    return {
        "planner": AINode(node_id="planner", role=NodeRole.PLANNER),
        "worker": AINode(node_id="worker", role=NodeRole.WORKER),
        "researcher": AINode(node_id="researcher", role=NodeRole.RESEARCHER),
        "critic": AINode(node_id="critic", role=NodeRole.CRITIC),
        "verifier": AINode(node_id="verifier", role=NodeRole.VERIFIER),
        "memory": AINode(node_id="memory", role=NodeRole.MEMORY),
    }
