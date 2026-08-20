"""Benchmark task set for exp7.

Multiple task classes to ensure one fixed topology isn't already
optimal for a narrow task type.

Classes:
  - simple_factual: direct lookup
  - research_synthesis: multi-source synthesis
  - coding_debugging: code analysis and fix
  - multi_step_reasoning: chained logic
  - verification_sensitive: requires careful checking
  - memory_dependent: requires context retrieval
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import random


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    task_id: str
    task_class: str
    input: str
    expected_output: str = ""
    difficulty: float = 0.5  # 0=easy, 1=hard
    requires_memory: bool = False
    requires_verification: bool = False


def generate_benchmark(
    n_per_class: int = 10,
    seed: int = 42,
) -> list[BenchmarkTask]:
    """Generate the benchmark task set."""
    rng = random.Random(seed)
    tasks = []

    # Simple factual tasks.
    for i in range(n_per_class):
        tasks.append(BenchmarkTask(
            task_id=f"simple_factual_{i}",
            task_class="simple_factual",
            input=f"What is the capital of country {i+1}?",
            expected_output=f"Capital city of country {i+1}",
            difficulty=0.2,
        ))

    # Research/synthesis tasks.
    for i in range(n_per_class):
        tasks.append(BenchmarkTask(
            task_id=f"research_synthesis_{i}",
            task_class="research_synthesis",
            input=f"Synthesize information about topic {i+1} from multiple sources.",
            expected_output=f"Comprehensive summary of topic {i+1}",
            difficulty=0.7,
        ))

    # Coding/debugging tasks.
    for i in range(n_per_class):
        tasks.append(BenchmarkTask(
            task_id=f"coding_debugging_{i}",
            task_class="coding_debugging",
            input=f"Debug the following code snippet {i+1}: function has a bug.",
            expected_output=f"Fixed code with explanation",
            difficulty=0.6,
        ))

    # Multi-step reasoning tasks.
    for i in range(n_per_class):
        tasks.append(BenchmarkTask(
            task_id=f"multi_step_reasoning_{i}",
            task_class="multi_step_reasoning",
            input=f"Solve this multi-step problem {i+1}: requires 3 logical steps.",
            expected_output=f"Step-by-step solution",
            difficulty=0.8,
        ))

    # Verification-sensitive tasks.
    for i in range(n_per_class):
        tasks.append(BenchmarkTask(
            task_id=f"verification_sensitive_{i}",
            task_class="verification_sensitive",
            input=f"Verify this claim {i+1}: check all assumptions carefully.",
            expected_output=f"Verified or refuted with evidence",
            difficulty=0.5,
            requires_verification=True,
        ))

    # Memory-dependent tasks.
    for i in range(n_per_class):
        tasks.append(BenchmarkTask(
            task_id=f"memory_dependent_{i}",
            task_class="memory_dependent",
            input=f"Recall and use context from previous discussion {i+1}.",
            expected_output=f"Context-aware response",
            difficulty=0.5,
            requires_memory=True,
        ))

    rng.shuffle(tasks)
    return tasks


def evaluate_quality(result, task: BenchmarkTask) -> float:
    """Evaluate the quality of a task result.

    For the mock LLM, quality is based on:
      - success (did the pipeline complete?)
      - confidence (node confidence in output)
      - verification (did the verifier pass?)
      - length (is the output non-trivial?)
    """
    if not result.success:
        return 0.0

    quality = result.quality_score  # base confidence

    # Bonus for verification pass.
    if task.requires_verification:
        # Check if verifier was in the execution trace.
        has_verifier = any("verifier" in t for t in result.execution_trace)
        if has_verifier:
            quality = min(1.0, quality + 0.1)

    # Bonus for memory usage.
    if task.requires_memory:
        has_memory = any("memory" in t for t in result.execution_trace)
        if has_memory:
            quality = min(1.0, quality + 0.1)

    # Penalty for excessive failures.
    if result.total_failures > 0:
        quality = max(0.0, quality - 0.1 * result.total_failures)

    # Penalty for very short outputs.
    if len(result.output) < 20:
        quality = max(0.0, quality - 0.2)

    return quality


TASK_CLASSES = [
    "simple_factual",
    "research_synthesis",
    "coding_debugging",
    "multi_step_reasoning",
    "verification_sensitive",
    "memory_dependent",
]
