"""Benchmark task set for exp7.2.

6 task families × 50-100 tasks = 300-600 tasks.

Families:
  - simple_factual: direct lookup
  - research_synthesis: benefits from Researcher node
  - coding_debugging: benefits from Critic feedback loop
  - multi_step_reasoning: benefits from Planner decomposition
  - verification_sensitive: benefits from Verifier
  - memory_dependent: benefits from Memory node

Each task has a difficulty and flags indicating which nodes help.
LGAE does NOT see these flags — it only sees telemetry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random


@dataclass
class BenchmarkTask:
    task_id: str
    task_class: str
    input: str
    expected_output: str = ""
    difficulty: float = 0.5
    # These flags are NOT visible to LGAE — only to the rule-based router.
    benefits_from_research: bool = False
    benefits_from_critic: bool = False
    benefits_from_verification: bool = False
    benefits_from_memory: bool = False
    benefits_from_planning: bool = False


TASK_CLASSES = [
    "simple_factual",
    "research_synthesis",
    "coding_debugging",
    "multi_step_reasoning",
    "verification_sensitive",
    "memory_dependent",
]


def generate_benchmark(
    n_per_class: int = 50,
    seed: int = 42,
) -> list[BenchmarkTask]:
    """Generate the benchmark task set."""
    rng = random.Random(seed)
    tasks = []

    # Simple factual tasks — don't need research/critic, just worker+verifier.
    for i in range(n_per_class):
        tasks.append(BenchmarkTask(
            task_id=f"simple_factual_{i}",
            task_class="simple_factual",
            input=f"What is the capital of country {i+1}?",
            expected_output=f"Capital city of country {i+1}",
            difficulty=0.2,
            benefits_from_verification=True,
        ))

    # Research/synthesis tasks — benefit from Researcher node.
    for i in range(n_per_class):
        tasks.append(BenchmarkTask(
            task_id=f"research_synthesis_{i}",
            task_class="research_synthesis",
            input=f"Synthesize information about topic {i+1} from multiple sources.",
            expected_output=f"Comprehensive summary of topic {i+1}",
            difficulty=0.7,
            benefits_from_research=True,
            benefits_from_planning=True,
        ))

    # Coding/debugging tasks — benefit from Critic feedback loop.
    for i in range(n_per_class):
        tasks.append(BenchmarkTask(
            task_id=f"coding_debugging_{i}",
            task_class="coding_debugging",
            input=f"Debug code snippet {i+1}: function has a subtle bug.",
            expected_output=f"Fixed code with explanation",
            difficulty=0.6,
            benefits_from_critic=True,
            benefits_from_verification=True,
        ))

    # Multi-step reasoning tasks — benefit from Planner.
    for i in range(n_per_class):
        tasks.append(BenchmarkTask(
            task_id=f"multi_step_reasoning_{i}",
            task_class="multi_step_reasoning",
            input=f"Solve multi-step problem {i+1}: requires 3 logical steps.",
            expected_output=f"Step-by-step solution",
            difficulty=0.8,
            benefits_from_planning=True,
            benefits_from_verification=True,
        ))

    # Verification-sensitive tasks — benefit from Verifier.
    for i in range(n_per_class):
        tasks.append(BenchmarkTask(
            task_id=f"verification_sensitive_{i}",
            task_class="verification_sensitive",
            input=f"Verify claim {i+1}: check all assumptions carefully.",
            expected_output=f"Verified or refuted with evidence",
            difficulty=0.5,
            benefits_from_verification=True,
            benefits_from_critic=True,
        ))

    # Memory-dependent tasks — benefit from Memory node.
    for i in range(n_per_class):
        tasks.append(BenchmarkTask(
            task_id=f"memory_dependent_{i}",
            task_class="memory_dependent",
            input=f"Recall and use context from previous discussion {i+1}.",
            expected_output=f"Context-aware response",
            difficulty=0.5,
            benefits_from_memory=True,
        ))

    rng.shuffle(tasks)
    return tasks
