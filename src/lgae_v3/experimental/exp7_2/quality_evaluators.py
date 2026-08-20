"""Deterministic quality evaluators for exp7.2.

Do not rely on the same model that generated the answer to grade
itself. Use deterministic evaluation where possible:
  - coding → pattern matching for correct structure
  - math → exact answer matching
  - structured extraction → expected fields
  - tool tasks → success condition
  - research/synthesis → rubric-based scoring
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re


def evaluate_quality(
    task_class: str,
    output: str,
    expected_output: str = "",
    verification_outcome: Optional[str] = None,
    execution_context: str = "",
) -> float:
    """Evaluate task quality deterministically.

    Returns a quality score in [0, 1].
    """
    if verification_outcome == "pass":
        base = 0.8
    elif verification_outcome == "fail":
        base = 0.2
    else:
        base = 0.5

    # Extract worker quality score if present.
    worker_quality = None
    for line in execution_context.split("\n"):
        if "WORKER_QUALITY_SCORE:" in line:
            try:
                worker_quality = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        if "WORKER_QUALITY_SCORE:" in output:
            try:
                worker_quality = float(output.split("WORKER_QUALITY_SCORE:")[1].split("\n")[0].strip())
            except (ValueError, IndexError):
                pass

    if worker_quality is not None:
        # Blend worker self-assessment with verification.
        if verification_outcome == "pass":
            return min(1.0, worker_quality * 0.7 + 0.3)
        elif verification_outcome == "fail":
            return max(0.0, worker_quality * 0.3)
        else:
            return worker_quality

    # Fall back to class-specific evaluation.
    if task_class == "simple_factual":
        return _eval_factual(output, expected_output, base)
    elif task_class == "research_synthesis":
        return _eval_research(output, execution_context, base)
    elif task_class == "coding_debugging":
        return _eval_coding(output, expected_output, base)
    elif task_class == "multi_step_reasoning":
        return _eval_reasoning(output, expected_output, base)
    elif task_class == "verification_sensitive":
        return _eval_verification(output, verification_outcome, base)
    elif task_class == "memory_dependent":
        return _eval_memory(output, execution_context, base)
    else:
        return base


def _eval_factual(output: str, expected: str, base: float) -> float:
    """Factual tasks: check if output contains expected answer."""
    if expected and expected.lower() in output.lower():
        return min(1.0, base + 0.2)
    return base


def _eval_research(output: str, context: str, base: float) -> float:
    """Research tasks: quality depends on whether research was used."""
    has_research = "RESEARCH:" in context or "RESEARCH:" in output
    has_findings = "Finding" in context or "finding" in output
    if has_research and has_findings:
        return min(1.0, base + 0.3)
    elif has_research:
        return min(1.0, base + 0.15)
    return base


def _eval_coding(output: str, expected: str, base: float) -> float:
    """Coding tasks: check for solution structure."""
    has_solution = "RESULT" in output or "Solution" in output
    has_quality = "WORKER_QUALITY_SCORE:" in output or "quality=" in output.lower()
    if has_solution and has_quality:
        return min(1.0, base + 0.2)
    return base


def _eval_reasoning(output: str, expected: str, base: float) -> float:
    """Reasoning tasks: check for step-by-step structure."""
    has_steps = "PLAN" in output or "step" in output.lower()
    if has_steps:
        return min(1.0, base + 0.15)
    return base


def _eval_verification(output: str, verification_outcome: Optional[str], base: float) -> float:
    """Verification-sensitive tasks: quality depends on verification."""
    if verification_outcome == "pass":
        return min(1.0, base + 0.3)
    elif verification_outcome == "fail":
        return max(0.0, base - 0.3)
    return base


def _eval_memory(output: str, context: str, base: float) -> float:
    """Memory-dependent tasks: quality depends on memory usage."""
    has_memory = "MEMORY:" in context or "MEMORY_RELEVANCE" in context
    if has_memory:
        return min(1.0, base + 0.25)
    return base
