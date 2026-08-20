"""Objective function for exp7.

J = w_Q * Q_task - λ_T * Tokens - λ_L * Latency - λ_F * Failures - λ_C * Calls

Weights are fixed before evaluation and not tuned after seeing results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ObjectiveWeights:
    """Fixed objective weights — set before evaluation, not tuned after."""
    w_quality: float = 1.0       # quality weight
    lambda_tokens: float = 0.001  # cost per token
    lambda_latency: float = 0.01  # cost per second
    lambda_failures: float = 0.5  # cost per failure
    lambda_calls: float = 0.05    # cost per LLM call

    def to_dict(self) -> dict:
        return {
            "w_quality": self.w_quality,
            "lambda_tokens": self.lambda_tokens,
            "lambda_latency": self.lambda_latency,
            "lambda_failures": self.lambda_failures,
            "lambda_calls": self.lambda_calls,
        }


def compute_objective(
    quality: float,
    tokens: int,
    latency_ms: float,
    failures: int,
    calls: int,
    weights: ObjectiveWeights,
) -> float:
    """Compute the objective J.

    J = w_Q * Q - λ_T * Tokens - λ_L * Latency_s - λ_F * Failures - λ_C * Calls
    """
    latency_s = latency_ms / 1000.0
    return (
        weights.w_quality * quality
        - weights.lambda_tokens * tokens
        - weights.lambda_latency * latency_s
        - weights.lambda_failures * failures
        - weights.lambda_calls * calls
    )


def compute_objective_from_result(result, weights: ObjectiveWeights) -> float:
    """Compute J from a TaskResult."""
    return compute_objective(
        quality=result.quality_score,
        tokens=result.total_tokens,
        latency_ms=result.total_latency_ms,
        failures=result.total_failures,
        calls=result.total_llm_calls,
        weights=weights,
    )


def compute_pareto_efficiency(
    results: list[dict],
    quality_key: str = "quality",
    cost_key: str = "cost",
) -> list[bool]:
    """Determine which results are Pareto efficient.

    A result is Pareto efficient if no other result has both
    higher quality AND lower cost.
    """
    n = len(results)
    is_efficient = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has >= quality and <= cost, with at least one strict.
            if (results[j][quality_key] >= results[i][quality_key]
                    and results[j][cost_key] <= results[i][cost_key]
                    and (results[j][quality_key] > results[i][quality_key]
                         or results[j][cost_key] < results[i][cost_key])):
                is_efficient[i] = False
                break
    return is_efficient
