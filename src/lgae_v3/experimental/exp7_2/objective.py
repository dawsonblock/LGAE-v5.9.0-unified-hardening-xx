"""Normalized objective function for exp7.2.

J = w_Q * Q
  - λ_T * (Tokens / T_budget)
  - λ_L * (Latency / L_budget)
  - λ_C * (Calls / C_budget)
  - λ_F * Failures

Budgets normalize raw counts so that tokens, latency, and calls
are on comparable scales. Weights are frozen before evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ObjectiveWeights:
    """Frozen objective weights with budget normalization."""
    w_quality: float = 1.0
    lambda_tokens: float = 0.3    # weight for normalized token cost
    lambda_latency: float = 0.2   # weight for normalized latency cost
    lambda_calls: float = 0.2     # weight for normalized call cost
    lambda_failures: float = 0.5  # weight per failure
    # Budgets for normalization (frozen before evaluation).
    token_budget: int = 2000      # expected tokens per task
    latency_budget_ms: float = 5000.0  # expected latency per task
    call_budget: int = 6          # expected LLM calls per task

    def to_dict(self) -> dict:
        return {
            "w_quality": self.w_quality,
            "lambda_tokens": self.lambda_tokens,
            "lambda_latency": self.lambda_latency,
            "lambda_calls": self.lambda_calls,
            "lambda_failures": self.lambda_failures,
            "token_budget": self.token_budget,
            "latency_budget_ms": self.latency_budget_ms,
            "call_budget": self.call_budget,
        }


def compute_objective(
    quality: float,
    tokens: int,
    latency_ms: float,
    failures: int,
    calls: int,
    weights: ObjectiveWeights,
) -> float:
    """Compute normalized objective J."""
    return (
        weights.w_quality * quality
        - weights.lambda_tokens * (tokens / weights.token_budget)
        - weights.lambda_latency * (latency_ms / weights.latency_budget_ms)
        - weights.lambda_calls * (calls / weights.call_budget)
        - weights.lambda_failures * failures
    )


def compute_objective_from_record(record, weights: ObjectiveWeights) -> float:
    """Compute J from a StructuralTransitionRecord."""
    return compute_objective(
        quality=record.final_quality,
        tokens=record.total_tokens,
        latency_ms=record.total_latency_ms,
        failures=record.total_failures,
        calls=record.total_llm_calls,
        weights=weights,
    )


def compute_quality_per_token(quality: float, tokens: int) -> float:
    """Quality-adjusted compute efficiency: Q / Tokens."""
    if tokens == 0:
        return 0.0
    return quality / tokens


def compute_quality_per_cost(quality: float, cost: float) -> float:
    """Quality per unit cost."""
    if cost == 0:
        return 0.0
    return quality / cost


def compute_pareto_efficiency(points: list[dict], quality_key: str = "quality", cost_key: str = "cost") -> list[bool]:
    """Determine Pareto-efficient points."""
    n = len(points)
    is_efficient = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (points[j][quality_key] >= points[i][quality_key]
                    and points[j][cost_key] <= points[i][cost_key]
                    and (points[j][quality_key] > points[i][quality_key]
                         or points[j][cost_key] < points[i][cost_key])):
                is_efficient[i] = False
                break
    return is_efficient
