"""Baseline competition framework (Phase 23).

Every learned policy must compete against strong algorithmic baselines:

  NO_OP, random, degree heuristic, Forman, effective resistance, FoSR,
  spectral heuristic, and oracle on tractable small graphs.

For every state: ``regret(a) = U(oracle) - U(a)``. Primary metrics are the
regret distribution: mean, median, P90, P99, and catastrophic-regret
frequency. Accuracy alone is insufficient.

This builds on the existing ``exact_candidate_deltas`` / ``candidate_regret``
oracle infrastructure. It does not re-implement the baselines; it scores any
policy that produces a ranking over a candidate set against the exact oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..executive import StructuralAction
from ..reasoning import ConcreteAction
from ..structural_intelligence import exact_candidate_deltas, RegretResult, candidate_regret


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((p / 100.0) * (len(s) - 1)))
    return float(s[idx])


@dataclass(frozen=True, slots=True)
class PolicyResult:
    """One policy's outcome on one state."""
    policy: str
    chosen_index: int
    chosen_delta: float
    regret: float

    @property
    def is_oracle(self) -> bool:
        return self.policy == "oracle"


@dataclass(slots=True)
class CompetitionReport:
    """Aggregated regret distribution across many states, per policy."""
    regrets: dict[str, list[float]] = field(default_factory=dict)

    def add(self, policy: str, regret: float) -> None:
        self.regrets.setdefault(policy, []).append(float(regret))

    def summary(self, *, catastrophic_threshold: float = 0.5) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for policy, regs in self.regrets.items():
            n = len(regs)
            if n == 0:
                out[policy] = {"count": 0}
                continue
            out[policy] = {
                "count": n,
                "mean_regret": sum(regs) / n,
                "median_regret": _percentile(regs, 50),
                "p90_regret": _percentile(regs, 90),
                "p99_regret": _percentile(regs, 99),
                "catastrophic_regret_frequency": sum(1 for r in regs if r >= catastrophic_threshold) / n,
            }
        return out

    def to_log(self) -> dict[str, Any]:
        return self.summary()


def select_by_scores(scores: Sequence[float]) -> int:
    """Return the index of the highest score (deterministic tie-break by index)."""
    if not scores:
        return -1
    return max(range(len(scores)), key=lambda i: (float(scores[i]), -i))


class BaselineCompetition:
    """Run a learned policy against a set of baseline policies over candidate sets.

    Each policy is a callable ``(candidates, exact_deltas) -> chosen_index``.
    The oracle is always included and selects ``argmax(exact_deltas)``. The
    learned policy supplies its own predicted scores. Baselines may use
    per-candidate prior scores (e.g. Forman/FoSR/ER channel scores) or
    structural heuristics.
    """

    def __init__(self, *, catastrophic_threshold: float = 0.5) -> None:
        self.catastrophic_threshold = float(catastrophic_threshold)
        self.report = CompetitionReport()
        self._policies: dict[str, Callable[[Sequence[ConcreteAction], Sequence[float]], int]] = {
            "no_op": self._no_op_policy,
            "oracle": self._oracle_policy,
        }

    def register_policy(
        self, name: str, fn: Callable[[Sequence[ConcreteAction], Sequence[float]], int]
    ) -> None:
        self._policies[name] = fn

    @staticmethod
    def _oracle_policy(candidates: Sequence[ConcreteAction], exact_deltas: Sequence[float]) -> int:
        return select_by_scores(exact_deltas)

    @staticmethod
    def _no_op_policy(candidates: Sequence[ConcreteAction], exact_deltas: Sequence[float]) -> int:
        for i, c in enumerate(candidates):
            if c.action == StructuralAction.NO_OP:
                return i
        return -1

    def evaluate_state(
        self,
        candidates: Sequence[ConcreteAction],
        exact_deltas: Sequence[float],
        *,
        learned_scores: Sequence[float] | None = None,
        extra_policies: dict[str, Callable[[Sequence[ConcreteAction], Sequence[float]], int]] | None = None,
    ) -> dict[str, PolicyResult]:
        """Evaluate all policies on one state and record regrets into the report."""
        if len(candidates) != len(exact_deltas) or not candidates:
            raise ValueError("candidates and exact_deltas must be non-empty and equal length")
        oracle_idx = select_by_scores(exact_deltas)
        oracle_delta = float(exact_deltas[oracle_idx])
        results: dict[str, PolicyResult] = {}
        policies = dict(self._policies)
        if learned_scores is not None:
            if len(learned_scores) != len(candidates):
                raise ValueError("learned_scores length must match candidates")
            policies["learned"] = lambda c, d: select_by_scores(learned_scores)
        if extra_policies:
            policies.update(extra_policies)
        for name, fn in policies.items():
            idx = fn(candidates, exact_deltas)
            if idx < 0:
                continue
            chosen_delta = float(exact_deltas[idx])
            regret = max(0.0, oracle_delta - chosen_delta)
            results[name] = PolicyResult(name, idx, chosen_delta, regret)
            self.report.add(name, regret)
        return results

    def summary(self) -> dict[str, Any]:
        return self.report.summary(catastrophic_threshold=self.catastrophic_threshold)


def learned_policy_from_scores(scores: Sequence[float]) -> Callable[[Sequence[ConcreteAction], Sequence[float]], int]:
    """Build a policy callable that selects by a fixed learned score vector."""
    scores = list(scores)

    def fn(candidates: Sequence[ConcreteAction], exact_deltas: Sequence[float]) -> int:
        return select_by_scores(scores)

    return fn
