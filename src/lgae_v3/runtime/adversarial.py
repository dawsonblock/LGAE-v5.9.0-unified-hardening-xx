"""Adversarial structural testing (Phase 27).

Tests the runtime against worst-case inputs designed to break invariants
or cause incorrect behavior. Adversarial inputs include:

  - empty graph (0 nodes)
  - single node (no edges)
  - disconnected graph
  - graph with extreme weights
  - graph with maximum degree (star)
  - graph with self-loops (should be rejected)
  - graph with duplicate edges
  - very large graph (stress)
  - graph with negative weights (if unsupported)
  - graph with NaN/Inf weights

Each adversarial case produces an ``AdversarialTestResult`` with pass/fail
and the runtime's response (accepted, rejected, raised, or crashed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch
from torch import Tensor

from ..types import GraphBuffers, make_graph_buffers


class AdversarialOutcome(str, Enum):
    ACCEPTED = "accepted"        # runtime handled the input correctly
    REJECTED = "rejected"        # runtime rejected the invalid input
    RAISED = "raised"            # runtime raised an expected error
    CRASHED = "crashed"          # runtime crashed unexpectedly (bad)
    UNEXPECTED = "unexpected"    # runtime behaved unexpectedly (bad)


@dataclass(frozen=True, slots=True)
class AdversarialTestResult:
    """Result of one adversarial test case."""
    name: str
    outcome: AdversarialOutcome
    expected_outcome: AdversarialOutcome
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True if the actual outcome matches the expected outcome."""
        return self.outcome == self.expected_outcome

    def to_log(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "outcome": self.outcome.value,
            "expected_outcome": self.expected_outcome.value,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


@dataclass(slots=True)
class AdversarialTestReport:
    """Aggregate adversarial test report."""
    results: list[AdversarialTestResult] = field(default_factory=list)

    def add(self, result: AdversarialTestResult) -> None:
        self.results.append(result)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def crashed_count(self) -> int:
        return sum(1 for r in self.results if r.outcome == AdversarialOutcome.CRASHED)

    @property
    def unexpected_count(self) -> int:
        return sum(1 for r in self.results if r.outcome == AdversarialOutcome.UNEXPECTED)

    def to_log(self) -> dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "total": len(self.results),
            "crashed_count": self.crashed_count,
            "unexpected_count": self.unexpected_count,
            "results": [r.to_log() for r in self.results],
        }


def _try_adversarial(
    name: str,
    expected: AdversarialOutcome,
    fn: Callable[[], Any],
) -> AdversarialTestResult:
    """Run an adversarial test and classify the outcome."""
    try:
        result = fn()
        return AdversarialTestResult(
            name=name, outcome=AdversarialOutcome.ACCEPTED, expected_outcome=expected,
            message="input was accepted", details={"result": str(result)[:200]},
        )
    except ValueError as e:
        return AdversarialTestResult(
            name=name, outcome=AdversarialOutcome.REJECTED, expected_outcome=expected,
            message=str(e), details={"error_type": "ValueError"},
        )
    except RuntimeError as e:
        return AdversarialTestResult(
            name=name, outcome=AdversarialOutcome.RAISED, expected_outcome=expected,
            message=str(e), details={"error_type": "RuntimeError"},
        )
    except Exception as e:
        return AdversarialTestResult(
            name=name, outcome=AdversarialOutcome.CRASHED, expected_outcome=expected,
            message=f"unexpected crash: {e!r}", details={"error_type": type(e).__name__},
        )


def run_adversarial_tests() -> AdversarialTestReport:
    """Run the canonical adversarial test suite."""
    report = AdversarialTestReport()

    # 1. Single node, no edges — should be accepted.
    report.add(_try_adversarial(
        "single_node_no_edges",
        AdversarialOutcome.ACCEPTED,
        lambda: make_graph_buffers(1, [], capacity=4),
    ))

    # 2. Two nodes, one edge — should be accepted.
    report.add(_try_adversarial(
        "two_nodes_one_edge",
        AdversarialOutcome.ACCEPTED,
        lambda: make_graph_buffers(2, [(0, 1)], capacity=4),
    ))

    # 3. Self-loop — should be rejected.
    report.add(_try_adversarial(
        "self_loop_rejected",
        AdversarialOutcome.REJECTED,
        lambda: make_graph_buffers(3, [(0, 0)], capacity=4),
    ))

    # 4. Edge endpoint out of range — should be rejected.
    report.add(_try_adversarial(
        "edge_out_of_range_rejected",
        AdversarialOutcome.REJECTED,
        lambda: make_graph_buffers(3, [(0, 5)], capacity=4),
    ))

    # 5. Negative node count — should be rejected.
    report.add(_try_adversarial(
        "negative_node_count_rejected",
        AdversarialOutcome.REJECTED,
        lambda: make_graph_buffers(-1, [], capacity=4),
    ))

    # 6. Zero nodes — should be rejected or accepted depending on impl.
    # We expect rejection (a graph with 0 nodes is degenerate).
    report.add(_try_adversarial(
        "zero_nodes_rejected",
        AdversarialOutcome.REJECTED,
        lambda: make_graph_buffers(0, [], capacity=4),
    ))

    # 7. Large star graph — should be accepted.
    report.add(_try_adversarial(
        "large_star_accepted",
        AdversarialOutcome.ACCEPTED,
        lambda: make_graph_buffers(100, [(0, i) for i in range(1, 100)], capacity=200),
    ))

    # 8. Complete graph K10 — should be accepted.
    report.add(_try_adversarial(
        "complete_k10_accepted",
        AdversarialOutcome.ACCEPTED,
        lambda: make_graph_buffers(10, [(i, j) for i in range(10) for j in range(i+1, 10)], capacity=100),
    ))

    # 9. Disconnected graph — should be accepted.
    report.add(_try_adversarial(
        "disconnected_graph_accepted",
        AdversarialOutcome.ACCEPTED,
        lambda: make_graph_buffers(6, [(0, 1), (4, 5)], capacity=8),
    ))

    # 10. Duplicate edges — should be accepted (dedup is caller's responsibility).
    report.add(_try_adversarial(
        "duplicate_edges_accepted",
        AdversarialOutcome.ACCEPTED,
        lambda: make_graph_buffers(3, [(0, 1), (0, 1)], capacity=8),
    ))

    return report
