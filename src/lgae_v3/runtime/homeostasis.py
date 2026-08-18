"""Phase 8 & 9: Structural Homeostasis & Anti-Oscillation Controller.

Homeostasis ensures that structural adaptation does not bloat graph topology:
    H(S, a) = alpha * Delta|E| + beta * Churn + gamma * Complexity + delta * Oscillation

Anti-oscillation controller tracks mutation history to prevent cyclic churn:
    ADD(u,v) <-> PRUNE(u,v) cycles require expected gain > BaseThreshold + HysteresisPenalty.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

from ..types import GraphBuffers


@dataclass
class HomeostasisConfig:
    """Weights and parameters for homeostasis and anti-oscillation."""
    edge_growth_weight: float = 0.05
    churn_weight: float = 0.05
    complexity_weight: float = 0.02
    oscillation_weight: float = 0.2
    min_persistence_interval: int = 5
    reversal_penalty: float = 0.3
    hysteresis_margin: float = 0.1
    history_capacity: int = 50


@dataclass(frozen=True, slots=True)
class HomeostasisPenalty:
    """Decomposition of the homeostasis penalty for a candidate mutation."""
    delta_edges_penalty: float = 0.0
    churn_penalty: float = 0.0
    complexity_penalty: float = 0.0
    oscillation_penalty: float = 0.0
    total_penalty: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "delta_edges_penalty": self.delta_edges_penalty,
            "churn_penalty": self.churn_penalty,
            "complexity_penalty": self.complexity_penalty,
            "oscillation_penalty": self.oscillation_penalty,
            "total_penalty": self.total_penalty,
        }


class AntiOscillationController:
    """Tracks historical structural actions to detect and penalize oscillatory churn."""

    def __init__(self, config: HomeostasisConfig | None = None) -> None:
        self.config = config or HomeostasisConfig()
        # Ring buffer of recent structural actions: (step, action_type, canonical_params)
        self._history: collections.deque[tuple[int, str, str]] = collections.deque(
            maxlen=self.config.history_capacity
        )
        # Map of (u, v) -> (last_action_type, step)
        self._edge_action_history: dict[tuple[int, int], tuple[str, int]] = {}

    def record_action(self, step: int, action_type: str, parameters: dict[str, Any]) -> None:
        u = parameters.get("u")
        v = parameters.get("v")
        key_str = f"{action_type}:{u}:{v}"
        self._history.append((step, action_type, key_str))
        if u is not None and v is not None:
            edge_key = (min(int(u), int(v)), max(int(u), int(v)))
            self._edge_action_history[edge_key] = (action_type, step)

    def compute_oscillation_penalty(
        self, current_step: int, action_type: str, parameters: dict[str, Any]
    ) -> float:
        """Compute penalty if this action reverses or oscillates recent history."""
        u = parameters.get("u")
        v = parameters.get("v")
        if u is None or v is None:
            return 0.0

        edge_key = (min(int(u), int(v)), max(int(u), int(v)))
        if edge_key not in self._edge_action_history:
            return 0.0

        last_action, last_step = self._edge_action_history[edge_key]
        elapsed = current_step - last_step

        # Detect direct reversals: add <-> prune
        is_reversal = (
            ("add" in action_type.lower() and "prune" in last_action.lower())
            or ("prune" in action_type.lower() and "add" in last_action.lower())
            or ("rew" in action_type.lower() and "rew" in last_action.lower())
        )

        if is_reversal and elapsed <= self.config.min_persistence_interval:
            # Reversal within min persistence interval incurs heavy penalty
            decay = max(0.0, 1.0 - (elapsed / max(1, self.config.min_persistence_interval)))
            penalty = self.config.reversal_penalty * (1.0 + decay) + self.config.hysteresis_margin
            return penalty

        # Multi-action cycle detection (e.g. A -> B -> A)
        pattern_count = 0
        target_token = f"{action_type}:{u}:{v}"
        for s, _, tok in self._history:
            if tok == target_token and (current_step - s) <= (self.config.min_persistence_interval * 2):
                pattern_count += 1

        if pattern_count > 1:
            return self.config.oscillation_penalty * pattern_count

        return 0.0


class StructuralHomeostasis:
    """Evaluates graph topology growth, churn, complexity, and anti-oscillation penalties."""

    def __init__(self, config: HomeostasisConfig | None = None) -> None:
        self.config = config or HomeostasisConfig()
        self.anti_oscillation = AntiOscillationController(self.config)
        self._action_count = 0

    def compute_homeostasis_penalty(
        self,
        graph: GraphBuffers,
        action_type: str,
        parameters: dict[str, Any],
        current_step: int = 0,
    ) -> HomeostasisPenalty:
        """Compute the full homeostasis penalty H(S, a)."""
        act = action_type.lower()
        delta_edges = 0.0
        if "add" in act:
            delta_edges = 1.0
        elif "prune" in act or "remove" in act:
            delta_edges = -0.5  # slight bonus/negative penalty for pruning

        # Edge growth penalty
        n_edges = float(graph.valid.sum().item()) if hasattr(graph, "valid") and hasattr(graph.valid, "sum") else 0.0
        capacity = float(len(graph.valid)) if hasattr(graph, "valid") else max(1.0, n_edges * 2)
        density = n_edges / max(1.0, capacity)
        edge_growth_penalty = self.config.edge_growth_weight * delta_edges * (1.0 + density)

        # Churn penalty
        churn_penalty = self.config.churn_weight * (0.5 if "rew" in act else 1.0)

        # Structural complexity (degree variance / density)
        complexity_penalty = self.config.complexity_weight * density

        # Oscillation / reversal penalty
        oscillation_penalty = self.anti_oscillation.compute_oscillation_penalty(
            current_step, action_type, parameters
        )

        total = max(
            0.0,
            edge_growth_penalty + churn_penalty + complexity_penalty + oscillation_penalty,
        )

        return HomeostasisPenalty(
            delta_edges_penalty=round(edge_growth_penalty, 6),
            churn_penalty=round(churn_penalty, 6),
            complexity_penalty=round(complexity_penalty, 6),
            oscillation_penalty=round(oscillation_penalty, 6),
            total_penalty=round(total, 6),
        )

    def record_committed_action(
        self, step: int, action_type: str, parameters: dict[str, Any]
    ) -> None:
        self._action_count += 1
        self.anti_oscillation.record_action(step, action_type, parameters)
