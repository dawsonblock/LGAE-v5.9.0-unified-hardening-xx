"""Exact and learned multi-step planning for exp6.3.

Exact MPC:
    For horizon H, enumerate all action sequences of length H.
    For each sequence, compute the exact total utility:
        Q = sum_{t=0}^{H-1} gamma^t * DeltaU_immediate(S_t, a_t)
    where S_{t+1} = T(S_t, a_t) (apply mutation to graph).

    The exact MPC picks the first action of the best sequence.

Learned value model:
    Predicts V(S') = future utility from state S'.
    The learned planner picks:
        argmax_a [ DeltaU_analytical(S,a) + gamma * V_hat(S') ]

    If V_hat is accurate, the learned planner agrees with exact MPC
    while evaluating far fewer branches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from itertools import product
import numpy as np
import torch

from ...types import GraphBuffers
from ...mutations import AddEdge, PruneEdge, ReweightAffinity
from .candidate_generator import StructuralCandidate, compute_exact_utility
from .analytical_utility import (
    compute_analytical_delta_utility, get_edge_weight,
)


@dataclass
class MPCResult:
    """Result of multi-step planning."""
    best_first_action: tuple[str, int, int] = ("", 0, 0)
    best_sequence: list[tuple[str, int, int, dict]] = field(default_factory=list)
    best_total_utility: float = float("-inf")
    all_first_action_utilities: dict[str, float] = field(default_factory=dict)
    n_evaluations: int = 0
    horizon: int = 0


def apply_action_to_graph(
    graph: GraphBuffers,
    action: tuple[str, int, int, dict],
) -> GraphBuffers:
    """Apply an action to a copy of the graph."""
    new_graph = graph.clone()
    action_type, u, v, params = action

    if action_type == "add_edge":
        try:
            AddEdge(u=u, v=v, weight=params.get("weight", 1.0)).apply(new_graph)
        except Exception:
            pass
    elif action_type == "remove_edge":
        try:
            PruneEdge(u=u, v=v).apply(new_graph)
        except Exception:
            pass
    elif action_type == "reweight_up":
        try:
            ReweightAffinity(u=u, v=v, factor=params.get("factor", 2.0)).apply(new_graph)
        except Exception:
            pass
    elif action_type == "reweight_down":
        try:
            ReweightAffinity(u=u, v=v, factor=params.get("factor", 0.5)).apply(new_graph)
        except Exception:
            pass

    return new_graph


def compute_action_delta_utility(
    graph: GraphBuffers,
    z: torch.Tensor,
    action: tuple[str, int, int, dict],
) -> float:
    """Compute the exact immediate delta utility of an action."""
    action_type, u, v, params = action
    cand = StructuralCandidate(
        candidate_id=0, action_type=action_type, u=u, v=v, params=params,
    )
    return compute_analytical_delta_utility(graph, z, cand)


def exact_mpc(
    graph: GraphBuffers,
    z: torch.Tensor,
    available_actions: list[tuple[str, int, int, dict]],
    *,
    horizon: int = 2,
    gamma: float = 0.9,
) -> MPCResult:
    """Exact multi-step MPC by full enumeration.

    For horizon H, evaluates all H-length action sequences.
    Total utility = sum_{t=0}^{H-1} gamma^t * DeltaU(S_t, a_t)

    Returns the best first action and its total utility.
    """
    result = MPCResult(horizon=horizon)

    if horizon == 0 or len(available_actions) == 0:
        return result

    n_sequences = len(available_actions) ** horizon
    result.n_evaluations = n_sequences

    best_utility = float("-inf")
    best_sequence = []
    first_action_utilities: dict[str, float] = {}

    # Enumerate all sequences.
    for seq in product(available_actions, repeat=horizon):
        current_graph = graph
        total_u = 0.0
        valid_seq = True

        for t, action in enumerate(seq):
            delta_u = compute_action_delta_utility(current_graph, z, action)
            total_u += (gamma ** t) * delta_u
            current_graph = apply_action_to_graph(current_graph, action)

            # If action was invalid (no change), skip.
            if delta_u == 0 and action[0] in ("remove_edge",):
                # Check if edge existed.
                w = get_edge_weight(current_graph, action[1], action[2])
                if w is None and t == 0:
                    valid_seq = False
                    break

        if not valid_seq:
            continue

        first_action_key = f"{seq[0][0]}_{seq[0][1]}_{seq[0][2]}"
        if first_action_key not in first_action_utilities or total_u > first_action_utilities[first_action_key]:
            first_action_utilities[first_action_key] = total_u

        if total_u > best_utility:
            best_utility = total_u
            best_sequence = list(seq)

    result.best_total_utility = best_utility
    result.best_sequence = best_sequence
    result.all_first_action_utilities = first_action_utilities

    if best_sequence:
        a = best_sequence[0]
        result.best_first_action = (a[0], a[1], a[2])

    return result


def greedy_one_step(
    graph: GraphBuffers,
    z: torch.Tensor,
    available_actions: list[tuple[str, int, int, dict]],
) -> MPCResult:
    """Greedy one-step optimization (horizon=1)."""
    return exact_mpc(graph, z, available_actions, horizon=1, gamma=1.0)


# ---------------------------------------------------------------------------
# Learned value model
# ---------------------------------------------------------------------------

@dataclass
class LearnedPlannerResult:
    """Result of learned planning."""
    best_first_action: tuple[str, int, int] = ("", 0, 0)
    best_total_utility: float = float("-inf")
    n_evaluations: int = 0
    horizon: int = 0
    predicted_values: dict[str, float] = field(default_factory=dict)


class FutureValueModel:
    """Simple learned model for future state value V(S').

    Predicts the future utility achievable from a given graph state.
    Uses graphlet features as input.

    Training: collect (S, V_exact) pairs from exact MPC runs,
    where V_exact = best total utility from S over horizon H.
    """

    def __init__(self, state_dim: int = 8, hidden_dim: int = 32, seed: int = 42):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.seed = seed
        rng = np.random.RandomState(seed)
        # Simple 2-layer MLP weights.
        self.W1 = rng.randn(state_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, 1) * 0.5
        self.b2 = np.zeros(1)
        self.trained = False

    def _features(self, graph: GraphBuffers, z: torch.Tensor) -> np.ndarray:
        """Extract graphlet-like features from the graph."""
        n = int(graph.num_nodes)
        valid = graph.valid.bool()

        # Basic graph statistics.
        n_edges = int(valid.sum().item())
        density = n_edges / max(n * (n - 1) / 2, 1)

        # Degree statistics.
        degrees = np.zeros(n)
        for i in range(graph.src.shape[0]):
            if valid[i]:
                s = int(graph.src[i].item())
                d = int(graph.dst[i].item())
                if s < n:
                    degrees[s] += 1
                if d < n:
                    degrees[d] += 1

        degree_mean = float(np.mean(degrees))
        degree_std = float(np.std(degrees))
        degree_max = float(np.max(degrees))

        # Utility-related features.
        u_current = compute_exact_utility(graph, z)

        # Latent distance statistics.
        src = graph.src[valid]
        dst = graph.dst[valid]
        if src.numel() > 0:
            d = (z[src] - z[dst]).pow(2).sum(-1)
            w = graph.weight[valid]
            d_mean = float(d.mean().item())
            d_std = float(d.std().item())
            d_min = float(d.min().item())
            d_max = float(d.max().item())
        else:
            d_mean = d_std = d_min = d_max = 0.0

        return np.array([
            n / 50.0,           # normalized size
            density,            # edge density
            degree_mean / 10.0, # mean degree
            degree_std / 10.0,  # degree std
            degree_max / 20.0,  # max degree
            u_current / 100.0,  # current utility
            d_mean,             # mean latent distance
            d_std,              # latent distance std
        ])

    def predict(self, graph: GraphBuffers, z: torch.Tensor) -> float:
        """Predict V(S) — future value from this state."""
        x = self._features(graph, z)
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        out = h @ self.W2 + self.b2
        return float(out[0])

    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 100, lr: float = 0.01):
        """Simple gradient descent training."""
        n = len(X)
        if n == 0:
            return

        for epoch in range(n_epochs):
            # Forward pass.
            h = np.maximum(0, X @ self.W1 + self.b1)
            pred = (h @ self.W2 + self.b2).flatten()

            # MSE loss gradient.
            error = pred - y
            grad_out = error.reshape(-1, 1) / n

            # Gradients.
            grad_W2 = h.T @ grad_out
            grad_b2 = grad_out.sum(axis=0)
            grad_h = grad_out @ self.W2.T
            grad_h[h <= 0] = 0  # ReLU gradient

            grad_W1 = X.T @ grad_h
            grad_b1 = grad_h.sum(axis=0)

            # Update.
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1
            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2

        self.trained = True


def learned_plan(
    graph: GraphBuffers,
    z: torch.Tensor,
    available_actions: list[tuple[str, int, int, dict]],
    value_model: FutureValueModel,
    *,
    horizon: int = 2,
    gamma: float = 0.9,
) -> LearnedPlannerResult:
    """Learned multi-step planning.

    For each first action:
        1. Compute exact immediate ΔU (analytical, O(1))
        2. Apply action to get S'
        3. Predict V(S') using learned model
        4. Q(a) = ΔU + gamma * V(S')

    For H=2, also enumerate second actions and use V for the third step.
    For H=3, enumerate first two actions and use V for the fourth step.

    This evaluates far fewer branches than exact MPC.
    """
    result = LearnedPlannerResult(horizon=horizon)

    if horizon == 0 or len(available_actions) == 0:
        return result

    best_utility = float("-inf")
    best_action = ("", 0, 0)
    predicted_values = {}

    if horizon == 1:
        # Just immediate utility.
        for action in available_actions:
            delta_u = compute_action_delta_utility(graph, z, action)
            key = f"{action[0]}_{action[1]}_{action[2]}"
            predicted_values[key] = delta_u
            result.n_evaluations += 1
            if delta_u > best_utility:
                best_utility = delta_u
                best_action = (action[0], action[1], action[2])

    elif horizon == 2:
        # For each first action, enumerate second actions and use V for the rest.
        for a0 in available_actions:
            delta_u0 = compute_action_delta_utility(graph, z, a0)
            s1 = apply_action_to_graph(graph, a0)

            # Enumerate second actions.
            best_q1 = float("-inf")
            for a1 in available_actions:
                delta_u1 = compute_action_delta_utility(s1, z, a1)
                s2 = apply_action_to_graph(s1, a1)
                v2 = value_model.predict(s2, z)
                q1 = delta_u1 + gamma * v2
                if q1 > best_q1:
                    best_q1 = q1
                result.n_evaluations += 1

            q0 = delta_u0 + gamma * best_q1
            key = f"{a0[0]}_{a0[1]}_{a0[2]}"
            predicted_values[key] = q0

            if q0 > best_utility:
                best_utility = q0
                best_action = (a0[0], a0[1], a0[2])

    elif horizon == 3:
        # For each first action, use learned planning for steps 2-3.
        for a0 in available_actions:
            delta_u0 = compute_action_delta_utility(graph, z, a0)
            s1 = apply_action_to_graph(graph, a0)

            # Use H=2 learned planning from s1.
            inner_result = learned_plan(s1, z, available_actions, value_model,
                                        horizon=2, gamma=gamma)
            q0 = delta_u0 + gamma * inner_result.best_total_utility
            result.n_evaluations += inner_result.n_evaluations + 1

            key = f"{a0[0]}_{a0[1]}_{a0[2]}"
            predicted_values[key] = q0

            if q0 > best_utility:
                best_utility = q0
                best_action = (a0[0], a0[1], a0[2])

    result.best_total_utility = best_utility
    result.best_first_action = best_action
    result.predicted_values = predicted_values

    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def first_action_agreement(
    exact: MPCResult,
    learned: LearnedPlannerResult,
) -> bool:
    """Check if learned planner picks the same first action as exact MPC."""
    return exact.best_first_action == learned.best_first_action


def planning_regret(
    exact: MPCResult,
    learned: LearnedPlannerResult,
) -> float:
    """Planning regret = Q(a*) - Q(a_model)."""
    exact_key = f"{exact.best_first_action[0]}_{exact.best_first_action[1]}_{exact.best_first_action[2]}"
    learned_key = f"{learned.best_first_action[0]}_{learned.best_first_action[1]}_{learned.best_first_action[2]}"

    exact_u = exact.all_first_action_utilities.get(exact_key, exact.best_total_utility)
    learned_u = exact.all_first_action_utilities.get(learned_key, learned.best_total_utility)

    return float(exact_u - learned_u)


def search_savings(
    exact: MPCResult,
    learned: LearnedPlannerResult,
) -> float:
    """Fraction of evaluations saved by learned planner."""
    if exact.n_evaluations == 0:
        return 0.0
    return 1.0 - learned.n_evaluations / exact.n_evaluations
