"""Exact multi-step MPC by exhaustive enumeration.

Supports both additive utility (analytical O(1) deltas) and
non-additive utility (full recomputation required).

For non-additive utilities, each step requires applying the mutation
and recomputing the full utility function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from itertools import product
import numpy as np
import torch

from ...types import GraphBuffers
from ...mutations import AddEdge, PruneEdge, ReweightAffinity
from ...runtime.analytical_utility import AnalyticalUtilityOracle


@dataclass
class ExactPlan:
    """Result of exact multi-step planning."""
    first_action: tuple[str, int, int] = ("", 0, 0)
    best_sequence: list[tuple[str, int, int, dict]] = field(default_factory=list)
    total_value: float = float("-inf")
    all_first_action_values: dict[str, float] = field(default_factory=dict)
    nodes_expanded: int = 0
    horizon: int = 0
    utility_type: str = "additive"


def apply_action(graph: GraphBuffers, action: tuple[str, int, int, dict]) -> GraphBuffers:
    """Apply an action to a copy of the graph."""
    new_graph = graph.clone()
    mt, u, v, params = action
    try:
        if mt in ("add_edge", "bridge", "local_rewire", "hub_connect"):
            AddEdge(u=u, v=v, weight=params.get("weight", 1.0)).apply(new_graph)
        elif mt == "remove_edge":
            PruneEdge(u=u, v=v).apply(new_graph)
        elif mt in ("reweight_up", "reweight_down"):
            ReweightAffinity(u=u, v=v, factor=params.get("factor", 2.0)).apply(new_graph)
    except Exception:
        pass
    return new_graph


def exact_mpc(
    graph: GraphBuffers,
    z: torch.Tensor,
    available_actions: list[tuple[str, int, int, dict]],
    utility_fn: Callable[[GraphBuffers, torch.Tensor], float],
    *,
    horizon: int = 2,
    gamma: float = 0.9,
) -> ExactPlan:
    """Exact MPC with arbitrary (possibly non-additive) utility.

    Total value = sum_{t=0}^{H-1} gamma^t * [U(S_{t+1}) - U(S_t)]

    For non-additive utilities, each step requires full utility recomputation.
    """
    result = ExactPlan(horizon=horizon, utility_type="non_additive")

    if horizon == 0 or not available_actions:
        return result

    u_before = utility_fn(graph, z)
    n_seqs = len(available_actions) ** horizon
    result.nodes_expanded = n_seqs

    best_val = float("-inf")
    best_seq: list[tuple[str, int, int, dict]] = []
    first_values: dict[str, float] = {}

    for seq in product(available_actions, repeat=horizon):
        current = graph
        total = 0.0
        for t, action in enumerate(seq):
            u_curr = utility_fn(current, z)
            next_g = apply_action(current, action)
            u_next = utility_fn(next_g, z)
            delta = u_next - u_curr
            total += (gamma ** t) * delta
            current = next_g

        key = f"{seq[0][0]}_{seq[0][1]}_{seq[0][2]}"
        if key not in first_values or total > first_values[key]:
            first_values[key] = total
        if total > best_val:
            best_val = total
            best_seq = list(seq)

    result.total_value = best_val
    result.best_sequence = best_seq
    result.all_first_action_values = first_values
    if best_seq:
        a = best_seq[0]
        result.first_action = (a[0], a[1], a[2])
    return result


def exact_mpc_additive(
    graph: GraphBuffers,
    z: torch.Tensor,
    available_actions: list[tuple[str, int, int, dict]],
    *,
    horizon: int = 2,
    gamma: float = 0.9,
) -> ExactPlan:
    """Exact MPC with additive utility (analytical O(1) deltas)."""
    oracle = AnalyticalUtilityOracle()
    result = ExactPlan(horizon=horizon, utility_type="additive")

    if horizon == 0 or not available_actions:
        return result

    n_seqs = len(available_actions) ** horizon
    result.nodes_expanded = n_seqs

    best_val = float("-inf")
    best_seq: list[tuple[str, int, int, dict]] = []
    first_values: dict[str, float] = {}

    for seq in product(available_actions, repeat=horizon):
        current = graph
        total = 0.0
        for t, action in enumerate(seq):
            mt, u, v, params = action
            delta = oracle.delta_for_mutation(current, z, mt, u, v, params)
            total += (gamma ** t) * delta
            current = apply_action(current, action)

        key = f"{seq[0][0]}_{seq[0][1]}_{seq[0][2]}"
        if key not in first_values or total > first_values[key]:
            first_values[key] = total
        if total > best_val:
            best_val = total
            best_seq = list(seq)

    result.total_value = best_val
    result.best_sequence = best_seq
    result.all_first_action_values = first_values
    if best_seq:
        a = best_seq[0]
        result.first_action = (a[0], a[1], a[2])
    return result


def greedy_one_step(
    graph: GraphBuffers,
    z: torch.Tensor,
    available_actions: list[tuple[str, int, int, dict]],
    utility_fn: Callable[[GraphBuffers, torch.Tensor], float],
) -> ExactPlan:
    """Greedy one-step optimization (horizon=1, gamma=1)."""
    return exact_mpc(graph, z, available_actions, utility_fn, horizon=1, gamma=1.0)
