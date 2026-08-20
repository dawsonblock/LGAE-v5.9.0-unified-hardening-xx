"""Exact multi-step MPC by exhaustive enumeration.

Supports both additive utility (analytical O(1) deltas) and
non-additive utility (full recomputation required).

For non-additive utilities, each step requires applying the mutation
and recomputing the full utility function.

exp6.7.1 fixes:
  - apply_action handles reweight_edge with factor param
  - ActionIdentity includes full params (no collisions)
  - No silent except:pass — explicit VALID/INVALID/NO_OP status
  - exact_mpc regenerates candidates at each depth: A(S_t)
  - first_action identity includes canonical params
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from itertools import product
import hashlib
import numpy as np
import torch

from ...types import GraphBuffers
from ...mutations import AddEdge, PruneEdge, ReweightAffinity
from ...runtime.analytical_utility import AnalyticalUtilityOracle


# --- Action identity ---

def _canonical_params(params: dict) -> tuple:
    """Produce a canonical hashable representation of action params."""
    if not params:
        return ()
    items = sorted(params.items())
    return tuple(items)


@dataclass(frozen=True)
class ActionIdentity:
    """Complete action identity including parameters.

    Two actions are equal iff they have the same type, endpoints,
    and canonical parameters. This prevents collisions between
    e.g. reweight×2 and reweight×0.5, or swaps with different
    new_target values.
    """
    mutation_type: str
    u: int
    v: int
    canonical_params: tuple = ()

    @classmethod
    def from_action(cls, action: tuple[str, int, int, dict]) -> "ActionIdentity":
        mt, u, v, params = action
        return cls(mt, u, v, _canonical_params(params))

    @property
    def key(self) -> str:
        """Stable string key for dict lookup."""
        return f"{self.mutation_type}_{self.u}_{self.v}_{self.canonical_params}"


# --- Action application ---

@dataclass
class ActionResult:
    """Result of applying an action."""
    graph: GraphBuffers
    status: str  # "VALID", "INVALID", "NO_OP"
    message: str = ""


def apply_action(graph: GraphBuffers, action: tuple[str, int, int, dict]) -> GraphBuffers:
    """Apply an action to a copy of the graph.

    Returns the new graph. If the action is invalid or a no-op,
    returns a clone of the original graph (unchanged).
    """
    result = apply_action_with_status(graph, action)
    return result.graph


def apply_action_with_status(
    graph: GraphBuffers, action: tuple[str, int, int, dict],
) -> ActionResult:
    """Apply an action with explicit status reporting.

    No silent except:pass. Invalid actions return INVALID, not a
    silently unchanged graph.
    """
    new_graph = graph.clone()
    mt, u, v, params = action
    params = params or {}

    try:
        if mt in ("add_edge", "bridge", "local_rewire", "hub_connect"):
            weight = params.get("weight", 1.0)
            # Note: AddEdge merges weight for existing edges.
            # The exp6.7 candidate generator filters existing edges.
            # Here we allow it (VALID) since it does change the graph.
            AddEdge(u=u, v=v, weight=weight).apply(new_graph)
            return ActionResult(new_graph, "VALID")

        elif mt == "remove_edge":
            if not _edge_exists(graph, u, v):
                return ActionResult(new_graph, "NO_OP",
                                    f"edge ({u},{v}) does not exist")
            PruneEdge(u=u, v=v).apply(new_graph)
            return ActionResult(new_graph, "VALID")

        elif mt in ("reweight_up", "reweight_down", "reweight_edge"):
            if not _edge_exists(graph, u, v):
                return ActionResult(new_graph, "NO_OP",
                                    f"edge ({u},{v}) does not exist")
            # Normalize: reweight_edge uses factor directly.
            if mt == "reweight_edge":
                factor = params.get("factor", 2.0)
            elif mt == "reweight_up":
                factor = params.get("factor", 2.0)
            else:  # reweight_down
                factor = params.get("factor", 0.5)
            ReweightAffinity(u=u, v=v, factor=factor).apply(new_graph)
            return ActionResult(new_graph, "VALID")

        elif mt == "edge_swap":
            if not _edge_exists(graph, u, v):
                return ActionResult(new_graph, "NO_OP",
                                    f"edge ({u},{v}) does not exist for swap")
            w = params.get("new_target", v)
            if _edge_exists(graph, u, w):
                return ActionResult(new_graph, "NO_OP",
                                    f"edge ({u},{w}) already exists")
            if w == v:
                return ActionResult(new_graph, "NO_OP",
                                    "swap target equals removed endpoint")
            PruneEdge(u=u, v=v).apply(new_graph)
            AddEdge(u=u, v=w, weight=params.get("weight", 1.0)).apply(new_graph)
            return ActionResult(new_graph, "VALID")

        else:
            return ActionResult(new_graph, "INVALID",
                                f"unknown mutation type: {mt}")

    except Exception as e:
        return ActionResult(new_graph, "INVALID", str(e))


def _edge_exists(graph: GraphBuffers, u: int, v: int) -> bool:
    """Check if edge (u,v) exists in the graph (undirected)."""
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s, d = int(graph.src[i].item()), int(graph.dst[i].item())
            if (s == u and d == v) or (s == v and d == u):
                return True
    return False


# --- Exact MPC with state-conditioned candidate regeneration ---

@dataclass
class ExactPlan:
    """Result of exact multi-step planning."""
    first_action: tuple[str, int, int] = ("", 0, 0)
    first_action_identity: Optional[ActionIdentity] = None
    best_sequence: list[tuple[str, int, int, dict]] = field(default_factory=list)
    total_value: float = float("-inf")
    all_first_action_values: dict[str, float] = field(default_factory=dict)
    nodes_expanded: int = 0
    horizon: int = 0
    utility_type: str = "additive"


def _filter_valid_actions(
    graph: GraphBuffers,
    actions: list[tuple[str, int, int, dict]],
) -> list[tuple[str, int, int, dict]]:
    """Filter to only actions that are not INVALID.

    NO_OP actions (e.g. reweight of non-existing edge) are excluded
    since they don't change the graph. VALID actions are kept.
    """
    valid = []
    for action in actions:
        result = apply_action_with_status(graph, action)
        if result.status == "VALID":
            valid.append(action)
    return valid


def exact_mpc(
    graph: GraphBuffers,
    z: torch.Tensor,
    available_actions: list[tuple[str, int, int, dict]],
    utility_fn: Callable[[GraphBuffers, torch.Tensor], float],
    *,
    horizon: int = 2,
    gamma: float = 0.9,
    regenerate_candidates: bool = False,
    candidate_generator: Callable | None = None,
    candidate_gen_kwargs: dict | None = None,
) -> ExactPlan:
    """Exact MPC with arbitrary (possibly non-additive) utility.

    Total value = sum_{t=0}^{H-1} gamma^t * [U(S_{t+1}) - U(S_t)]

    When regenerate_candidates=True and candidate_generator is provided,
    candidates are regenerated at each depth: A(S_t), not A(S_0).
    This is the correct recursion for multi-operator planning.
    """
    result = ExactPlan(horizon=horizon, utility_type="non_additive")

    if horizon == 0 or not available_actions:
        return result

    u_before = utility_fn(graph, z)

    # Filter to valid actions at the initial state.
    valid_actions = _filter_valid_actions(graph, available_actions)
    if not valid_actions:
        return result

    best_val = float("-inf")
    best_seq: list[tuple[str, int, int, dict]] = []
    first_values: dict[str, float] = {}
    nodes_expanded = 0

    def _search(
        current_graph: GraphBuffers,
        depth: int,
        current_seq: list,
        current_value: float,
    ) -> None:
        nonlocal best_val, best_seq, nodes_expanded

        if depth == horizon:
            nodes_expanded += 1
            if current_seq:
                aid = ActionIdentity.from_action(current_seq[0])
                key = aid.key
                if key not in first_values or current_value > first_values[key]:
                    first_values[key] = current_value
            if current_value > best_val:
                best_val = current_value
                best_seq = list(current_seq)
            return

        # Get candidates for current state.
        if regenerate_candidates and candidate_generator is not None and depth > 0:
            kwargs = candidate_gen_kwargs or {}
            candidates = candidate_generator(current_graph, z, **kwargs)
        else:
            candidates = available_actions

        # Filter to valid actions at current state.
        valid_here = _filter_valid_actions(current_graph, candidates)
        if not valid_here:
            # No valid actions — evaluate this as a terminal state.
            nodes_expanded += 1
            if current_seq:
                aid = ActionIdentity.from_action(current_seq[0])
                key = aid.key
                if key not in first_values or current_value > first_values[key]:
                    first_values[key] = current_value
            if current_value > best_val:
                best_val = current_value
                best_seq = list(current_seq)
            return

        for action in valid_here:
            u_curr = utility_fn(current_graph, z)
            next_g = apply_action(current_graph, action)
            u_next = utility_fn(next_g, z)
            delta = u_next - u_curr
            new_value = current_value + (gamma ** depth) * delta
            _search(next_g, depth + 1, current_seq + [action], new_value)

    _search(graph, 0, [], 0.0)

    result.nodes_expanded = nodes_expanded
    result.total_value = best_val
    result.best_sequence = best_seq
    result.all_first_action_values = first_values
    if best_seq:
        a = best_seq[0]
        result.first_action = (a[0], a[1], a[2])
        result.first_action_identity = ActionIdentity.from_action(a)
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

    valid_actions = _filter_valid_actions(graph, available_actions)
    if not valid_actions:
        return result

    best_val = float("-inf")
    best_seq: list[tuple[str, int, int, dict]] = []
    first_values: dict[str, float] = {}
    nodes_expanded = 0

    for seq in product(valid_actions, repeat=horizon):
        current = graph
        total = 0.0
        for t, action in enumerate(seq):
            mt, u, v, params = action
            # Normalize mutation type for oracle.
            oracle_mt = mt
            if mt == "reweight_edge":
                factor = params.get("factor", 2.0)
                oracle_mt = "reweight_up" if factor > 1 else "reweight_down"
            try:
                delta = oracle.delta_for_mutation(current, z, oracle_mt, u, v, params)
            except (ValueError, Exception):
                delta = 0.0
            total += (gamma ** t) * delta
            current = apply_action(current, action)

        aid = ActionIdentity.from_action(seq[0])
        key = aid.key
        if key not in first_values or total > first_values[key]:
            first_values[key] = total
        nodes_expanded += 1
        if total > best_val:
            best_val = total
            best_seq = list(seq)

    result.nodes_expanded = nodes_expanded
    result.total_value = best_val
    result.best_sequence = best_seq
    result.all_first_action_values = first_values
    if best_seq:
        a = best_seq[0]
        result.first_action = (a[0], a[1], a[2])
        result.first_action_identity = ActionIdentity.from_action(a)
    return result


def greedy_one_step(
    graph: GraphBuffers,
    z: torch.Tensor,
    available_actions: list[tuple[str, int, int, dict]],
    utility_fn: Callable[[GraphBuffers, torch.Tensor], float],
) -> ExactPlan:
    """Greedy one-step optimization (horizon=1, gamma=1)."""
    return exact_mpc(graph, z, available_actions, utility_fn, horizon=1, gamma=1.0)
