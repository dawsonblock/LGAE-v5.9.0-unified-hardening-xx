"""Real structural candidate generator for exp6.1.

Generates heterogeneous structural candidates from actual graphs,
not synthetic cross-products. Each candidate is a concrete mutation
that can be applied to the graph and exactly evaluated.

Candidate types:
- ADD_EDGE: connect disconnected node pairs
- REMOVE_EDGE: prune existing edges
- REWEIGHT_UP: increase edge weight
- REWEIGHT_DOWN: decrease edge weight
- BRIDGE: connect two low-degree nodes across the graph
- LOCAL_REWIRE: remove an edge and add a nearby one
- HUB_CONNECT: connect a low-degree node to a high-degree node

Each candidate has a natural utility that varies based on graph structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import random
import numpy as np
import torch

from ...types import GraphBuffers, make_graph_buffers
from ...mutations import AddEdge, PruneEdge, ReweightAffinity


@dataclass(slots=True)
class StructuralCandidate:
    """A concrete structural candidate with exact evaluation."""
    candidate_id: int
    action_type: str       # "add_edge", "remove_edge", "reweight_up", etc.
    u: int
    v: int
    params: dict[str, Any] = field(default_factory=dict)
    # Filled after exact evaluation.
    exact_utility: float = 0.0
    exact_delta_utility: float = 0.0
    # Filled by learned scorer.
    predicted_utility: float = 0.0
    predicted_uncertainty: float = 0.0
    ucb_score: float = 0.0
    # State encoding for the learned model.
    z_t: np.ndarray | None = None
    a_t: np.ndarray | None = None


def _get_adjacency(graph: GraphBuffers) -> dict[int, set[int]]:
    """Extract adjacency from graph buffers."""
    n = int(graph.num_nodes)
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            if s < n and d < n:
                adj[s].add(d)
                adj[d].add(s)
    return adj


def _get_degrees(graph: GraphBuffers) -> list[int]:
    adj = _get_adjacency(graph)
    return [len(adj[i]) for i in range(len(adj))]


def _get_existing_edges(graph: GraphBuffers) -> list[tuple[int, int]]:
    """Get list of existing edges."""
    edges = []
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            edges.append((min(s, d), max(s, d)))
    return edges


def _get_non_edges(graph: GraphBuffers, max_count: int = 100) -> list[tuple[int, int]]:
    """Get non-existing edges (disconnected pairs)."""
    n = int(graph.num_nodes)
    adj = _get_adjacency(graph)
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                non_edges.append((i, j))
                if len(non_edges) >= max_count:
                    return non_edges
    return non_edges


def generate_candidates(
    graph: GraphBuffers,
    *,
    n_candidates: int = 50,
    seed: int = 42,
) -> list[StructuralCandidate]:
    """Generate a heterogeneous set of structural candidates.

    Generates a mix of:
    - ADD_EDGE candidates (connect disconnected pairs)
    - REMOVE_EDGE candidates (prune existing edges)
    - REWEIGHT_UP candidates (strengthen edges)
    - REWEIGHT_DOWN candidates (weaken edges)
    - BRIDGE candidates (connect low-degree nodes across graph)
    - LOCAL_REWIRE candidates (swap local connections)
    - HUB_CONNECT candidates (connect peripheral to hub)

    The mix is roughly equal across types, with natural utility variation.
    """
    rng = random.Random(seed)
    n = int(graph.num_nodes)
    adj = _get_adjacency(graph)
    degrees = _get_degrees(graph)
    existing_edges = _get_existing_edges(graph)
    non_edges = _get_non_edges(graph, max_count=n_candidates * 2)

    candidates: list[StructuralCandidate] = []
    cid = 0

    # Target count per type.
    n_per_type = max(1, n_candidates // 7)

    # --- ADD_EDGE: connect disconnected pairs ---
    if non_edges:
        sampled = rng.sample(non_edges, min(n_per_type, len(non_edges)))
        for u, v in sampled:
            candidates.append(StructuralCandidate(
                candidate_id=cid, action_type="add_edge", u=u, v=v,
                params={"weight": 1.0},
            ))
            cid += 1

    # --- REMOVE_EDGE: prune existing edges ---
    if existing_edges:
        sampled = rng.sample(existing_edges, min(n_per_type, len(existing_edges)))
        for u, v in sampled:
            candidates.append(StructuralCandidate(
                candidate_id=cid, action_type="remove_edge", u=u, v=v,
            ))
            cid += 1

    # --- REWEIGHT_UP: strengthen edges ---
    if existing_edges:
        sampled = rng.sample(existing_edges, min(n_per_type, len(existing_edges)))
        for u, v in sampled:
            candidates.append(StructuralCandidate(
                candidate_id=cid, action_type="reweight_up", u=u, v=v,
                params={"factor": 2.0},
            ))
            cid += 1

    # --- REWEIGHT_DOWN: weaken edges ---
    if existing_edges:
        sampled = rng.sample(existing_edges, min(n_per_type, len(existing_edges)))
        for u, v in sampled:
            candidates.append(StructuralCandidate(
                candidate_id=cid, action_type="reweight_down", u=u, v=v,
                params={"factor": 0.5},
            ))
            cid += 1

    # --- BRIDGE: connect two low-degree nodes far apart ---
    low_degree_nodes = sorted(range(n), key=lambda i: degrees[i])[:n // 2]
    for _ in range(n_per_type):
        if len(low_degree_nodes) < 2:
            break
        u, v = rng.sample(low_degree_nodes, 2)
        if v not in adj[u] and u != v:
            candidates.append(StructuralCandidate(
                candidate_id=cid, action_type="bridge", u=u, v=v,
                params={"weight": 1.0},
            ))
            cid += 1

    # --- LOCAL_REWIRE: remove edge and add nearby ---
    if existing_edges:
        for _ in range(n_per_type):
            if not existing_edges:
                break
            u, v = rng.choice(existing_edges)
            # Find a neighbor of u that isn't connected to v
            u_neighbors = list(adj[u])
            if u_neighbors:
                w = rng.choice(u_neighbors)
                if w != v and w not in adj[v]:
                    candidates.append(StructuralCandidate(
                        candidate_id=cid, action_type="local_rewire", u=v, v=w,
                        params={"weight": 1.0, "remove_edge": (u, v)},
                    ))
                    cid += 1

    # --- HUB_CONNECT: connect low-degree to high-degree ---
    high_degree = sorted(range(n), key=lambda i: -degrees[i])[:n // 4]
    low_degree = sorted(range(n), key=lambda i: degrees[i])[:n // 4]
    for _ in range(n_per_type):
        if not high_degree or not low_degree:
            break
        u = rng.choice(low_degree)
        v = rng.choice(high_degree)
        if u != v and v not in adj[u]:
            candidates.append(StructuralCandidate(
                candidate_id=cid, action_type="hub_connect", u=u, v=v,
                params={"weight": 1.0},
            ))
            cid += 1

    return candidates


def apply_candidate(
    graph: GraphBuffers,
    candidate: StructuralCandidate,
) -> GraphBuffers:
    """Apply a candidate mutation to a copy of the graph.

    Returns a new GraphBuffers with the mutation applied.
    """
    # Deep copy the graph.
    new_graph = graph.clone()

    if candidate.action_type in ("add_edge", "bridge", "local_rewire", "hub_connect"):
        mutation = AddEdge(
            u=candidate.u, v=candidate.v,
            weight=candidate.params.get("weight", 1.0),
        )
        try:
            mutation.apply(new_graph)
        except Exception:
            pass  # invalid candidate, skip
    elif candidate.action_type == "remove_edge":
        mutation = PruneEdge(u=candidate.u, v=candidate.v)
        try:
            mutation.apply(new_graph)
        except Exception:
            pass
    elif candidate.action_type == "reweight_up":
        mutation = ReweightAffinity(
            u=candidate.u, v=candidate.v,
            factor=candidate.params.get("factor", 2.0),
        )
        try:
            mutation.apply(new_graph)
        except Exception:
            pass
    elif candidate.action_type == "reweight_down":
        mutation = ReweightAffinity(
            u=candidate.u, v=candidate.v,
            factor=candidate.params.get("factor", 0.5),
        )
        try:
            mutation.apply(new_graph)
        except Exception:
            pass

    return new_graph


def compute_exact_utility(
    graph: GraphBuffers,
    z: torch.Tensor,
) -> float:
    """Compute the default structural utility of a graph.

    U = -sum(w * ||z_u - z_v||^2) over active edges.
    """
    with torch.no_grad():
        src = graph.src[graph.valid]
        dst = graph.dst[graph.valid]
        if src.numel() == 0:
            return 0.0
        d = (z[src] - z[dst]).pow(2).sum(-1)
        w = graph.weight[graph.valid]
        return float(-(w * d).sum().item())


def evaluate_candidates_exact(
    graph: GraphBuffers,
    z: torch.Tensor,
    candidates: list[StructuralCandidate],
) -> None:
    """Evaluate all candidates exactly (oracle evaluation).

    Fills in:
    - exact_utility: utility after applying the candidate
    - exact_delta_utility: change in utility
    """
    u_before = compute_exact_utility(graph, z)

    for cand in candidates:
        new_graph = apply_candidate(graph, cand)
        u_after = compute_exact_utility(new_graph, z)
        cand.exact_utility = u_after
        cand.exact_delta_utility = u_after - u_before
