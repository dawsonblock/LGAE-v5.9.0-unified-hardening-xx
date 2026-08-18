"""v6 baseline runners.

Nine baselines that compete against any learned v6 policy:

1. **Fixed-topology**: no mutations at all (do-nothing).
2. **Random-rewiring**: random edge add/prune each step.
3. **Greedy**: pick the candidate with highest immediate ΔU (1-step greedy).
4. **Curvature-only**: pick the candidate that most improves Forman curvature.
5. **FoSR**: Fielder's spectral gap optimization (add edge maximizing λ₂).
6. **BORF**: Bottleneck-oriented edge rewiring (prune low-curvature, add
   high-curvature).
7. **Effective-resistance**: add edge minimizing total effective resistance.
8. **One-step counterfactual**: evaluate all candidates with exact shadow
   execution, pick the best (this is the v5.11 1-step lookahead).
9. **MPC**: v5.11 with multi-step planning horizon.
10. **MPC+IG**: v5.11 with MPC and information-gain weighted selection.
11. **Full-v5.11**: the complete v5.11 runtime (all phases active).

Each baseline implements the ``BaselineRunner`` protocol:

    run(graph, config, seed, n_steps) -> BaselineResult

Results are comparable across baselines because they all use the same
graph families, seeds, and metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import random
import math

import torch
import numpy as np

from ..types import GraphBuffers, make_graph_buffers
from ..config import ResearchConfig, LGAEConfig
from ..operators import spectral_gap_graphbuffers


@dataclass(slots=True)
class BaselineResult:
    """Result of running a baseline on one graph instance."""
    baseline_name: str
    final_utility: float
    utility_history: list[float] = field(default_factory=list)
    n_mutations: int = 0
    n_rejected: int = 0
    n_steps: int = 0
    compute_cost: float = 0.0  # wall-clock seconds or FLOPs
    final_n_edges: int = 0
    final_n_nodes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "final_utility": float(self.final_utility),
            "utility_history": [float(u) for u in self.utility_history],
            "n_mutations": int(self.n_mutations),
            "n_rejected": int(self.n_rejected),
            "n_steps": int(self.n_steps),
            "compute_cost": float(self.compute_cost),
            "final_n_edges": int(self.final_n_edges),
            "final_n_nodes": int(self.final_n_nodes),
            "metadata": self.metadata,
        }


class BaselineRunner(Protocol):
    """Protocol for all baseline runners."""
    name: str

    def run(
        self,
        graph: GraphBuffers,
        config: ResearchConfig | LGAEConfig,
        seed: int,
        n_steps: int,
    ) -> BaselineResult:
        ...


# ---------------------------------------------------------------------------
# Utility functions shared by baselines.
# ---------------------------------------------------------------------------

def _graph_utility(graph: GraphBuffers) -> float:
    """Spectral gap as the utility function for baselines."""
    try:
        lam, _ = spectral_gap_graphbuffers(graph)
        return float(lam)
    except Exception:
        return 0.0


def _count_edges(graph: GraphBuffers) -> int:
    valid = graph.valid.bool()
    return int(valid.sum().item())


def _add_edge(graph: GraphBuffers, u: int, v: int) -> GraphBuffers:
    """Create a new GraphBuffers with one edge added."""
    valid = graph.valid.bool()
    edges = []
    for i in range(graph.src.shape[0]):
        if valid[i]:
            edges.append((int(graph.src[i].item()), int(graph.dst[i].item())))
    if (u, v) not in edges and (v, u) not in edges and u != v:
        edges.append((u, v))
    n = int(graph.num_nodes)
    capacity = max(len(edges) + 8, n * 2)
    return make_graph_buffers(n, edges, capacity=capacity)


def _remove_edge(graph: GraphBuffers, u: int, v: int) -> GraphBuffers:
    """Create a new GraphBuffers with one edge removed."""
    valid = graph.valid.bool()
    edges = []
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s, d = int(graph.src[i].item()), int(graph.dst[i].item())
            if (s, d) != (u, v) and (d, s) != (u, v):
                edges.append((s, d))
    n = int(graph.num_nodes)
    if not edges:
        edges = [(0, 1)] if n >= 2 else []
    capacity = max(len(edges) + 8, n * 2)
    return make_graph_buffers(n, edges, capacity=capacity)


def _get_valid_edges(graph: GraphBuffers) -> list[tuple[int, int]]:
    valid = graph.valid.bool()
    edges = []
    for i in range(graph.src.shape[0]):
        if valid[i]:
            edges.append((int(graph.src[i].item()), int(graph.dst[i].item())))
    return edges


def _get_non_edges(graph: GraphBuffers, max_count: int = 50) -> list[tuple[int, int]]:
    """Get non-edges (candidate additions), bounded for speed."""
    n = int(graph.num_nodes)
    existing = set()
    for u, v in _get_valid_edges(graph):
        existing.add((u, v))
        existing.add((v, u))
    non_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in existing:
                non_edges.append((u, v))
                if len(non_edges) >= max_count:
                    return non_edges
    return non_edges


# ---------------------------------------------------------------------------
# 1. Fixed-topology baseline (do nothing).
# ---------------------------------------------------------------------------

class FixedTopologyBaseline:
    """No mutations. Establishes the floor."""
    name = "fixed_topology"

    def run(self, graph: GraphBuffers, config, seed: int, n_steps: int) -> BaselineResult:
        u = _graph_utility(graph)
        history = [u] * max(n_steps, 1)
        return BaselineResult(
            baseline_name=self.name,
            final_utility=u,
            utility_history=history,
            n_mutations=0,
            n_rejected=0,
            n_steps=n_steps,
            compute_cost=0.0,
            final_n_edges=_count_edges(graph),
            final_n_nodes=int(graph.num_nodes),
        )


# ---------------------------------------------------------------------------
# 2. Random-rewiring baseline.
# ---------------------------------------------------------------------------

class RandomRewiringBaseline:
    """Random edge add/prune each step."""
    name = "random_rewiring"

    def run(self, graph: GraphBuffers, config, seed: int, n_steps: int) -> BaselineResult:
        rng = random.Random(seed)
        current = graph
        u0 = _graph_utility(current)
        history = [u0]
        n_mut = 0
        for _ in range(max(n_steps, 1)):
            n = int(current.num_nodes)
            action = rng.choice(["add", "prune"])
            if action == "add":
                non_edges = _get_non_edges(current, max_count=20)
                if non_edges:
                    u, v = rng.choice(non_edges)
                    current = _add_edge(current, u, v)
                    n_mut += 1
            else:
                edges = _get_valid_edges(current)
                if len(edges) > 1:
                    u, v = rng.choice(edges)
                    current = _remove_edge(current, u, v)
                    n_mut += 1
            u = _graph_utility(current)
            history.append(u)
        return BaselineResult(
            baseline_name=self.name,
            final_utility=history[-1],
            utility_history=history,
            n_mutations=n_mut,
            n_rejected=0,
            n_steps=n_steps,
            compute_cost=float(n_steps),
            final_n_edges=_count_edges(current),
            final_n_nodes=int(current.num_nodes),
        )


# ---------------------------------------------------------------------------
# 3. Greedy baseline (pick highest immediate ΔU).
# ---------------------------------------------------------------------------

class GreedyBaseline:
    """1-step greedy: try all candidate adds, pick the one with highest ΔU."""
    name = "greedy"

    def run(self, graph: GraphBuffers, config, seed: int, n_steps: int) -> BaselineResult:
        rng = random.Random(seed)
        current = graph
        u0 = _graph_utility(current)
        history = [u0]
        n_mut = 0
        for _ in range(max(n_steps, 1)):
            u_base = _graph_utility(current)
            best_delta = 0.0
            best_graph = current
            non_edges = _get_non_edges(current, max_count=30)
            for u, v in non_edges:
                candidate = _add_edge(current, u, v)
                u_new = _graph_utility(candidate)
                delta = u_new - u_base
                if delta > best_delta:
                    best_delta = delta
                    best_graph = candidate
            if best_graph is not current:
                current = best_graph
                n_mut += 1
            history.append(_graph_utility(current))
        return BaselineResult(
            baseline_name=self.name,
            final_utility=history[-1],
            utility_history=history,
            n_mutations=n_mut,
            n_rejected=0,
            n_steps=n_steps,
            compute_cost=float(n_steps * 30),
            final_n_edges=_count_edges(current),
            final_n_nodes=int(current.num_nodes),
        )


# ---------------------------------------------------------------------------
# 4. Curvature-only baseline.
# ---------------------------------------------------------------------------

class CurvatureOnlyBaseline:
    """Pick the edge addition that most improves Forman curvature.

    Uses a simplified Forman curvature proxy: for each candidate edge (u,v),
    the curvature improvement is estimated as:
        Δκ ≈ 2/deg(u) + 2/deg(v) - 1
    Higher is better (less negative curvature).
    """
    name = "curvature_only"

    def run(self, graph: GraphBuffers, config, seed: int, n_steps: int) -> BaselineResult:
        rng = random.Random(seed)
        current = graph
        u0 = _graph_utility(current)
        history = [u0]
        n_mut = 0
        for _ in range(max(n_steps, 1)):
            n = int(current.num_nodes)
            edges = _get_valid_edges(current)
            degrees = [0] * n
            for u, v in edges:
                degrees[u] += 1
                degrees[v] += 1
            non_edges = _get_non_edges(current, max_count=30)
            best_curv = float("-inf")
            best_edge = None
            for u, v in non_edges:
                du = max(degrees[u], 1)
                dv = max(degrees[v], 1)
                curv = 2.0 / du + 2.0 / dv - 1.0
                if curv > best_curv:
                    best_curv = curv
                    best_edge = (u, v)
            if best_edge is not None and best_curv > -0.5:
                current = _add_edge(current, *best_edge)
                n_mut += 1
            history.append(_graph_utility(current))
        return BaselineResult(
            baseline_name=self.name,
            final_utility=history[-1],
            utility_history=history,
            n_mutations=n_mut,
            n_rejected=0,
            n_steps=n_steps,
            compute_cost=float(n_steps * 30),
            final_n_edges=_count_edges(current),
            final_n_nodes=int(current.num_nodes),
        )


# ---------------------------------------------------------------------------
# 5. FoSR baseline (Fiedler's spectral gap optimization).
# ---------------------------------------------------------------------------

class FoSRBaseline:
    """FoSR: add the edge that maximizes the spectral gap (λ₂).

    This is the classic spectral-gap optimization heuristic. It tries all
    candidate edge additions and picks the one that maximizes λ₂ of the
    graph Laplacian.
    """
    name = "fosr"

    def run(self, graph: GraphBuffers, config, seed: int, n_steps: int) -> BaselineResult:
        current = graph
        u0 = _graph_utility(current)
        history = [u0]
        n_mut = 0
        for _ in range(max(n_steps, 1)):
            u_base = _graph_utility(current)
            best_gap = u_base
            best_graph = current
            non_edges = _get_non_edges(current, max_count=30)
            for u, v in non_edges:
                candidate = _add_edge(current, u, v)
                u_new = _graph_utility(candidate)
                if u_new > best_gap:
                    best_gap = u_new
                    best_graph = candidate
            if best_graph is not current:
                current = best_graph
                n_mut += 1
            history.append(_graph_utility(current))
        return BaselineResult(
            baseline_name=self.name,
            final_utility=history[-1],
            utility_history=history,
            n_mutations=n_mut,
            n_rejected=0,
            n_steps=n_steps,
            compute_cost=float(n_steps * 30),
            final_n_edges=_count_edges(current),
            final_n_nodes=int(current.num_nodes),
        )


# ---------------------------------------------------------------------------
# 6. BORF baseline (Bottleneck-Oriented Rewiring).
# ---------------------------------------------------------------------------

class BORFBaseline:
    """BORF: prune the lowest-curvature edge, add the highest-curvature edge.

    For each step:
    1. Find the edge with the most negative Forman curvature.
    2. Remove it.
    3. Add the non-edge with the highest Forman curvature.
    """
    name = "borf"

    def run(self, graph: GraphBuffers, config, seed: int, n_steps: int) -> BaselineResult:
        rng = random.Random(seed)
        current = graph
        u0 = _graph_utility(current)
        history = [u0]
        n_mut = 0
        for _ in range(max(n_steps, 1)):
            n = int(current.num_nodes)
            edges = _get_valid_edges(current)
            if len(edges) <= 1:
                history.append(_graph_utility(current))
                continue
            degrees = [0] * n
            for u, v in edges:
                degrees[u] += 1
                degrees[v] += 1
            # Find lowest-curvature edge.
            worst_curv = float("inf")
            worst_edge = None
            for u, v in edges:
                du = max(degrees[u], 1)
                dv = max(degrees[v], 1)
                curv = 2.0 / du + 2.0 / dv - 1.0
                if curv < worst_curv:
                    worst_curv = curv
                    worst_edge = (u, v)
            # Find highest-curvature non-edge.
            non_edges = _get_non_edges(current, max_count=30)
            best_curv = float("-inf")
            best_edge = None
            for u, v in non_edges:
                du = max(degrees[u], 1)
                dv = max(degrees[v], 1)
                curv = 2.0 / du + 2.0 / dv - 1.0
                if curv > best_curv:
                    best_curv = curv
                    best_edge = (u, v)
            # Rewire: remove worst, add best.
            if worst_edge is not None:
                current = _remove_edge(current, *worst_edge)
                n_mut += 1
            if best_edge is not None:
                current = _add_edge(current, *best_edge)
                n_mut += 1
            history.append(_graph_utility(current))
        return BaselineResult(
            baseline_name=self.name,
            final_utility=history[-1],
            utility_history=history,
            n_mutations=n_mut,
            n_rejected=0,
            n_steps=n_steps,
            compute_cost=float(n_steps * 60),
            final_n_edges=_count_edges(current),
            final_n_nodes=int(current.num_nodes),
        )


# ---------------------------------------------------------------------------
# 7. Effective-resistance baseline.
# ---------------------------------------------------------------------------

class EffectiveResistanceBaseline:
    """Add the edge that minimizes total effective resistance.

    Uses a simplified proxy: the edge (u,v) that connects the most
    topologically distant nodes (highest resistance) is added.
    """
    name = "effective_resistance"

    def run(self, graph: GraphBuffers, config, seed: int, n_steps: int) -> BaselineResult:
        current = graph
        u0 = _graph_utility(current)
        history = [u0]
        n_mut = 0
        for _ in range(max(n_steps, 1)):
            n = int(current.num_nodes)
            # Compute shortest-path distances (BFS from each node).
            import networkx as nx
            edges = _get_valid_edges(current)
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from(edges)
            non_edges = _get_non_edges(current, max_count=30)
            best_resistance = -1.0
            best_edge = None
            for u, v in non_edges:
                try:
                    dist = nx.shortest_path_length(G, u, v)
                except nx.NetworkXNoPath:
                    dist = n  # disconnected → high resistance
                if dist > best_resistance:
                    best_resistance = dist
                    best_edge = (u, v)
            if best_edge is not None:
                current = _add_edge(current, *best_edge)
                n_mut += 1
            history.append(_graph_utility(current))
        return BaselineResult(
            baseline_name=self.name,
            final_utility=history[-1],
            utility_history=history,
            n_mutations=n_mut,
            n_rejected=0,
            n_steps=n_steps,
            compute_cost=float(n_steps * 30),
            final_n_edges=_count_edges(current),
            final_n_nodes=int(current.num_nodes),
        )


# ---------------------------------------------------------------------------
# 8. One-step counterfactual baseline.
# ---------------------------------------------------------------------------

class OneStepCounterfactualBaseline:
    """Evaluate all candidates with exact shadow execution, pick the best.

    This is equivalent to the v5.11 1-step lookahead without MPC, IG, or
    governance complexity. It establishes what pure 1-step optimization
    can achieve.
    """
    name = "one_step_counterfactual"

    def run(self, graph: GraphBuffers, config, seed: int, n_steps: int) -> BaselineResult:
        current = graph
        u0 = _graph_utility(current)
        history = [u0]
        n_mut = 0
        n_rejected = 0
        for _ in range(max(n_steps, 1)):
            u_base = _graph_utility(current)
            best_delta = 0.0
            best_graph = current
            # Try all adds.
            non_edges = _get_non_edges(current, max_count=30)
            for u, v in non_edges:
                candidate = _add_edge(current, u, v)
                u_new = _graph_utility(candidate)
                delta = u_new - u_base
                if delta > best_delta:
                    best_delta = delta
                    best_graph = candidate
            # Try all prunes.
            edges = _get_valid_edges(current)
            for u, v in edges:
                if len(edges) <= 1:
                    break
                candidate = _remove_edge(current, u, v)
                u_new = _graph_utility(candidate)
                delta = u_new - u_base
                if delta > best_delta:
                    best_delta = delta
                    best_graph = candidate
            if best_graph is not current:
                current = best_graph
                n_mut += 1
            else:
                n_rejected += 1
            history.append(_graph_utility(current))
        return BaselineResult(
            baseline_name=self.name,
            final_utility=history[-1],
            utility_history=history,
            n_mutations=n_mut,
            n_rejected=n_rejected,
            n_steps=n_steps,
            compute_cost=float(n_steps * 60),
            final_n_edges=_count_edges(current),
            final_n_nodes=int(current.num_nodes),
        )


# ---------------------------------------------------------------------------
# 9. MPC baseline (multi-step planning, no IG).
# ---------------------------------------------------------------------------

class MPCBaseline:
    """MPC: multi-step planning with horizon > 1.

    Looks ahead H steps by greedily simulating the best add at each step.
    No information-gain weighting.
    """
    name = "mpc"

    def __init__(self, horizon: int = 3) -> None:
        self.horizon = horizon

    def run(self, graph: GraphBuffers, config, seed: int, n_steps: int) -> BaselineResult:
        current = graph
        u0 = _graph_utility(current)
        history = [u0]
        n_mut = 0
        for _ in range(max(n_steps, 1)):
            # Plan H steps ahead, execute the first.
            best_first_graph = current
            best_final_utility = _graph_utility(current)
            non_edges = _get_non_edges(current, max_count=15)
            for u, v in non_edges:
                sim_graph = _add_edge(current, u, v)
                # Greedy rollout for H-1 more steps.
                for _h in range(self.horizon - 1):
                    sim_u_base = _graph_utility(sim_graph)
                    sim_non_edges = _get_non_edges(sim_graph, max_count=10)
                    sim_best = sim_graph
                    sim_best_u = sim_u_base
                    for su, sv in sim_non_edges:
                        sc = _add_edge(sim_graph, su, sv)
                        su_new = _graph_utility(sc)
                        if su_new > sim_best_u:
                            sim_best_u = su_new
                            sim_best = sc
                    sim_graph = sim_best
                final_u = _graph_utility(sim_graph)
                if final_u > best_final_utility:
                    best_final_utility = final_u
                    best_first_graph = _add_edge(current, u, v)
            if best_first_graph is not current:
                current = best_first_graph
                n_mut += 1
            history.append(_graph_utility(current))
        return BaselineResult(
            baseline_name=self.name,
            final_utility=history[-1],
            utility_history=history,
            n_mutations=n_mut,
            n_rejected=0,
            n_steps=n_steps,
            compute_cost=float(n_steps * 15 * self.horizon * 10),
            final_n_edges=_count_edges(current),
            final_n_nodes=int(current.num_nodes),
            metadata={"horizon": self.horizon},
        )


# ---------------------------------------------------------------------------
# 10. MPC + IG baseline.
# ---------------------------------------------------------------------------

class MPCWithIGBaseline:
    """MPC with information-gain weighted selection.

    Same as MPC but adds an exploration bonus for edges that connect
    topologically distant regions (high information gain proxy).
    """
    name = "mpc_with_ig"

    def __init__(self, horizon: int = 3, ig_weight: float = 0.1) -> None:
        self.horizon = horizon
        self.ig_weight = ig_weight

    def run(self, graph: GraphBuffers, config, seed: int, n_steps: int) -> BaselineResult:
        import networkx as nx
        current = graph
        u0 = _graph_utility(current)
        history = [u0]
        n_mut = 0
        for _ in range(max(n_steps, 1)):
            n = int(current.num_nodes)
            edges = _get_valid_edges(current)
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from(edges)
            best_score = float("-inf")
            best_first_graph = current
            non_edges = _get_non_edges(current, max_count=15)
            for u, v in non_edges:
                # IG proxy: topological distance before adding.
                try:
                    dist = nx.shortest_path_length(G, u, v)
                except nx.NetworkXNoPath:
                    dist = n
                ig_bonus = self.ig_weight * float(dist)
                sim_graph = _add_edge(current, u, v)
                # Greedy rollout.
                for _h in range(self.horizon - 1):
                    sim_u_base = _graph_utility(sim_graph)
                    sim_non_edges = _get_non_edges(sim_graph, max_count=10)
                    sim_best = sim_graph
                    sim_best_u = sim_u_base
                    for su, sv in sim_non_edges:
                        sc = _add_edge(sim_graph, su, sv)
                        su_new = _graph_utility(sc)
                        if su_new > sim_best_u:
                            sim_best_u = su_new
                            sim_best = sc
                    sim_graph = sim_best
                final_u = _graph_utility(sim_graph)
                score = final_u + ig_bonus
                if score > best_score:
                    best_score = score
                    best_first_graph = _add_edge(current, u, v)
            if best_first_graph is not current:
                current = best_first_graph
                n_mut += 1
            history.append(_graph_utility(current))
        return BaselineResult(
            baseline_name=self.name,
            final_utility=history[-1],
            utility_history=history,
            n_mutations=n_mut,
            n_rejected=0,
            n_steps=n_steps,
            compute_cost=float(n_steps * 15 * self.horizon * 10),
            final_n_edges=_count_edges(current),
            final_n_nodes=int(current.num_nodes),
            metadata={"horizon": self.horizon, "ig_weight": self.ig_weight},
        )


# ---------------------------------------------------------------------------
# 11. Full v5.11 baseline.
# ---------------------------------------------------------------------------

class FullV511Baseline:
    """The complete v5.11 runtime as a baseline.

    This runs the full canonical 8-phase cycle with all v5.11 features
    active. It is the reference that v6 must beat.
    """
    name = "full_v511"

    def run(self, graph: GraphBuffers, config, seed: int, n_steps: int) -> BaselineResult:
        from ..runtime import LGAERuntime, RuntimeConfig
        from ..runtime.runtime_config import RuntimeConfig as RTC

        rt_config = RTC()
        runtime = LGAERuntime(graph=graph, config=config, runtime_config=rt_config)
        history = []
        n_mut = 0
        n_rejected = 0
        for i in range(max(n_steps, 1)):
            result = runtime.step()
            history.append(float(result.utility_after))
            if result.committed:
                n_mut += 1
            else:
                n_rejected += 1
        final_graph = runtime._engine.graph
        return BaselineResult(
            baseline_name=self.name,
            final_utility=history[-1] if history else 0.0,
            utility_history=history,
            n_mutations=n_mut,
            n_rejected=n_rejected,
            n_steps=n_steps,
            compute_cost=float(n_steps * 100),  # approximate
            final_n_edges=_count_edges(final_graph),
            final_n_nodes=int(final_graph.num_nodes),
            metadata={"runtime_config": str(rt_config)},
        )


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------

ALL_V6_BASELINES: dict[str, BaselineRunner] = {
    "fixed_topology": FixedTopologyBaseline(),
    "random_rewiring": RandomRewiringBaseline(),
    "greedy": GreedyBaseline(),
    "curvature_only": CurvatureOnlyBaseline(),
    "fosr": FoSRBaseline(),
    "borf": BORFBaseline(),
    "effective_resistance": EffectiveResistanceBaseline(),
    "one_step_counterfactual": OneStepCounterfactualBaseline(),
    "mpc": MPCBaseline(),
    "mpc_with_ig": MPCWithIGBaseline(),
    "full_v511": FullV511Baseline(),
}
