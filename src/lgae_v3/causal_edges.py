"""v5.1 Causal edge semantics.

Distinguishes between association edges and causal edges in the graph.
Association edges capture statistical correlation; causal edges capture
directed influence. This distinction matters because:

1. Pruning an association edge changes representation but not causality.
2. Pruning a causal edge changes the actual data-generating process.
3. Counterfactual reasoning requires knowing which edges are causal.

The module provides:
- EdgeSemantics: enum for edge types (ASSOCIATION, CAUSAL, BIDIRECTIONAL)
- CausalEdgeRegistry: tracks edge semantics with metadata
- causal_intervention: do-calculus style intervention on causal edges
- causal_path_analysis: finds causal paths between nodes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections import defaultdict, deque

import torch
from torch import Tensor

from .version import VERSION


class EdgeSemantics(Enum):
    """Semantic type of an edge."""
    ASSOCIATION = "association"   # Statistical correlation, undirected
    CAUSAL = "causal"             # Directed cause → effect
    BIDIRECTIONAL = "bidirectional"  # Both directions (feedback loop)


@dataclass
class CausalEdge:
    """A single edge with causal semantics."""
    src: int
    dst: int
    semantics: EdgeSemantics
    confidence: float = 1.0       # How confident we are in the causal interpretation
    mechanism: str = ""           # Optional description of the causal mechanism
    metadata: dict[str, Any] = field(default_factory=dict)


class CausalEdgeRegistry:
    """Registry of edge causal semantics.

    Tracks which edges are associations, which are causal, and the
    direction of causality. Supports interventions and path analysis.
    """

    def __init__(self):
        self._edges: dict[tuple[int, int], CausalEdge] = {}
        self._causal_parents: dict[int, set[int]] = defaultdict(set)
        self._causal_children: dict[int, set[int]] = defaultdict(set)

    def register(
        self,
        src: int,
        dst: int,
        semantics: EdgeSemantics = EdgeSemantics.ASSOCIATION,
        confidence: float = 1.0,
        mechanism: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CausalEdge:
        """Register or replace an edge with causal semantics.

        Replacement first removes any causal-parent/child contribution from
        the old semantic record so the cached causal graph cannot retain stale
        structure after an edge is reclassified as association.
        """
        old = self._edges.get((src, dst))
        if old is not None:
            if old.semantics in (EdgeSemantics.CAUSAL, EdgeSemantics.BIDIRECTIONAL):
                self._causal_parents[old.dst].discard(old.src)
                self._causal_children[old.src].discard(old.dst)
            if old.semantics == EdgeSemantics.BIDIRECTIONAL:
                self._causal_parents[old.src].discard(old.dst)
                self._causal_children[old.dst].discard(old.src)
        edge = CausalEdge(
            src=src, dst=dst, semantics=semantics,
            confidence=confidence, mechanism=mechanism,
            metadata=metadata or {},
        )
        self._edges[(src, dst)] = edge

        if semantics == EdgeSemantics.CAUSAL:
            self._causal_parents[dst].add(src)
            self._causal_children[src].add(dst)
        elif semantics == EdgeSemantics.BIDIRECTIONAL:
            self._causal_parents[dst].add(src)
            self._causal_children[src].add(dst)
            self._causal_parents[src].add(dst)
            self._causal_children[dst].add(src)

        return edge

    def get(self, src: int, dst: int) -> CausalEdge | None:
        """Get edge semantics for a specific edge."""
        return self._edges.get((src, dst))

    def is_causal(self, src: int, dst: int) -> bool:
        """Check if an edge is causal (src → dst)."""
        edge = self._edges.get((src, dst))
        if edge is None:
            return False
        return edge.semantics in (EdgeSemantics.CAUSAL, EdgeSemantics.BIDIRECTIONAL)

    def is_association(self, src: int, dst: int) -> bool:
        """Check if an edge is purely associational."""
        edge = self._edges.get((src, dst))
        if edge is None:
            return True  # Default: association
        return edge.semantics == EdgeSemantics.ASSOCIATION

    def causal_parents(self, node: int) -> set[int]:
        """Get the causal parents of a node."""
        return self._causal_parents.get(node, set())

    def causal_children(self, node: int) -> set[int]:
        """Get the causal children of a node."""
        return self._causal_children.get(node, set())

    def causal_paths(self, src: int, dst: int) -> list[list[int]]:
        """Find all causal paths from src to dst.

        Uses BFS to find all directed causal paths.
        """
        if src == dst:
            return [[src]]

        paths: list[list[int]] = []
        queue: deque[list[int]] = deque([[src]])
        visited_in_path: set[tuple[int, ...]] = set()

        while queue:
            path = queue.popleft()
            current = path[-1]
            path_key = tuple(path)
            if path_key in visited_in_path:
                continue
            visited_in_path.add(path_key)

            for child in self._causal_children.get(current, set()):
                if child == dst:
                    paths.append(path + [child])
                elif child not in path:  # Avoid cycles
                    queue.append(path + [child])

        return paths

    def intervene(
        self,
        node: int,
        value: Tensor,
        z: Tensor,
    ) -> Tensor:
        """Perform a do-intervention: set node's value and propagate.

        do(node := value) cuts all incoming causal edges to node and
        sets its value. The effect propagates through causal children.

        Args:
            node: The node to intervene on
            value: The value to set
            z: Current latent states [N, D]

        Returns:
            Updated latent states after intervention propagation
        """
        z_new = z.clone()
        z_new[node] = value

        # Propagate through causal children (BFS) using a linear influence
        # model: each child shifts toward the parent's new value by a
        # damping factor that decreases with depth.
        queue: deque[tuple[int, int]] = deque([(node, 0)])
        visited = {node}
        max_depth = z.shape[0]
        damping = 0.5  # Influence strength per hop

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for child in self._causal_children.get(current, set()):
                if child in visited:
                    continue
                visited.add(child)
                # Linear influence: child moves toward parent's new value
                # by a depth-discounted factor.
                influence = damping ** (depth + 1)
                z_new[child] = z_new[child] + influence * (z_new[current] - z_new[child])
                queue.append((child, depth + 1))

        return z_new

    def counterfactual(
        self,
        node: int,
        counterfactual_value: Tensor,
        z: Tensor,
    ) -> Tensor:
        """Compute a counterfactual: "what if node had value X instead?"

        This is similar to intervene() but preserves the factual values
        for non-causal parents of the intervened node.
        """
        return self.intervene(node, counterfactual_value, z)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the causal edge registry."""
        counts = {s.value: 0 for s in EdgeSemantics}
        for edge in self._edges.values():
            counts[edge.semantics.value] += 1
        return {
            "total_edges": len(self._edges),
            "semantic_counts": counts,
            "causal_nodes": len(self._causal_parents),
            "version": VERSION,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "edges": [
                {
                    "src": e.src, "dst": e.dst,
                    "semantics": e.semantics.value,
                    "confidence": e.confidence,
                    "mechanism": e.mechanism,
                }
                for e in self._edges.values()
            ],
            "version": VERSION,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CausalEdgeRegistry":
        """Deserialize from dictionary."""
        registry = cls()
        for e in data.get("edges", []):
            registry.register(
                src=e["src"], dst=e["dst"],
                semantics=EdgeSemantics(e["semantics"]),
                confidence=e.get("confidence", 1.0),
                mechanism=e.get("mechanism", ""),
            )
        return registry


def infer_causality_from_temporal(
    z_history: Tensor,
    src: int,
    dst: int,
    lag: int = 1,
    threshold: float = 0.05,
) -> EdgeSemantics:
    """Predictive temporal-direction test inspired by Granger causality.

    This is still a diagnostic heuristic rather than a full structural causal
    model, but unlike the previous cosine-mean proxy it performs the defining
    predictive comparison: does adding ``src`` history reduce held-out linear
    prediction error for ``dst`` beyond ``dst`` history alone?
    """
    T, N, D = z_history.shape
    if T < max(lag + 4, 8):
        return EdgeSemantics.ASSOCIATION

    def improvement(cause: int, effect: int) -> float:
        x_self = z_history[:-lag, effect].reshape(T - lag, -1).to(torch.float64)
        x_cause = z_history[:-lag, cause].reshape(T - lag, -1).to(torch.float64)
        y = z_history[lag:, effect].reshape(T - lag, -1).to(torch.float64)
        n = x_self.shape[0]
        split = max(3, int(0.7 * n))
        if split >= n:
            return 0.0

        def fit_mse(x: Tensor) -> float:
            ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
            X = torch.cat([x, ones], dim=1)
            Xtr, Xte = X[:split], X[split:]
            ytr, yte = y[:split], y[split:]
            ridge = 1e-5 * torch.eye(Xtr.shape[1], dtype=X.dtype, device=X.device)
            beta = torch.linalg.solve(Xtr.T @ Xtr + ridge, Xtr.T @ ytr)
            pred = Xte @ beta
            return float((pred - yte).square().mean().item())

        base = fit_mse(x_self)
        full = fit_mse(torch.cat([x_self, x_cause], dim=1))
        return (base - full) / max(abs(base), 1e-12)

    forward = improvement(src, dst)
    backward = improvement(dst, src)
    if forward > float(threshold) and forward > backward + float(threshold):
        return EdgeSemantics.CAUSAL
    return EdgeSemantics.ASSOCIATION


def temporal_direction_heuristic(*args, **kwargs) -> EdgeSemantics:
    """Explicit name for the predictive temporal diagnostic."""
    return infer_causality_from_temporal(*args, **kwargs)

