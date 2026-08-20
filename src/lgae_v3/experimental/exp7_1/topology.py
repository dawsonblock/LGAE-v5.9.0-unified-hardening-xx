"""AI Topology: routing graph for AI execution.

The topology is a directed weighted graph where:
  - Nodes are AINode instances (Planner, Worker, Critic, Verifier, Memory)
  - Edges represent message routes with weights
  - Edge weight controls routing probability/priority

LGAE controls only the topology — not the nodes themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .ai_node import AINode, NodeRole, NodeTelemetry


@dataclass
class EdgeTelemetry:
    """Telemetry emitted by an edge (route)."""
    edge_id: str
    source: str
    destination: str
    messages_sent: int = 0
    tokens_transferred: int = 0
    latency_contribution_ms: float = 0.0
    successful_downstream: int = 0
    failure_propagation: int = 0


@dataclass
class TopologyEdge:
    """A directed edge in the routing topology."""
    source: str  # node_id
    destination: str  # node_id
    weight: float = 1.0  # routing weight (higher = more likely to be used)
    active: bool = True  # if False, edge is bypassed (BYPASS_NODE)
    edge_id: str = ""

    def __post_init__(self):
        if not self.edge_id:
            self.edge_id = f"{self.source}->{self.destination}"

    @property
    def is_active(self) -> bool:
        return self.active and self.weight > 0.0


class AITopology:
    """The routing topology for AI execution.

    This is the structure that LGAE adapts. Nodes are fixed;
    edges can be added, removed, reweighted, or bypassed.
    """

    def __init__(self, nodes: dict[str, AINode], edges: list[TopologyEdge]) -> None:
        self.nodes = nodes
        self.edges: dict[str, TopologyEdge] = {}
        for edge in edges:
            self.edges[edge.edge_id] = edge
        self._edge_telemetry: dict[str, EdgeTelemetry] = {}
        for eid in self.edges:
            self._edge_telemetry[eid] = EdgeTelemetry(
                edge_id=eid,
                source=self.edges[eid].source,
                destination=self.edges[eid].destination,
            )

    def get_active_edges_from(self, node_id: str) -> list[TopologyEdge]:
        """Get all active edges originating from a node, sorted by weight."""
        active = [
            e for e in self.edges.values()
            if e.source == node_id and e.is_active
        ]
        return sorted(active, key=lambda e: -e.weight)

    def get_active_edges_to(self, node_id: str) -> list[TopologyEdge]:
        """Get all active edges terminating at a node."""
        return [
            e for e in self.edges.values()
            if e.destination == node_id and e.is_active
        ]

    def has_edge(self, source: str, destination: str) -> bool:
        """Check if an active edge exists."""
        eid = f"{source}->{destination}"
        edge = self.edges.get(eid)
        return edge is not None and edge.is_active

    def get_edge(self, source: str, destination: str) -> Optional[TopologyEdge]:
        """Get a specific edge."""
        return self.edges.get(f"{source}->{destination}")

    def add_edge(self, source: str, destination: str, weight: float = 1.0) -> TopologyEdge:
        """Add a new edge. Returns the created edge."""
        eid = f"{source}->{destination}"
        if eid in self.edges:
            # Reactivate if exists.
            self.edges[eid].active = True
            self.edges[eid].weight = weight
            return self.edges[eid]
        edge = TopologyEdge(source=source, destination=destination, weight=weight)
        self.edges[eid] = edge
        self._edge_telemetry[eid] = EdgeTelemetry(
            edge_id=eid, source=source, destination=destination,
        )
        return edge

    def remove_edge(self, source: str, destination: str) -> bool:
        """Remove an edge. Returns True if removed."""
        eid = f"{source}->{destination}"
        if eid in self.edges:
            self.edges[eid].active = False
            self.edges[eid].weight = 0.0
            return True
        return False

    def reweight_edge(self, source: str, destination: str, weight: float) -> bool:
        """Change an edge's weight. Returns True if successful."""
        eid = f"{source}->{destination}"
        if eid in self.edges:
            self.edges[eid].weight = weight
            self.edges[eid].active = weight > 0.0
            return True
        return False

    def bypass_node(self, node_id: str) -> int:
        """Bypass a node by deactivating all edges to/from it.

        Returns the number of edges deactivated.
        """
        count = 0
        for edge in self.edges.values():
            if edge.source == node_id or edge.destination == node_id:
                if edge.is_active:
                    edge.active = False
                    edge.weight = 0.0
                    count += 1
        return count

    def record_edge_usage(
        self,
        source: str,
        destination: str,
        tokens: int,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Record telemetry for an edge traversal."""
        eid = f"{source}->{destination}"
        if eid in self._edge_telemetry:
            tel = self._edge_telemetry[eid]
            tel.messages_sent += 1
            tel.tokens_transferred += tokens
            tel.latency_contribution_ms += latency_ms
            if success:
                tel.successful_downstream += 1
            else:
                tel.failure_propagation += 1

    def get_edge_telemetry(self) -> dict[str, EdgeTelemetry]:
        """Get all edge telemetry."""
        return dict(self._edge_telemetry)

    def to_adjacency_matrix(self) -> np.ndarray:
        """Convert to weighted adjacency matrix for structural analysis."""
        node_ids = sorted(self.nodes.keys())
        n = len(node_ids)
        idx = {nid: i for i, nid in enumerate(node_ids)}
        adj = np.zeros((n, n), dtype=np.float32)
        for edge in self.edges.values():
            if edge.is_active and edge.source in idx and edge.destination in idx:
                adj[idx[edge.source], idx[edge.destination]] = edge.weight
        return adj

    def get_node_ids(self) -> list[str]:
        """Get sorted node IDs."""
        return sorted(self.nodes.keys())

    def clone(self) -> "AITopology":
        """Create a deep copy of the topology."""
        edges = [
            TopologyEdge(
                source=e.source, destination=e.destination,
                weight=e.weight, active=e.active, edge_id=e.edge_id,
            )
            for e in self.edges.values()
        ]
        return AITopology(dict(self.nodes), edges)

    def summary(self) -> dict:
        """Get a summary of the topology state."""
        active_edges = sum(1 for e in self.edges.values() if e.is_active)
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_active_edges": active_edges,
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "source": e.source,
                    "destination": e.destination,
                    "weight": e.weight,
                    "active": e.is_active,
                }
                for e in self.edges.values()
            ],
        }


def create_default_topology(nodes: dict[str, AINode]) -> AITopology:
    """Create the default 5-node topology.

    Planner → Worker → Critic → Verifier
    Memory ↔ Planner
    Memory ↔ Worker
    """
    edges = [
        TopologyEdge("planner", "worker", weight=1.0),
        TopologyEdge("worker", "critic", weight=1.0),
        TopologyEdge("critic", "verifier", weight=1.0),
        TopologyEdge("memory", "planner", weight=1.0),
        TopologyEdge("planner", "memory", weight=0.5),
        TopologyEdge("memory", "worker", weight=0.5),
        TopologyEdge("worker", "memory", weight=0.3),
        # Feedback loops
        TopologyEdge("critic", "worker", weight=0.3),  # retry on bad output
        TopologyEdge("verifier", "planner", weight=0.2),  # replan on failure
    ]
    return AITopology(nodes, edges)
