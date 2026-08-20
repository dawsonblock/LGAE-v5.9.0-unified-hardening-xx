"""Topology actions for exp7.2 (re-export from exp7.1 pattern)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TopologyActionType(str, Enum):
    ADD_ROUTE = "add_route"
    REMOVE_ROUTE = "remove_route"
    REWEIGHT_ROUTE = "reweight_route"
    BYPASS_NODE = "bypass_node"


@dataclass
class TopologyAction:
    action_type: TopologyActionType
    source: Optional[str] = None
    destination: Optional[str] = None
    weight: Optional[float] = None
    node_id: Optional[str] = None
    reason: str = ""

    @property
    def action_id(self) -> str:
        if self.action_type == TopologyActionType.BYPASS_NODE:
            return f"{self.action_type.value}:{self.node_id}"
        return f"{self.action_type.value}:{self.source}->{self.destination}"

    def apply(self, topology) -> bool:
        if self.action_type == TopologyActionType.ADD_ROUTE:
            if self.source and self.destination:
                topology.add_edge(self.source, self.destination, self.weight or 1.0)
                return True
        elif self.action_type == TopologyActionType.REMOVE_ROUTE:
            if self.source and self.destination:
                return topology.remove_edge(self.source, self.destination)
        elif self.action_type == TopologyActionType.REWEIGHT_ROUTE:
            if self.source and self.destination and self.weight is not None:
                return topology.reweight_edge(self.source, self.destination, self.weight)
        elif self.action_type == TopologyActionType.BYPASS_NODE:
            if self.node_id:
                count = topology.bypass_node(self.node_id)
                return count > 0
        return False

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type.value,
            "source": self.source,
            "destination": self.destination,
            "weight": self.weight,
            "node_id": self.node_id,
            "reason": self.reason,
            "action_id": self.action_id,
        }


def generate_candidate_actions(topology) -> list[TopologyAction]:
    """Generate candidate topology mutations."""
    candidates = []
    node_ids = topology.get_node_ids()

    # ADD_ROUTE: try adding missing edges.
    for src in node_ids:
        for dst in node_ids:
            if src == dst:
                continue
            if not topology.has_edge(src, dst):
                candidates.append(TopologyAction(
                    action_type=TopologyActionType.ADD_ROUTE,
                    source=src, destination=dst, weight=0.5,
                    reason=f"potential route {src}->{dst}",
                ))

    # REWEIGHT_ROUTE: try adjusting existing edges.
    for edge in topology.edges.values():
        if not edge.is_active:
            continue
        candidates.append(TopologyAction(
            action_type=TopologyActionType.REWEIGHT_ROUTE,
            source=edge.source, destination=edge.destination,
            weight=min(2.0, edge.weight * 1.5),
            reason=f"increase weight {edge.edge_id}",
        ))
        candidates.append(TopologyAction(
            action_type=TopologyActionType.REWEIGHT_ROUTE,
            source=edge.source, destination=edge.destination,
            weight=max(0.1, edge.weight * 0.5),
            reason=f"decrease weight {edge.edge_id}",
        ))

    # REMOVE_ROUTE: try removing edges with high failure rates.
    edge_telemetry = topology.get_edge_telemetry()
    for eid, tel in edge_telemetry.items():
        if tel.messages_sent > 2 and tel.failure_propagation > tel.successful_downstream:
            edge = topology.edges.get(eid)
            if edge and edge.is_active:
                candidates.append(TopologyAction(
                    action_type=TopologyActionType.REMOVE_ROUTE,
                    source=edge.source, destination=edge.destination,
                    reason=f"high failure rate on {eid}",
                ))

    # BYPASS_NODE: try bypassing non-essential nodes.
    bypassable = [n for n in node_ids if n not in ("planner", "verifier")]
    for node_id in bypassable:
        candidates.append(TopologyAction(
            action_type=TopologyActionType.BYPASS_NODE,
            node_id=node_id,
            reason=f"test bypass of {node_id}",
        ))

    return candidates
