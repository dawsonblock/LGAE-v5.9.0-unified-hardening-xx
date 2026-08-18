"""Structural experience memory for LGAE.

This is a *derived* graph.  It can evolve, consolidate, prune or be rebuilt
from the immutable evidence ledger.  It is therefore never an authority for
what actually happened.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor

from .reasoning import ConcreteAction, CounterfactualOutcome
from .types import GraphBuffers


class MemoryKind(str, Enum):
    STATE = "state"
    DIAGNOSIS = "diagnosis"
    ACTION = "action"
    PREDICTION = "prediction"
    OUTCOME = "outcome"
    CERTIFICATE = "certificate"
    FAILURE = "failure"


@dataclass(slots=True)
class MemoryNode:
    id: str
    kind: MemoryKind
    features: tuple[float, ...]
    payload: dict[str, Any] = field(default_factory=dict)
    evidence_hash: str | None = None
    confidence: float = 1.0
    authority: float = 0.0
    access_count: int = 0


@dataclass(slots=True)
class MemoryEdge:
    src: str
    dst: str
    relation: str
    weight: float = 1.0


@dataclass(slots=True)
class MemoryMatch:
    node: MemoryNode
    similarity: float


class StructuralExperienceMemory:
    """Small dependency-free temporal/associative experience graph.

    Retrieval uses cosine similarity over explicit structural fingerprints.
    The interface deliberately permits replacing the index with ANN/Graphiti/
    SAGE-style storage later without changing reasoning-loop contracts.
    """

    def __init__(self, max_nodes: int = 100_000):
        self.max_nodes = int(max_nodes)
        self.nodes: dict[str, MemoryNode] = {}
        self.edges: list[MemoryEdge] = []
        self._order: list[str] = []

    @staticmethod
    def fingerprint(graph: GraphBuffers, z: Tensor) -> tuple[float, ...]:
        valid = graph.valid
        n = float(graph.num_nodes)
        e = float(graph.edge_count)
        if graph.edge_count:
            w = graph.weight[valid].detach().float().cpu()
            ell = graph.active_length().detach().float().cpu()
            w_mean, w_std = float(w.mean()), float(w.std(unbiased=False))
            l_mean, l_std = float(ell.mean()), float(ell.std(unbiased=False))
        else:
            w_mean = w_std = l_mean = l_std = 0.0
        if z.numel():
            norms = torch.linalg.vector_norm(z.detach().float().cpu(), dim=-1)
            z_mean, z_std = float(norms.mean()), float(norms.std(unbiased=False))
        else:
            z_mean = z_std = 0.0
        density = (2.0 * e / (n * (n - 1.0))) if n > 1 else 0.0
        return (math.log1p(n), math.log1p(e), density, w_mean, w_std, l_mean, l_std, z_mean, z_std)

    @staticmethod
    def _id(kind: MemoryKind, payload: dict[str, Any], evidence_hash: str | None) -> str:
        blob = json.dumps({"kind": kind.value, "payload": payload, "evidence": evidence_hash}, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:24]

    @staticmethod
    def authority_score(*, evidence_hash: str | None, confidence: float, accepted: bool | None = None) -> float:
        """Evidence-grounded authority in [0,1].

        Cryptographically linked evidence dominates free-floating traces; confidence
        modulates rather than overrides provenance. A governor rejection can still be
        authoritative evidence about an unsafe action, so acceptance is only a small
        calibration term rather than a hard filter.
        """
        provenance = 0.75 if evidence_hash else 0.20
        conf = max(0.0, min(1.0, float(confidence)))
        decision = 0.05 if accepted is True else (0.03 if accepted is False else 0.0)
        return max(0.0, min(1.0, provenance + 0.20 * conf + decision))

    def add_node(self, kind: MemoryKind, features: Sequence[float], payload: dict[str, Any], *, evidence_hash: str | None = None, confidence: float = 1.0) -> MemoryNode:
        nid = self._id(kind, payload, evidence_hash)
        accepted = payload.get("accepted") if isinstance(payload, dict) else None
        authority = self.authority_score(evidence_hash=evidence_hash, confidence=confidence, accepted=accepted if isinstance(accepted, bool) else None)
        node = MemoryNode(nid, kind, tuple(float(x) for x in features), dict(payload), evidence_hash, float(confidence), authority)
        prior = self.nodes.get(nid)
        if prior is not None and prior.authority > node.authority:
            # Low-authority replays may increase access but cannot overwrite a
            # stronger grounded memory slot.
            prior.access_count += 1
            return prior
        self.nodes[nid] = node
        if nid not in self._order:
            self._order.append(nid)
        while len(self._order) > self.max_nodes:
            old = self._order.pop(0)
            self.nodes.pop(old, None)
            self.edges = [e for e in self.edges if e.src != old and e.dst != old]
        return node

    def link(self, src: MemoryNode | str, dst: MemoryNode | str, relation: str, weight: float = 1.0) -> None:
        s = src.id if isinstance(src, MemoryNode) else src
        d = dst.id if isinstance(dst, MemoryNode) else dst
        if s in self.nodes and d in self.nodes:
            self.edges.append(MemoryEdge(s, d, str(relation), float(weight)))

    def record_outcome(self, graph: GraphBuffers, z: Tensor, outcome: CounterfactualOutcome, *, evidence_hash: str | None = None, prediction: dict[str, float] | None = None) -> tuple[MemoryNode, MemoryNode]:
        fp = self.fingerprint(graph, z)
        state = self.add_node(MemoryKind.STATE, fp, {"graph_hash": outcome.graph_hash_before}, evidence_hash=evidence_hash)
        action = self.add_node(MemoryKind.ACTION, fp, {
            "action": outcome.candidate.action.value,
            "target": outcome.candidate.target,
            "channel": outcome.candidate.channel,
        }, evidence_hash=evidence_hash)
        result = self.add_node(MemoryKind.OUTCOME, fp, {
            "delta_utility": outcome.delta_utility,
            "accepted": outcome.accepted_by_governor,
            "decision": outcome.decision,
            "graph_hash_after": outcome.graph_hash_after,
        }, evidence_hash=evidence_hash)
        self.link(state, action, "PROPOSED")
        self.link(action, result, "CAUSED", max(abs(float(outcome.delta_utility)), 1e-6))
        if prediction is not None:
            pred = self.add_node(MemoryKind.PREDICTION, fp, dict(prediction), evidence_hash=evidence_hash)
            self.link(action, pred, "PREDICTED")
            self.link(pred, result, "VERIFIED_BY")
        return state, result

    @staticmethod
    def _cos(a: Sequence[float], b: Sequence[float]) -> float:
        ta = torch.tensor(a, dtype=torch.float32)
        tb = torch.tensor(b, dtype=torch.float32)
        denom = float(torch.linalg.vector_norm(ta) * torch.linalg.vector_norm(tb))
        return 0.0 if denom <= 1e-12 else float(torch.dot(ta, tb) / denom)

    def retrieve(self, graph: GraphBuffers, z: Tensor, *, k: int = 8, kinds: set[MemoryKind] | None = None) -> list[MemoryMatch]:
        q = self.fingerprint(graph, z)
        matches: list[MemoryMatch] = []
        for node in self.nodes.values():
            if kinds is not None and node.kind not in kinds:
                continue
            sim = self._cos(q, node.features)
            matches.append(MemoryMatch(node, sim))
        matches.sort(key=lambda m: (m.similarity, m.node.authority, m.node.confidence), reverse=True)
        for match in matches[:k]:
            match.node.access_count += 1
        return matches[:k]

    def action_prior(self, graph: GraphBuffers, z: Tensor, action: ConcreteAction, *, k: int = 16) -> tuple[float, int]:
        """Return experience prior for a concrete action type/target class."""
        nearby = self.retrieve(graph, z, k=k, kinds={MemoryKind.STATE})
        if not nearby:
            return 0.0, 0
        state_ids = {m.node.id: m.similarity for m in nearby}
        weighted = 0.0
        mass = 0.0
        by_src = [e for e in self.edges if e.src in state_ids and e.relation == "PROPOSED"]
        for edge in by_src:
            action_node = self.nodes.get(edge.dst)
            if action_node is None or action_node.payload.get("action") != action.action.value:
                continue
            outcomes = [x for x in self.edges if x.src == action_node.id and x.relation == "CAUSED"]
            for oe in outcomes:
                out = self.nodes.get(oe.dst)
                if out is None:
                    continue
                sim = state_ids[edge.src]
                weighted += sim * float(out.payload.get("delta_utility", 0.0))
                mass += abs(sim)
        return (weighted / mass if mass > 1e-12 else 0.0, int(len(by_src)))

    def state_dict(self) -> dict[str, Any]:
        return {
            "nodes": [{"id": n.id, "kind": n.kind.value, "features": list(n.features), "payload": n.payload, "evidence_hash": n.evidence_hash, "confidence": n.confidence, "authority": n.authority, "access_count": n.access_count} for n in self.nodes.values()],
            "edges": [{"src": e.src, "dst": e.dst, "relation": e.relation, "weight": e.weight} for e in self.edges],
        }
