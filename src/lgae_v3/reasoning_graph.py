"""Typed parallel reasoning DAG for LGAE.

Nodes receive an immutable context mapping and return typed evidence objects.
The graph executor runs dependency-ready nodes concurrently and deterministically
reduces their outputs by node id.  It is orchestration, not an authority layer.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Callable, Mapping


@dataclass(slots=True, frozen=True)
class ReasoningEvidence:
    node_id: str
    evidence_type: str
    payload: dict[str, Any]
    confidence: float = 1.0
    source_hash: str | None = None

    @property
    def evidence_hash(self) -> str:
        blob = json.dumps({"node_id": self.node_id, "evidence_type": self.evidence_type, "payload": self.payload, "confidence": self.confidence, "source_hash": self.source_hash}, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()


@dataclass(slots=True)
class ReasoningNode:
    id: str
    fn: Callable[[Mapping[str, Any]], ReasoningEvidence]
    dependencies: tuple[str, ...] = ()


@dataclass(slots=True)
class ReasoningRun:
    run_id: str
    outputs: dict[str, ReasoningEvidence]
    execution_layers: list[list[str]]


class ReasoningGraph:
    def __init__(self, nodes: list[ReasoningNode], max_workers: int = 8):
        self.nodes = {n.id: n for n in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("reasoning node ids must be unique")
        self.max_workers = max(1, int(max_workers))
        for node in nodes:
            missing = set(node.dependencies) - set(self.nodes)
            if missing:
                raise ValueError(f"node {node.id} has missing dependencies: {sorted(missing)}")
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        indegree = {k: len(v.dependencies) for k, v in self.nodes.items()}
        ready = [k for k, d in indegree.items() if d == 0]
        seen = 0
        while ready:
            nid = ready.pop()
            seen += 1
            for other in self.nodes.values():
                if nid in other.dependencies:
                    indegree[other.id] -= 1
                    if indegree[other.id] == 0:
                        ready.append(other.id)
        if seen != len(self.nodes):
            raise ValueError("reasoning graph contains a cycle")

    def run(self, context: Mapping[str, Any]) -> ReasoningRun:
        outputs: dict[str, ReasoningEvidence] = {}
        pending = set(self.nodes)
        layers: list[list[str]] = []
        while pending:
            ready = sorted(nid for nid in pending if all(dep in outputs for dep in self.nodes[nid].dependencies))
            if not ready:
                raise RuntimeError("reasoning graph stalled")
            layers.append(ready)
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(ready))) as pool:
                futures = {}
                for nid in ready:
                    merged = dict(context)
                    merged["dependencies"] = MappingProxyType({dep: outputs[dep] for dep in self.nodes[nid].dependencies})
                    futures[pool.submit(self.nodes[nid].fn, MappingProxyType(merged))] = nid
                results: dict[str, ReasoningEvidence] = {}
                for fut in as_completed(futures):
                    nid = futures[fut]
                    result = fut.result()
                    if not isinstance(result, ReasoningEvidence) or result.node_id != nid:
                        raise TypeError(f"reasoning node {nid} violated its output contract")
                    results[nid] = result
            for nid in ready:  # deterministic reduce order
                outputs[nid] = results[nid]
                pending.remove(nid)
        run_blob = json.dumps({k: v.evidence_hash for k, v in sorted(outputs.items())}, sort_keys=True).encode()
        return ReasoningRun(hashlib.sha256(run_blob).hexdigest()[:24], outputs, layers)
