"""Atomic graph/cache transaction helpers.

The authoritative engine normally mutates a shadow graph and therefore never exposes
partial state to an ANN cache. This context manager is provided for external/in-place
workflows: graph tensors are restored atomically on rollback and attached neighbor
indices are generation-invalidated so stale leaves are rebuilt lazily on next query.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch

from .types import GraphBuffers
from .cache_coherence import ChangeKind, CommitEventBus, GraphCommitEvent, GraphReadCoordinator, normalize_edges


class GraphTransaction:
    def __init__(self, graph: GraphBuffers, indices: Iterable[Any] = (), *, event_bus: CommitEventBus | None = None, changes: ChangeKind = ChangeKind.ALL, read_coordinator: GraphReadCoordinator | None = None) -> None:
        self.graph = graph
        self.indices = list(indices)
        self.event_bus = event_bus
        self.changes = ChangeKind(changes)
        self.read_coordinator = read_coordinator
        self._write_epoch_open = False
        self._backup: GraphBuffers | None = None
        self._committed = False
        self._rolled_back = False

    def __enter__(self) -> "GraphTransaction":
        if self.read_coordinator is not None:
            self.read_coordinator.begin_write(); self._write_epoch_open = True
        self._backup = self.graph.clone()
        return self

    @torch.no_grad()
    def rollback(self) -> None:
        if self._backup is None or self._rolled_back:
            return
        b = self._backup
        self.graph.src.copy_(b.src)
        self.graph.dst.copy_(b.dst)
        self.graph.weight.copy_(b.weight)
        self.graph.valid.copy_(b.valid)
        if self.graph.length is not None and b.length is not None:
            self.graph.length.copy_(b.length)
        if self.graph.role is not None and b.role is not None:
            self.graph.role.copy_(b.role)
        if self.graph.slot_generation is not None and b.slot_generation is not None:
            self.graph.slot_generation.copy_(b.slot_generation)
        self.graph.version = int(b.version)
        for index in self.indices:
            if hasattr(index, "invalidate"):
                index.invalidate(graph_version=int(self.graph.version), reason="transaction_rollback")
            elif hasattr(index, "mark_dirty"):
                index.mark_dirty(graph_version=int(self.graph.version), reason="transaction_rollback")
        self._rolled_back = True
        if self.read_coordinator is not None and self._write_epoch_open:
            self.read_coordinator.end_write(); self._write_epoch_open = False

    def commit(self) -> None:
        self.graph.validate()
        if (self.event_bus is not None or self.read_coordinator is not None) and self._backup is not None and int(self.graph.version) == int(self._backup.version):
            # A direct in-place transaction may not have used GraphBuffers mutation
            # helpers, so establish exactly one new authoritative generation here.
            self.graph.bump_version()
        self._committed = True
        for index in self.indices:
            if hasattr(index, "invalidate"):
                index.invalidate(graph_version=int(self.graph.version), reason="transaction_commit")
            elif hasattr(index, "mark_dirty"):
                index.mark_dirty(graph_version=int(self.graph.version), reason="transaction_commit")
        if self.event_bus is not None:
            self.event_bus.publish(GraphCommitEvent(
                generation=int(self.graph.version), changes=self.changes, reason="transaction_commit"
            ))
        if self.read_coordinator is not None and self._write_epoch_open:
            self.read_coordinator.end_write(); self._write_epoch_open = False

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None or not self._committed:
            self.rollback()
        return False


def graph_transaction(graph: GraphBuffers, *indices: Any, event_bus: CommitEventBus | None = None, changes: ChangeKind = ChangeKind.ALL, read_coordinator: GraphReadCoordinator | None = None) -> GraphTransaction:
    return GraphTransaction(graph, indices, event_bus=event_bus, changes=changes, read_coordinator=read_coordinator)

class DeltaGraphTransaction:
    """Copy-on-write slot journal for bounded graph mutations.

    Unlike :class:`GraphTransaction`, this does not clone full graph buffers.
    Callers must mutate through the transaction methods so the original values
    of touched slots are journaled exactly once. This is intended for hot-loop
    internal mutation paths; the legacy snapshot transaction remains available
    for arbitrary third-party direct tensor edits that cannot be intercepted.
    """
    def __init__(self, graph: GraphBuffers, indices: Iterable[Any] = (), *, event_bus: CommitEventBus | None = None, read_coordinator: GraphReadCoordinator | None = None) -> None:
        self.graph = graph
        self.indices = list(indices)
        self.event_bus = event_bus
        self._journal: dict[int, tuple] = {}
        self._version_before = int(graph.version)
        self._committed = False
        self._rolled_back = False
        self.read_coordinator = read_coordinator
        self._write_epoch_open = False

    def __enter__(self) -> "DeltaGraphTransaction":
        if self.read_coordinator is not None:
            self.read_coordinator.begin_write(); self._write_epoch_open = True
        return self

    def _capture(self, slot: int) -> None:
        i = int(slot)
        if i < 0 or i >= self.graph.capacity:
            raise IndexError("edge slot out of range")
        if i in self._journal:
            return
        g = self.graph
        self._journal[i] = (
            int(g.src[i]), int(g.dst[i]), g.weight[i].clone(), bool(g.valid[i]),
            None if g.length is None else g.length[i].clone(),
            None if g.role is None else g.role[i].clone(),
            None if g.slot_generation is None else g.slot_generation[i].clone(),
        )

    @torch.no_grad()
    def set_slot(self, slot: int, *, src: int | None = None, dst: int | None = None,
                 weight: float | None = None, length: float | None = None,
                 valid: bool | None = None, role: int | None = None,
                 bump_generation: bool = True) -> None:
        self._capture(slot)
        i = int(slot); g = self.graph
        if src is not None: g.src[i] = int(src)
        if dst is not None: g.dst[i] = int(dst)
        if weight is not None: g.weight[i] = float(weight)
        if length is not None and g.length is not None: g.length[i] = float(length)
        if valid is not None: g.valid[i] = bool(valid)
        if role is not None and g.role is not None: g.role[i] = int(role)
        if bump_generation and g.slot_generation is not None:
            g.slot_generation[i] += 1

    @torch.no_grad()
    def rollback(self) -> None:
        if self._rolled_back:
            return
        g = self.graph
        for i, row in reversed(list(self._journal.items())):
            src, dst, weight, valid, length, role, generation = row
            g.src[i] = src; g.dst[i] = dst; g.weight[i].copy_(weight); g.valid[i] = valid
            if g.length is not None and length is not None: g.length[i].copy_(length)
            if g.role is not None and role is not None: g.role[i].copy_(role)
            if g.slot_generation is not None and generation is not None: g.slot_generation[i].copy_(generation)
        g.version = self._version_before
        for index in self.indices:
            if hasattr(index, "invalidate"):
                index.invalidate(graph_version=int(g.version), reason="delta_transaction_rollback")
            elif hasattr(index, "mark_dirty"):
                index.mark_dirty(graph_version=int(g.version), reason="delta_transaction_rollback")
        self._rolled_back = True
        if self.read_coordinator is not None and self._write_epoch_open:
            self.read_coordinator.end_write(); self._write_epoch_open = False

    def commit(self) -> None:
        self.graph.validate()
        self.graph.bump_version()
        self._committed = True
        for index in self.indices:
            if hasattr(index, "invalidate"):
                index.invalidate(graph_version=int(self.graph.version), reason="delta_transaction_commit")
            elif hasattr(index, "mark_dirty"):
                index.mark_dirty(graph_version=int(self.graph.version), reason="delta_transaction_commit")
        if self.event_bus is not None:
            changed_edges = []
            changed_nodes = set()
            changes = ChangeKind.NONE
            g = self.graph
            for i, before in self._journal.items():
                old_src, old_dst, old_weight, old_valid, old_length, old_role, _ = before
                new_valid = bool(g.valid[i])
                if bool(old_valid) != new_valid or int(old_src) != int(g.src[i]) or int(old_dst) != int(g.dst[i]):
                    changes |= ChangeKind.TOPOLOGY
                if float(old_weight) != float(g.weight[i]):
                    changes |= ChangeKind.WEIGHTS
                if old_length is not None and g.length is not None and float(old_length) != float(g.length[i]):
                    changes |= ChangeKind.LENGTHS
                if old_role is not None and g.role is not None and int(old_role) != int(g.role[i]):
                    changes |= ChangeKind.ROLES
                for u, v, valid in ((old_src, old_dst, bool(old_valid)), (int(g.src[i]), int(g.dst[i]), new_valid)):
                    if valid:
                        changed_edges.append((int(u), int(v))); changed_nodes.update((int(u), int(v)))
            self.event_bus.publish(GraphCommitEvent(
                generation=int(g.version), changes=changes or ChangeKind.ALL,
                changed_nodes=tuple(sorted(changed_nodes)), changed_edges=normalize_edges(changed_edges),
                reason="delta_transaction_commit",
            ))
        if self.read_coordinator is not None and self._write_epoch_open:
            self.read_coordinator.end_write(); self._write_epoch_open = False

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None or not self._committed:
            self.rollback()
        return False


def journaled_graph_transaction(graph: GraphBuffers, *indices: Any, event_bus: CommitEventBus | None = None, read_coordinator: GraphReadCoordinator | None = None) -> DeltaGraphTransaction:
    """Create an O(number-of-touched-slots) graph transaction."""
    return DeltaGraphTransaction(graph, indices, event_bus=event_bus, read_coordinator=read_coordinator)
