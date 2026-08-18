"""Candidate-union architecture (Phase 8).

The learned model is never limited to candidates it generates itself. The
unified candidate set is::

    A = A_learned ∪ A_FoSR ∪ A_ER ∪ A_Forman ∪ A_memory ∪ A_retrieval ∪ {NO_OP}

Every candidate gets a canonical identity::

    CandidateID = SHA256(canonical_state_id || canonical_action_representation)

Candidates are deduplicated deterministically by ``CandidateID`` and returned
in a deterministic order (sorted by id). We never rely on Python set or dict
iteration order for canonical behavior.

This builds on the existing ``ConcreteAction`` / ``merge_candidate_channels``
infrastructure in ``structural_intelligence`` / ``reasoning``. It adds the
state-bound canonical identity and the deterministic-order guarantee.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..executive import StructuralAction
from ..reasoning import ConcreteAction


def _canonical_action_repr(action: StructuralAction, target: dict[str, Any]) -> str:
    """Canonical, hash-stable representation of an action + target.

    Continuous values are quantized for deduplication. Edge endpoints are
    sorted. The representation is JSON with sorted keys so it is independent
    of dict insertion order.
    """
    if action == StructuralAction.NO_OP:
        return json.dumps({"action": action.value}, sort_keys=True, separators=(",", ":"))
    u = target.get("u")
    v = target.get("v")
    if u is not None and v is not None:
        u, v = sorted((int(u), int(v)))
    q = lambda x: None if x is None else round(float(x), 6)
    payload = {
        "action": action.value,
        "u": u,
        "v": v,
        "factor": q(target.get("factor")),
        "weight": q(target.get("weight")),
        "length": q(target.get("length")),
        "node": target.get("node"),
        "width": target.get("width"),
        "magnitude": q(target.get("magnitude")),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def candidate_id(state_id: str, action: StructuralAction, target: dict[str, Any]) -> str:
    """Canonical SHA-256 CandidateID binding state + action representation."""
    h = hashlib.sha256()
    h.update(str(state_id).encode("utf-8"))
    h.update(b"||")
    h.update(_canonical_action_repr(action, target).encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class Candidate:
    """A union candidate with canonical identity and source channel."""
    id: str
    action: StructuralAction
    target: dict[str, Any]
    channel: str
    prior_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_no_op(self) -> bool:
        return self.action == StructuralAction.NO_OP

    @classmethod
    def from_concrete(cls, concrete: ConcreteAction, *, state_id: str) -> "Candidate":
        cid = candidate_id(state_id, concrete.action, concrete.target)
        return cls(
            id=cid, action=concrete.action, target=dict(concrete.target),
            channel=str(concrete.channel), prior_score=float(concrete.prior_score),
            metadata=dict(concrete.metadata),
        )

    def to_concrete(self) -> ConcreteAction:
        return ConcreteAction(
            action=self.action, target=dict(self.target),
            channel=self.channel, prior_score=float(self.prior_score),
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class CandidateUnion:
    """Unified, deduplicated, deterministically-ordered candidate set.

    Channels are merged by canonical ``CandidateID``. The first occurrence of
    each id wins (channels added earlier take precedence). The output order is
    sorted by ``Candidate.id`` so it never depends on Python set/dict iteration.
    NO_OP is always present.
    """
    state_id: str
    _by_id: dict[str, Candidate] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)

    def add_channel(self, channel: str, candidates: Iterable[ConcreteAction]) -> None:
        for c in candidates:
            base = Candidate.from_concrete(c, state_id=self.state_id)
            # Always stamp the channel name from the caller so channel_counts
            # reflects the union source, not the original ConcreteAction channel.
            cand = Candidate(
                id=base.id, action=base.action, target=base.target,
                channel=str(channel), prior_score=base.prior_score, metadata=base.metadata,
            )
            if cand.id not in self._by_id:
                self._by_id[cand.id] = cand
                self._order.append(cand.id)

    def add_no_op(self) -> None:
        noop = ConcreteAction(action=StructuralAction.NO_OP, target={}, channel="no_op")
        self.add_channel("no_op", [noop])

    def candidates(self) -> list[Candidate]:
        """Return candidates in deterministic order (sorted by id)."""
        # Always include NO_OP.
        if not any(self._by_id[c].is_no_op for c in self._order):
            self.add_no_op()
        return [self._by_id[cid] for cid in sorted(self._by_id.keys())]

    def concrete_candidates(self) -> list[ConcreteAction]:
        return [c.to_concrete() for c in self.candidates()]

    @property
    def size(self) -> int:
        return len(self._by_id)

    def channel_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self._by_id.values():
            counts[c.channel] = counts.get(c.channel, 0) + 1
        return counts

    def to_log(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "size": self.size,
            "channel_counts": self.channel_counts,
            "ids": sorted(self._by_id.keys()),
        }


def build_candidate_union(
    state_id: str,
    *,
    channels: dict[str, Sequence[ConcreteAction] | None],
    include_no_op: bool = True,
) -> CandidateUnion:
    """Build a unified candidate set from named channels.

    Channel names are arbitrary (e.g. "learned", "fosr", "er", "forman",
    "memory", "retrieval"). ``None`` / empty channels are skipped. The union
    always includes NO_OP unless ``include_no_op`` is False.
    """
    union = CandidateUnion(state_id=state_id)
    # Iterate channels in sorted name order for deterministic precedence.
    for name in sorted(channels.keys()):
        cands = channels[name]
        if cands:
            union.add_channel(name, cands)
    if include_no_op:
        union.add_no_op()
    return union
