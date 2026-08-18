"""Candidate retrieval evaluation (Phase 9).

The ANN mechanism becomes a real scalable proposal preselector. The most
important metric is oracle recall@K: if the best action is removed before
ranking, the learner cannot recover.

Metrics:
  - recall@K        : fraction of the brute-force top-K present in retrieval top-K
  - oracle_recall@K : 1 if the oracle-best action is in retrieval top-K, else 0
  - latency_ms      : retrieval wall time
  - memory_bytes    : index footprint estimate
  - retrieval_regret: utility(oracle_top1) - utility(retrieval_top1)

This module is evaluation tooling. It does not re-implement retrieval; it
scores any retriever that exposes a ``retrieve(query, k) -> list[index]``
interface against a brute-force oracle over a scored candidate set.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    recall_at_k: float
    oracle_recall_at_k: float
    latency_ms: float
    memory_bytes: int
    retrieval_regret: float
    k: int
    n_candidates: int

    def to_log(self) -> dict[str, Any]:
        return {
            "k": int(self.k),
            "n_candidates": int(self.n_candidates),
            "recall_at_k": float(self.recall_at_k),
            "oracle_recall_at_k": float(self.oracle_recall_at_k),
            "latency_ms": float(self.latency_ms),
            "memory_bytes": int(self.memory_bytes),
            "retrieval_regret": float(self.retrieval_regret),
        }


def brute_force_top_k(scores: Tensor, k: int) -> Tensor:
    """Return indices of the top-k scores, sorted descending."""
    k = int(min(k, scores.numel()))
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    return torch.argsort(scores, descending=True)[:k]


def evaluate_retrieval(
    scores: Tensor,
    retrieve_fn: Callable[[Tensor, int], Tensor],
    *,
    k: int,
    query: Tensor | None = None,
    memory_bytes: int = 0,
) -> RetrievalMetrics:
    """Evaluate a retriever against a brute-force oracle.

    Parameters
    ----------
    scores:
        1-D tensor of oracle scores for every candidate (higher is better).
    retrieve_fn:
        ``(query, k) -> retrieved_indices``. The retrieved indices are the
        retriever's top-K.
    k:
        Number of candidates to retrieve.
    query:
        Optional query embedding passed to the retriever.
    memory_bytes:
        Optional index footprint estimate for the memory metric.
    """
    if scores.ndim != 1:
        raise ValueError("scores must be 1-D")
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")
    n = int(scores.numel())
    oracle_topk = brute_force_top_k(scores, k)
    oracle_set = set(int(i) for i in oracle_topk.tolist())

    t0 = time.perf_counter()
    retrieved = retrieve_fn(query if query is not None else scores, k)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    retrieved = torch.as_tensor(retrieved, dtype=torch.long).flatten()[:k]
    retrieved_set = set(int(i) for i in retrieved.tolist())

    if oracle_set:
        recall = len(oracle_set & retrieved_set) / len(oracle_set)
    else:
        recall = 1.0

    oracle_best = int(oracle_topk[0].item()) if oracle_topk.numel() else -1
    oracle_recall = 1.0 if oracle_best in retrieved_set else 0.0

    ret_best = int(retrieved[0].item()) if retrieved.numel() else -1
    retrieval_regret = float(scores[oracle_best].item() - scores[ret_best]) if (oracle_best >= 0 and ret_best >= 0) else 0.0

    return RetrievalMetrics(
        recall_at_k=float(recall),
        oracle_recall_at_k=float(oracle_recall),
        latency_ms=float(latency_ms),
        memory_bytes=int(memory_bytes),
        retrieval_regret=float(retrieval_regret),
        k=k, n_candidates=n,
    )


@dataclass(slots=True)
class RetrievalBenchmark:
    """Accumulates retrieval metrics over many queries for a retriever."""
    name: str
    _metrics: list[RetrievalMetrics] = field(default_factory=list)

    def add(self, m: RetrievalMetrics) -> None:
        self._metrics.append(m)

    def evaluate(self, scores: Tensor, retrieve_fn: Callable[[Tensor, int], Tensor],
                 *, k: int, query: Tensor | None = None, memory_bytes: int = 0) -> RetrievalMetrics:
        m = evaluate_retrieval(scores, retrieve_fn, k=k, query=query, memory_bytes=memory_bytes)
        self.add(m)
        return m

    @property
    def count(self) -> int:
        return len(self._metrics)

    def aggregate(self) -> dict[str, Any]:
        if not self._metrics:
            return {"name": self.name, "count": 0}
        n = len(self._metrics)
        return {
            "name": self.name,
            "count": n,
            "mean_recall_at_k": sum(m.recall_at_k for m in self._metrics) / n,
            "mean_oracle_recall_at_k": sum(m.oracle_recall_at_k for m in self._metrics) / n,
            "mean_latency_ms": sum(m.latency_ms for m in self._metrics) / n,
            "mean_retrieval_regret": sum(m.retrieval_regret for m in self._metrics) / n,
            "p99_oracle_recall_at_k": _percentile([m.oracle_recall_at_k for m in self._metrics], 99),
        }

    def to_log(self) -> dict[str, Any]:
        agg = self.aggregate()
        agg["per_query"] = [m.to_log() for m in self._metrics]
        return agg


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((p / 100.0) * (len(s) - 1)))
    return float(s[idx])
