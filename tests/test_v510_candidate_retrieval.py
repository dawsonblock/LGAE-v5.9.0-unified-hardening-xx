"""v5.10 Phase 9: candidate retrieval evaluation tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3.runtime import (
    RetrievalMetrics, RetrievalBenchmark, evaluate_retrieval, brute_force_top_k,
)


def test_brute_force_top_k_returns_sorted_descending():
    scores = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5])
    top = brute_force_top_k(scores, 3)
    assert top.tolist() == [1, 3, 4]


def test_evaluate_retrieval_perfect_recall():
    scores = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5, 0.2])
    # Perfect retriever returns the oracle top-k.
    def retrieve(query, k):
        return brute_force_top_k(scores, k)
    m = evaluate_retrieval(scores, retrieve, k=3)
    assert m.recall_at_k == 1.0
    assert m.oracle_recall_at_k == 1.0
    assert m.retrieval_regret == pytest.approx(0.0, abs=1e-9)


def test_evaluate_retrieval_partial_recall():
    scores = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5, 0.2])
    # Retriever returns 2 of 3 oracle top-k.
    def retrieve(query, k):
        return torch.tensor([1, 4, 0])  # misses 3 (0.7)
    m = evaluate_retrieval(scores, retrieve, k=3)
    assert m.recall_at_k == pytest.approx(2 / 3, abs=1e-6)
    # Oracle best (index 1, score 0.9) is present -> oracle recall 1.
    assert m.oracle_recall_at_k == 1.0


def test_evaluate_retrieval_oracle_recall_zero_when_best_missing():
    scores = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5, 0.2])
    def retrieve(query, k):
        return torch.tensor([3, 4, 0])  # misses 1 (oracle best)
    m = evaluate_retrieval(scores, retrieve, k=3)
    assert m.oracle_recall_at_k == 0.0
    # Regret = oracle_best_score - retrieved_best_score = 0.9 - 0.7 = 0.2
    assert m.retrieval_regret == pytest.approx(0.2, abs=1e-6)


def test_evaluate_retrieval_rejects_invalid_inputs():
    scores = torch.tensor([0.1, 0.2])
    with pytest.raises(ValueError):
        evaluate_retrieval(scores, lambda q, k: torch.tensor([0]), k=0)
    with pytest.raises(ValueError):
        evaluate_retrieval(torch.tensor([[0.1, 0.2]]), lambda q, k: torch.tensor([0]), k=1)


def test_retrieval_benchmark_aggregates():
    bench = RetrievalBenchmark(name="hnsw")
    scores = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5])
    bench.evaluate(scores, lambda q, k: brute_force_top_k(scores, k), k=2)
    bench.evaluate(scores, lambda q, k: torch.tensor([3, 4]), k=2)  # misses oracle best 1
    agg = bench.aggregate()
    assert agg["count"] == 2
    # First query perfect oracle recall, second zero -> mean 0.5.
    assert agg["mean_oracle_recall_at_k"] == 0.5
    assert agg["mean_recall_at_k"] > 0.0


def test_retrieval_metrics_to_log():
    m = RetrievalMetrics(recall_at_k=0.5, oracle_recall_at_k=1.0, latency_ms=1.2,
                         memory_bytes=1024, retrieval_regret=0.0, k=5, n_candidates=100)
    log = m.to_log()
    assert log["k"] == 5 and log["n_candidates"] == 100
    assert log["recall_at_k"] == 0.5
