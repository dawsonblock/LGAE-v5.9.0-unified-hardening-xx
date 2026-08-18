"""v5.10 Phase 36: batched counterfactuals tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3.runtime import (
    CounterfactualResult, batched_apply_actions, batched_compute_utilities,
    batched_counterfactual_eval, select_best_counterfactual,
)


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(edges, dtype=torch.long).T


def _utility_fn(ei: torch.Tensor) -> float:
    """Simple utility: number of edges."""
    return float(ei.shape[1])


def test_batched_apply_actions_add_edge():
    ei = _edge_index([(0, 1), (1, 2)])
    actions = [("add_edge", {"u": 0, "v": 2})]
    results = batched_apply_actions(ei, actions, 3)
    assert len(results) == 1
    assert results[0].shape[1] == 3  # original 2 + 1 new


def test_batched_apply_actions_prune_edge():
    ei = _edge_index([(0, 1), (1, 2)])
    actions = [("prune_edge", {"u": 0, "v": 1})]
    results = batched_apply_actions(ei, actions, 3)
    assert len(results) == 1
    assert results[0].shape[1] == 1  # original 2 - 1 removed


def test_batched_apply_actions_multiple():
    ei = _edge_index([(0, 1)])
    actions = [
        ("add_edge", {"u": 1, "v": 2}),
        ("prune_edge", {"u": 0, "v": 1}),
        ("add_edge", {"u": 0, "v": 2}),
    ]
    results = batched_apply_actions(ei, actions, 3)
    assert len(results) == 3


def test_batched_compute_utilities():
    eis = [_edge_index([(0, 1), (1, 2)]), _edge_index([(0, 1)])]
    utilities, valid = batched_compute_utilities(eis, _utility_fn, 3)
    assert utilities.tolist() == [2.0, 1.0]
    assert valid.tolist() == [True, True]


def test_batched_compute_utilities_empty():
    utilities, valid = batched_compute_utilities([], _utility_fn, 3)
    assert len(utilities) == 0
    assert len(valid) == 0


def test_batched_counterfactual_eval():
    ei = _edge_index([(0, 1), (1, 2)])
    result = batched_counterfactual_eval(
        edge_index=ei,
        candidate_ids=["c1", "c2"],
        actions=[("add_edge", {"u": 0, "v": 2}), ("prune_edge", {"u": 0, "v": 1})],
        utility_fn=_utility_fn,
        num_nodes=3,
    )
    assert len(result.candidate_ids) == 2
    assert result.utilities[0] == 3.0  # 3 edges after add
    assert result.utilities[1] == 1.0  # 1 edge after prune


def test_batched_counterfactual_eval_to_log():
    result = CounterfactualResult(
        candidate_ids=["c1", "c2"],
        utilities=torch.tensor([1.0, 2.0]),
        valid=torch.tensor([True, False]),
    )
    log = result.to_log()
    assert log["n_candidates"] == 2
    assert log["utilities"] == [1.0, 2.0]
    assert log["valid"] == [True, False]


def test_select_best_counterfactual():
    result = CounterfactualResult(
        candidate_ids=["c1", "c2", "c3"],
        utilities=torch.tensor([0.5, 0.9, 0.3]),
        valid=torch.tensor([True, True, True]),
    )
    assert select_best_counterfactual(result) == "c2"


def test_select_best_counterfactual_with_invalid():
    result = CounterfactualResult(
        candidate_ids=["c1", "c2"],
        utilities=torch.tensor([0.5, 0.9]),
        valid=torch.tensor([True, False]),
    )
    assert select_best_counterfactual(result) == "c1"


def test_select_best_counterfactual_empty():
    result = CounterfactualResult(
        candidate_ids=[],
        utilities=torch.tensor([]),
        valid=torch.tensor([], dtype=torch.bool),
    )
    assert select_best_counterfactual(result) is None


def test_select_best_counterfactual_all_invalid():
    result = CounterfactualResult(
        candidate_ids=["c1", "c2"],
        utilities=torch.tensor([0.5, 0.9]),
        valid=torch.tensor([False, False]),
    )
    assert select_best_counterfactual(result) is None
