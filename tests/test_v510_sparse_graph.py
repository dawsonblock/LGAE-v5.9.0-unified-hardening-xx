"""v5.10 Phase 34: sparse-first graph representation tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3.runtime import (
    SparseGraph, build_sparse_graph, sparse_adjacency_matrix,
    sparse_to_edge_index,
)


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(edges, dtype=torch.long).T


def test_build_sparse_graph_empty():
    sg = build_sparse_graph(torch.zeros(2, 0, dtype=torch.long), 5)
    assert sg.num_nodes == 5
    assert sg.num_edges == 0


def test_build_sparse_graph_basic():
    ei = _edge_index([(0, 1), (1, 2), (2, 0)])
    sg = build_sparse_graph(ei, 3)
    assert sg.num_nodes == 3
    assert sg.num_edges == 3


def test_sparse_graph_neighbors():
    ei = _edge_index([(0, 1), (0, 2), (1, 2)])
    sg = build_sparse_graph(ei, 3)
    neighbors = sg.neighbors(0)
    assert set(neighbors.tolist()) == {1, 2}


def test_sparse_graph_degree():
    ei = _edge_index([(0, 1), (0, 2), (1, 2)])
    sg = build_sparse_graph(ei, 3)
    assert sg.degree(0) == 2
    assert sg.degree(1) == 2
    assert sg.degree(2) == 2


def test_sparse_graph_degree_isolated_node():
    ei = _edge_index([(0, 1)])
    sg = build_sparse_graph(ei, 3)
    assert sg.degree(2) == 0  # isolated


def test_sparse_adjacency_matrix():
    ei = _edge_index([(0, 1), (1, 2)])
    sg = build_sparse_graph(ei, 3)
    adj = sparse_adjacency_matrix(sg)
    assert adj[0, 1] == 1.0
    assert adj[1, 0] == 1.0
    assert adj[1, 2] == 1.0
    assert adj[0, 2] == 0.0


def test_sparse_to_edge_index_roundtrip():
    ei = _edge_index([(0, 1), (1, 2), (2, 0)])
    sg = build_sparse_graph(ei, 3)
    ei_back = sparse_to_edge_index(sg)
    # Should have the same number of unique edges.
    assert ei_back.shape[1] == 3


def test_sparse_graph_to_log():
    ei = _edge_index([(0, 1), (1, 2)])
    sg = build_sparse_graph(ei, 3)
    log = sg.to_log()
    assert log["num_nodes"] == 3
    assert log["num_edges"] == 2
    assert log["has_values"] is False


def test_sparse_graph_num_edges_empty():
    sg = build_sparse_graph(torch.zeros(2, 0, dtype=torch.long), 0)
    assert sg.num_edges == 0


def test_sparse_graph_neighbors_isolated():
    ei = _edge_index([(0, 1)])
    sg = build_sparse_graph(ei, 3)
    neighbors = sg.neighbors(2)
    assert len(neighbors) == 0
