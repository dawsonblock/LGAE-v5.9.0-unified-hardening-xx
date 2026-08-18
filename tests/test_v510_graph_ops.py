"""v5.10 Phase 33: tensor-based graph ops (no NetworkX) tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3.runtime import (
    compute_degrees, get_neighbors, build_adjacency_matrix,
    connected_components, shortest_path_length, count_triangles,
    graph_diameter,
)


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(edges, dtype=torch.long).T


def test_compute_degrees():
    ei = _edge_index([(0, 1), (1, 2), (2, 0)])
    degrees = compute_degrees(ei, 3)
    assert degrees.tolist() == [2, 2, 2]


def test_compute_degrees_empty():
    degrees = compute_degrees(torch.zeros(2, 0, dtype=torch.long), 5)
    assert degrees.tolist() == [0, 0, 0, 0, 0]


def test_get_neighbors():
    ei = _edge_index([(0, 1), (0, 2), (1, 2)])
    neighbors = get_neighbors(ei, 0)
    assert set(neighbors.tolist()) == {1, 2}


def test_get_neighbors_empty():
    neighbors = get_neighbors(torch.zeros(2, 0, dtype=torch.long), 0)
    assert len(neighbors) == 0


def test_build_adjacency_matrix():
    ei = _edge_index([(0, 1), (1, 2)])
    adj = build_adjacency_matrix(ei, 3)
    assert adj[0, 1] == 1.0
    assert adj[1, 0] == 1.0
    assert adj[1, 2] == 1.0
    assert adj[0, 2] == 0.0
    assert adj[0, 0] == 0.0  # no self-loops


def test_connected_components():
    ei = _edge_index([(0, 1), (2, 3)])
    labels = connected_components(ei, 4)
    assert labels[0] == labels[1]  # 0 and 1 are connected
    assert labels[2] == labels[3]  # 2 and 3 are connected
    assert labels[0] != labels[2]  # different components


def test_connected_components_empty():
    labels = connected_components(torch.zeros(2, 0, dtype=torch.long), 3)
    assert labels.tolist() == [0, 1, 2]  # each node is its own component


def test_shortest_path_length():
    ei = _edge_index([(0, 1), (1, 2), (2, 3)])
    assert shortest_path_length(ei, 4, 0, 3) == 3
    assert shortest_path_length(ei, 4, 0, 0) == 0


def test_shortest_path_length_no_path():
    ei = _edge_index([(0, 1), (2, 3)])
    assert shortest_path_length(ei, 4, 0, 3) == -1


def test_count_triangles():
    # Triangle: 0-1-2-0
    ei = _edge_index([(0, 1), (1, 2), (2, 0)])
    counts = count_triangles(ei, 3)
    # Each node is in 1 triangle.
    assert counts.tolist() == [1, 1, 1]


def test_count_triangles_no_triangles():
    ei = _edge_index([(0, 1), (1, 2)])  # path, no triangle
    counts = count_triangles(ei, 3)
    assert counts.tolist() == [0, 0, 0]


def test_graph_diameter():
    ei = _edge_index([(0, 1), (1, 2), (2, 3)])
    assert graph_diameter(ei, 4) == 3


def test_graph_diameter_disconnected():
    ei = _edge_index([(0, 1), (2, 3)])
    assert graph_diameter(ei, 4) == -1
