"""v5.10 Phase 17: sheaf consistency certification tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3.runtime import SheafConsistencyResult, certify_sheaf_consistency


def test_sheaf_empty_graph():
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    result = certify_sheaf_consistency(edge_index=edge_index)
    assert result.cycle_count == 0
    assert result.is_flat is True
    assert result.max_inconsistency == 0.0


def test_sheaf_tree_is_flat():
    # A tree has no cycles, so the sheaf is trivially flat.
    edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    result = certify_sheaf_consistency(edge_index=edge_index)
    assert result.cycle_count == 0
    assert result.is_flat is True


def test_sheaf_cycle_without_gauge_is_flat():
    # A simple cycle without gauge data: structural inconsistency is 0.
    edges = [(0, 1), (1, 2), (2, 0)]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    result = certify_sheaf_consistency(edge_index=edge_index)
    assert result.cycle_count >= 1
    assert result.is_flat is True  # no gauge data -> structural only -> flat


def test_sheaf_cycle_with_identity_gauge_is_flat():
    edges = [(0, 1), (1, 2), (2, 0)]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    gauge = torch.eye(2).unsqueeze(0).repeat(3, 1, 1)  # identity for each edge
    result = certify_sheaf_consistency(edge_index=edge_index, gauge_connections=gauge)
    assert result.is_flat is True
    assert result.max_inconsistency < 1e-5


def test_sheaf_cycle_with_non_identity_gauge_is_not_flat():
    edges = [(0, 1), (1, 2), (2, 0)]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    # Non-identity gauge: rotation by 90 degrees.
    rot90 = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    gauge = rot90.unsqueeze(0).repeat(3, 1, 1)
    result = certify_sheaf_consistency(edge_index=edge_index, gauge_connections=gauge)
    # Three 90-degree rotations = 270 degrees, not identity.
    assert result.max_inconsistency > 0.1


def test_sheaf_result_to_log():
    result = SheafConsistencyResult(
        cycle_count=3, max_inconsistency=0.5, mean_inconsistency=0.2,
        is_flat=False, per_cycle_inconsistency=[0.1, 0.5, 0.0],
    )
    log = result.to_log()
    assert log["cycle_count"] == 3
    assert log["max_inconsistency"] == 0.5
    assert log["is_flat"] is False
    assert len(log["per_cycle_inconsistency"]) == 3


def test_sheaf_complete_graph_k4_has_cycles():
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    result = certify_sheaf_consistency(edge_index=edge_index)
    assert result.cycle_count > 0


def test_sheaf_max_cycles_bound():
    # Large graph with many cycles; verify we don't enumerate all.
    edges = [(i, j) for i in range(10) for j in range(i + 1, 10)]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    result = certify_sheaf_consistency(edge_index=edge_index, max_cycles=5)
    assert result.cycle_count <= 5
