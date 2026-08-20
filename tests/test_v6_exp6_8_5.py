"""Tests for v6.0-exp6.8.5: Full Structural Advantage Features."""
import numpy as np
import pytest
import torch

from lgae_v3 import make_graph_buffers
from lgae_v3.experimental.exp6_8_5 import (
    GraphAdvantageRecord, generate_graph_advantage_records,
    build_features_for_records, run_exp6_8_5,
)
from lgae_v3.experimental.exp6_3.exact_mpc import ActionIdentity


class TestGraphRecords:
    """Test graph-storing advantage records."""

    def test_graph_record_creation(self):
        adj = np.zeros((5, 5), dtype=bool)
        adj[0, 1] = adj[1, 0] = True
        adj[1, 2] = adj[2, 1] = True

        record = GraphAdvantageRecord(
            state_id=0,
            adjacency=adj,
            n_nodes=5,
            z=torch.randn(5, 4),
            state_features=np.zeros(20, dtype=np.float32),
            objective_features=np.zeros(10, dtype=np.float32),
            baseline_action=("add_edge", 0, 3, {"weight": 1.0}),
            learned_action=("add_edge", 0, 4, {"weight": 1.0}),
            baseline_action_id=ActionIdentity.from_action(("add_edge", 0, 3, {"weight": 1.0})),
            learned_action_id=ActionIdentity.from_action(("add_edge", 0, 4, {"weight": 1.0})),
            baseline_q=10.0,
            learned_q=15.0,
            advantage=5.0,
            mechanism="connectivity_threshold",
            split="train",
            config_seed=42,
            threshold=2,
            lambda_bonus=30.0,
        )
        assert record.advantage == 5.0
        assert record.is_beneficial is True
        assert record.adjacency.shape == (5, 5)
        assert record.adjacency[0, 1] == True
        assert record.n_nodes == 5

    def test_graph_record_negative_advantage(self):
        record = GraphAdvantageRecord(
            state_id=0,
            adjacency=np.zeros((3, 3), dtype=bool),
            n_nodes=3,
            z=torch.randn(3, 4),
            state_features=np.zeros(20, dtype=np.float32),
            objective_features=np.zeros(10, dtype=np.float32),
            baseline_action=("add_edge", 0, 1, {}),
            learned_action=("add_edge", 0, 2, {}),
            baseline_action_id=ActionIdentity.from_action(("add_edge", 0, 1, {})),
            learned_action_id=ActionIdentity.from_action(("add_edge", 0, 2, {})),
            baseline_q=10.0,
            learned_q=5.0,
            advantage=-5.0,
            mechanism="connectivity_threshold",
            split="test",
            config_seed=42,
            threshold=2,
            lambda_bonus=30.0,
        )
        assert record.is_beneficial is False

    def test_generate_graph_advantage_records(self):
        """Test that we can generate records with stored graphs."""
        records = generate_graph_advantage_records(
            mechanism="connectivity_threshold",
            n_tasks=10,
            seed=42,
            split="train",
        )
        # Should generate at least some records.
        assert isinstance(records, list)
        if len(records) > 0:
            rec = records[0]
            assert rec.adjacency.shape[0] == rec.n_nodes
            assert rec.adjacency.shape[1] == rec.n_nodes
            assert rec.mechanism == "connectivity_threshold"
            assert rec.split == "train"

    def test_build_features_f1(self):
        """Test F1 feature extraction from graph records."""
        records = generate_graph_advantage_records(
            mechanism="connectivity_threshold",
            n_tasks=10,
            seed=42,
            split="train",
        )
        if len(records) < 2:
            pytest.skip("Not enough records generated")
        X, y, bq = build_features_for_records(records[:5], "F1_current")
        assert X.shape[0] == min(5, len(records))
        assert y.shape[0] == X.shape[0]
        assert bq.shape[0] == X.shape[0]

    def test_build_features_f4(self):
        """Test F4 feature extraction from graph records."""
        records = generate_graph_advantage_records(
            mechanism="connectivity_threshold",
            n_tasks=10,
            seed=42,
            split="train",
        )
        if len(records) < 2:
            pytest.skip("Not enough records generated")
        X, y, bq = build_features_for_records(records[:5], "F4_full")
        assert X.shape[0] == min(5, len(records))
        # F4 should have more features than F1.
        X_f1, _, _ = build_features_for_records(records[:5], "F1_current")
        assert X.shape[1] > X_f1.shape[1]


class TestExperimentRunner:
    """Test the experiment runner."""

    def test_run_exp6_8_5_smoke(self):
        """Smoke test with minimal data."""
        result = run_exp6_8_5(
            n_train_per_mechanism=200,
            n_calibration=20,
            n_test=20,
            data_sizes=[100],
            feature_levels=["F1_current", "F4_full"],
            mechanisms=["connectivity_threshold"],
        )
        assert result is not None
        assert result.decision in [
            "F4_MATERIALLY_IMPROVES",
            "F4_BEATS_F1_BUT_NO_LEARNING_CURVE",
            "F4_DOES_NOT_IMPROVE",
        ]
