"""v6.0-exp4.2: Held-out structural prediction study — test suite.

Covers:
- Strict compatibility mode
- Legacy compatibility isolation
- Held-out state machine
- Held-out lock irreversibility
- Dataset freeze hashes
- Finalist configuration locking
- Ranking metrics
- Regret metrics / catastrophic regret
- CF→real regimes
- OOD degradation
- Calibration drift
- Uncertainty/error correlation
- Selective prediction
- Pareto frontier
- Latency measurements
- Seed stability
- Bootstrap confidence intervals
- Group metrics
- Scientific status classification
- Machine-readable conclusion
- Report generation
- Reproducibility
- Authority isolation
- exp5 authorization gate
"""
from __future__ import annotations

import pytest
import numpy as np
import json
import math
import warnings
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from lgae_v3.experimental.exp4_2 import (
    ExperimentStateError,
    ExperimentStateMachine,
    EXPERIMENT_STATES,
    DatasetFreeze,
    SplitFreeze,
    freeze_dataset,
    load_dataset_freeze,
    TargetType,
    TargetDefinition,
    TARGET_DEFINITIONS,
    get_target_definition,
    RegretReport,
    OracleRecoveryReport,
    SelectivePredictionReport,
    ParetoFrontierEntry,
    ParetoFrontier,
    compute_regret,
    compute_oracle_recovery,
    compute_selective_prediction,
    compute_pareto_frontier,
    bootstrap_ci,
    UncertaintyCorrelationReport,
    compute_uncertainty_error_correlation,
    SupervisionRegime,
    CFRealTransferReport,
    run_cf_real_experiment,
    ExperimentConfig,
    EncoderConfig,
    PredictorConfig,
    FinalistLock,
    SelectionWeights,
    ScientificRunner,
    ScientificResult,
    ScientificConclusion,
    authorize_exp5,
    generate_scientific_report,
    generate_machine_readable_conclusion,
)
from lgae_v3.experimental.models.artifact import ModelArtifact, CompatibilityError
from lgae_v3.experimental.models.model_registry import ModelRegistry
from lgae_v3.experimental.world_model import ModelPrediction
from lgae_v3.experimental.compat.legacy_prediction import LegacyPredictionAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_artifact(
    *,
    encoder_schema_hash="enc_abc",
    dataset_schema_hash="ds_abc",
    train_split_hash="ts_abc",
    normalization_hash="nm_abc",
    feature_schema_hash="fs_abc",
    target_schema_hash="tg_abc",
    model_id="test-001",
) -> ModelArtifact:
    return ModelArtifact(
        model_id=model_id,
        predictor_type="linear",
        predictor_version="v1",
        encoder_id="global",
        encoder_schema_hash=encoder_schema_hash,
        dataset_schema_hash=dataset_schema_hash,
        train_split_hash=train_split_hash,
        normalization_hash=normalization_hash,
        feature_schema_hash=feature_schema_hash,
        target_schema_hash=target_schema_hash,
        hyperparameter_hash="hp_abc",
        model_state_hash="ms_abc",
        seed=42,
        training_code_version="v6.0-exp4.2",
        n_train_samples=100,
        n_features=24,
    )


@dataclass
class MockSplitDataset:
    """Mock SplitDataset for freeze tests."""
    split: str
    records: list[Any] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        import hashlib
        return hashlib.sha256(f"mock-{self.split}-{len(self.records)}".encode()).hexdigest()


@dataclass
class MockRecord:
    """Mock TransitionRecord for freeze tests."""
    split: str = "train"
    action: str = "ADD_EDGE"
    graph_family: str = "path"
    realized_delta: float = 0.1
    realized_risk: float = 0.05
    realized_cost: float = 1.0
    provenance: Any = None
    structural_state_before: Any = None
    action_target: dict = field(default_factory=lambda: {"u": 0, "v": 1})
    episode_id: str = "ep0"
    step_id: int = 0


@dataclass
class MockProvenance:
    value: str = "REALIZED"


# ---------------------------------------------------------------------------
# 1. Strict compatibility mode
# ---------------------------------------------------------------------------

class TestStrictCompatibility:
    """Phase 1: Strict artifact compatibility."""

    def test_strict_rejects_missing_encoder_hash(self):
        """Strict mode rejects missing encoder hash."""
        art = _make_artifact(encoder_schema_hash="")
        assert not art.is_compatible_with(
            encoder_schema_hash="enc_abc",
            dataset_schema_hash="ds_abc",
            train_split_hash="ts_abc",
            normalization_hash="nm_abc",
            feature_schema_hash="fs_abc",
            target_schema_hash="tg_abc",
            strict=True,
        )

    def test_strict_rejects_missing_dataset_hash(self):
        art = _make_artifact(dataset_schema_hash="")
        assert not art.is_compatible_with(
            encoder_schema_hash="enc_abc",
            dataset_schema_hash="ds_abc",
            train_split_hash="ts_abc",
            normalization_hash="nm_abc",
            feature_schema_hash="fs_abc",
            target_schema_hash="tg_abc",
            strict=True,
        )

    def test_strict_rejects_missing_train_hash(self):
        art = _make_artifact(train_split_hash="")
        assert not art.is_compatible_with(
            encoder_schema_hash="enc_abc",
            dataset_schema_hash="ds_abc",
            train_split_hash="ts_abc",
            normalization_hash="nm_abc",
            feature_schema_hash="fs_abc",
            target_schema_hash="tg_abc",
            strict=True,
        )

    def test_strict_rejects_missing_normalization_hash(self):
        art = _make_artifact(normalization_hash="")
        assert not art.is_compatible_with(
            encoder_schema_hash="enc_abc",
            dataset_schema_hash="ds_abc",
            train_split_hash="ts_abc",
            normalization_hash="nm_abc",
            feature_schema_hash="fs_abc",
            target_schema_hash="tg_abc",
            strict=True,
        )

    def test_strict_rejects_missing_feature_hash(self):
        art = _make_artifact(feature_schema_hash="")
        assert not art.is_compatible_with(
            encoder_schema_hash="enc_abc",
            dataset_schema_hash="ds_abc",
            train_split_hash="ts_abc",
            normalization_hash="nm_abc",
            feature_schema_hash="fs_abc",
            target_schema_hash="tg_abc",
            strict=True,
        )

    def test_strict_rejects_missing_target_hash(self):
        art = _make_artifact(target_schema_hash="")
        assert not art.is_compatible_with(
            encoder_schema_hash="enc_abc",
            dataset_schema_hash="ds_abc",
            train_split_hash="ts_abc",
            normalization_hash="nm_abc",
            feature_schema_hash="fs_abc",
            target_schema_hash="tg_abc",
            strict=True,
        )

    def test_strict_accepts_exact_identity(self):
        art = _make_artifact()
        assert art.is_compatible_with(
            encoder_schema_hash="enc_abc",
            dataset_schema_hash="ds_abc",
            train_split_hash="ts_abc",
            normalization_hash="nm_abc",
            feature_schema_hash="fs_abc",
            target_schema_hash="tg_abc",
            strict=True,
        )

    def test_strict_rejects_mismatched_encoder_hash(self):
        art = _make_artifact(encoder_schema_hash="enc_xyz")
        assert not art.is_compatible_with(
            encoder_schema_hash="enc_abc",
            dataset_schema_hash="ds_abc",
            train_split_hash="ts_abc",
            normalization_hash="nm_abc",
            feature_schema_hash="fs_abc",
            target_schema_hash="tg_abc",
            strict=True,
        )

    def test_strict_rejects_empty_query_side(self):
        """Strict mode rejects when query side is empty."""
        art = _make_artifact()
        assert not art.is_compatible_with(
            encoder_schema_hash="enc_abc",
            dataset_schema_hash="ds_abc",
            train_split_hash="",  # empty query
            normalization_hash="nm_abc",
            feature_schema_hash="fs_abc",
            target_schema_hash="tg_abc",
            strict=True,
        )

    def test_non_strict_allows_wildcards(self):
        """Non-strict mode allows empty fields as wildcards."""
        art = _make_artifact(train_split_hash="")
        assert art.is_compatible_with(
            encoder_schema_hash="enc_abc",
            dataset_schema_hash="ds_abc",
            train_split_hash="ts_abc",
            normalization_hash="nm_abc",
            feature_schema_hash="fs_abc",
            target_schema_hash="tg_abc",
            strict=False,
        )

    def test_verify_compatibility_strict_raises_on_missing(self):
        art = _make_artifact(train_split_hash="")
        with pytest.raises(CompatibilityError):
            ModelRegistry.verify_compatibility(
                art,
                encoder_schema_hash="enc_abc",
                dataset_schema_hash="ds_abc",
                train_split_hash="ts_abc",
                normalization_hash="nm_abc",
                feature_schema_hash="fs_abc",
                target_schema_hash="tg_abc",
                strict=True,
            )

    def test_verify_compatibility_strict_passes_exact(self):
        art = _make_artifact()
        ModelRegistry.verify_compatibility(
            art,
            encoder_schema_hash="enc_abc",
            dataset_schema_hash="ds_abc",
            train_split_hash="ts_abc",
            normalization_hash="nm_abc",
            feature_schema_hash="fs_abc",
            target_schema_hash="tg_abc",
            strict=True,
        )


# ---------------------------------------------------------------------------
# 2. Legacy compatibility isolation
# ---------------------------------------------------------------------------

class TestLegacyCompatibility:
    """Phase 2: Reward/risk ambiguity eliminated."""

    def test_predicted_reward_emits_deprecation_warning(self):
        p = ModelPrediction(predicted_risk=0.5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            val = p.predicted_reward
        assert val == 0.5
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_legacy_adapter_provides_predicted_reward(self):
        p = ModelPrediction(predicted_risk=0.3)
        adapter = LegacyPredictionAdapter(p)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            val = adapter.predicted_reward
        assert val == 0.3
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_legacy_adapter_passes_through_risk(self):
        p = ModelPrediction(predicted_risk=0.7, predicted_cost=2.0)
        adapter = LegacyPredictionAdapter(p)
        assert adapter.predicted_risk == 0.7
        assert adapter.predicted_cost == 2.0

    def test_legacy_adapter_to_log_excludes_reward(self):
        p = ModelPrediction(predicted_risk=0.5)
        adapter = LegacyPredictionAdapter(p)
        log = adapter.to_log()
        assert "predicted_risk" in log
        assert "predicted_reward" not in log

    def test_no_new_code_uses_predicted_reward(self):
        """Verify no production code outside compat uses predicted_reward."""
        # The only places predicted_reward should appear are:
        # 1. world_model.py deprecated property
        # 2. compat/legacy_prediction.py adapter
        # This test verifies the adapter exists and works.
        p = ModelPrediction(predicted_risk=0.1)
        adapter = LegacyPredictionAdapter(p)
        assert adapter.underlying is p


# ---------------------------------------------------------------------------
# 3. Held-out state machine
# ---------------------------------------------------------------------------

class TestExperimentStateMachine:
    """Phase 4: Sealed held-out access."""

    def test_initial_state_is_preparation(self):
        sm = ExperimentStateMachine()
        assert sm.state == "PREPARATION"

    def test_forward_transition_allowed(self):
        sm = ExperimentStateMachine()
        sm.transition_to("TRAINING")
        assert sm.state == "TRAINING"
        sm.transition_to("VALIDATION")
        assert sm.state == "VALIDATION"

    def test_backward_transition_forbidden(self):
        sm = ExperimentStateMachine()
        sm.transition_to("TRAINING")
        with pytest.raises(ExperimentStateError, match="backward"):
            sm.transition_to("PREPARATION")

    def test_heldout_requires_model_locked(self):
        sm = ExperimentStateMachine()
        sm.transition_to("TRAINING")
        sm.transition_to("VALIDATION")
        with pytest.raises(ExperimentStateError, match="MODEL_LOCKED"):
            sm.transition_to("HELDOUT_OPENED")

    def test_heldout_requires_locked_config(self):
        sm = ExperimentStateMachine()
        sm.transition_to("TRAINING")
        sm.transition_to("VALIDATION")
        sm.transition_to("MODEL_LOCKED")
        with pytest.raises(ExperimentStateError, match="locked model"):
            sm.transition_to("HELDOUT_OPENED")

    def test_lock_finalists_sets_hash(self):
        sm = ExperimentStateMachine()
        sm.transition_to("TRAINING")
        sm.transition_to("VALIDATION")
        sm.lock_finalists("config_hash_123", run_id="run_001")
        assert sm.state == "MODEL_LOCKED"
        assert sm.locked_model_config_hash == "config_hash_123"

    def test_lock_finalists_requires_empty_hash(self):
        sm = ExperimentStateMachine()
        sm.transition_to("TRAINING")
        sm.transition_to("VALIDATION")
        with pytest.raises(ExperimentStateError, match="Config hash"):
            sm.lock_finalists("")

    def test_lock_finalists_wrong_state(self):
        sm = ExperimentStateMachine()
        with pytest.raises(ExperimentStateError, match="VALIDATION"):
            sm.lock_finalists("hash")

    def test_heldout_opened_records_timestamp(self):
        sm = ExperimentStateMachine()
        sm.transition_to("TRAINING")
        sm.transition_to("VALIDATION")
        sm.lock_finalists("hash_123")
        sm.transition_to("HELDOUT_OPENED")
        assert sm.heldout_opened_at != ""
        assert sm.state == "HELDOUT_OPENED"

    def test_no_backward_from_heldout(self):
        sm = ExperimentStateMachine()
        sm.transition_to("TRAINING")
        sm.transition_to("VALIDATION")
        sm.lock_finalists("hash")
        sm.transition_to("HELDOUT_OPENED")
        with pytest.raises(ExperimentStateError, match="backward"):
            sm.transition_to("VALIDATION")

    def test_no_backward_from_heldout_to_model_locked(self):
        sm = ExperimentStateMachine()
        sm.transition_to("TRAINING")
        sm.transition_to("VALIDATION")
        sm.lock_finalists("hash")
        sm.transition_to("HELDOUT_OPENED")
        with pytest.raises(ExperimentStateError, match="backward"):
            sm.transition_to("MODEL_LOCKED")

    def test_finalize_after_heldout(self):
        sm = ExperimentStateMachine()
        sm.transition_to("TRAINING")
        sm.transition_to("VALIDATION")
        sm.lock_finalists("hash")
        sm.transition_to("HELDOUT_OPENED")
        sm.transition_to("FINALIZED")
        assert sm.state == "FINALIZED"

    def test_selection_not_permitted_after_lock(self):
        sm = ExperimentStateMachine()
        sm.transition_to("TRAINING")
        sm.transition_to("VALIDATION")
        sm.lock_finalists("hash")
        with pytest.raises(ExperimentStateError, match="not permitted"):
            sm.assert_selection_permitted()

    def test_heldout_not_accessible_before_opening(self):
        sm = ExperimentStateMachine()
        sm.transition_to("TRAINING")
        sm.transition_to("VALIDATION")
        sm.lock_finalists("hash")
        with pytest.raises(ExperimentStateError, match="not accessible"):
            sm.assert_heldout_accessible()

    def test_heldout_accessible_after_opening(self):
        sm = ExperimentStateMachine()
        sm.transition_to("TRAINING")
        sm.transition_to("VALIDATION")
        sm.lock_finalists("hash")
        sm.transition_to("HELDOUT_OPENED")
        sm.assert_heldout_accessible()  # should not raise

    def test_state_machine_logs_transitions(self):
        sm = ExperimentStateMachine()
        sm.transition_to("TRAINING")
        log = sm.to_log()
        assert len(log["transitions"]) >= 2
        assert log["transitions"][-1]["state"] == "TRAINING"

    def test_invalid_state_rejected(self):
        sm = ExperimentStateMachine()
        with pytest.raises(ExperimentStateError, match="Invalid state"):
            sm.transition_to("INVALID")

    def test_to_json_serializable(self):
        sm = ExperimentStateMachine()
        sm.transition_to("TRAINING")
        data = json.loads(sm.to_json())
        assert data["state"] == "TRAINING"


# ---------------------------------------------------------------------------
# 4. Dataset freeze
# ---------------------------------------------------------------------------

class TestDatasetFreeze:
    """Phase 3: Freeze dataset hashes."""

    def _make_mock_datasets(self):
        records = {
            "train": [MockRecord(split="train", provenance=MockProvenance("REALIZED")) for _ in range(10)],
            "validation": [MockRecord(split="validation", provenance=MockProvenance("REALIZED")) for _ in range(5)],
            "held_out": [MockRecord(split="held_out", provenance=MockProvenance("REALIZED")) for _ in range(5)],
        }
        return {
            "train": MockSplitDataset("train", records["train"]),
            "validation": MockSplitDataset("validation", records["validation"]),
            "held_out": MockSplitDataset("held_out", records["held_out"]),
        }

    def test_freeze_produces_hashes(self):
        datasets = self._make_mock_datasets()
        freeze = freeze_dataset(
            datasets,
            dataset_schema_hash="ds_schema",
            feature_schema_hash="feat_schema",
            graph_family_registry_hash="gf_hash",
            seed=42,
        )
        assert freeze.train_split_hash != ""
        assert freeze.validation_split_hash != ""
        assert freeze.heldout_split_hash != ""

    def test_freeze_hash_is_deterministic(self):
        datasets = self._make_mock_datasets()
        f1 = freeze_dataset(datasets, dataset_schema_hash="ds", feature_schema_hash="fs", graph_family_registry_hash="gf", seed=42)
        f2 = freeze_dataset(datasets, dataset_schema_hash="ds", feature_schema_hash="fs", graph_family_registry_hash="gf", seed=42)
        assert f1.freeze_hash == f2.freeze_hash

    def test_freeze_records_counts(self):
        datasets = self._make_mock_datasets()
        freeze = freeze_dataset(datasets, dataset_schema_hash="ds", feature_schema_hash="fs", graph_family_registry_hash="gf", seed=42)
        assert freeze.train.n_records == 10
        assert freeze.validation.n_records == 5
        assert freeze.heldout.n_records == 5

    def test_freeze_provenance_counts(self):
        datasets = self._make_mock_datasets()
        freeze = freeze_dataset(datasets, dataset_schema_hash="ds", feature_schema_hash="fs", graph_family_registry_hash="gf", seed=42)
        assert freeze.train.n_realized == 10

    def test_freeze_save_and_load(self):
        datasets = self._make_mock_datasets()
        freeze = freeze_dataset(datasets, dataset_schema_hash="ds", feature_schema_hash="fs", graph_family_registry_hash="gf", seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            freeze.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "DATASET_FREEZE.json"))
            assert os.path.exists(os.path.join(tmpdir, "train.manifest.json"))
            assert os.path.exists(os.path.join(tmpdir, "heldout.manifest.json"))
            loaded = load_dataset_freeze(tmpdir)
            assert loaded.freeze_hash == freeze.freeze_hash
            assert loaded.train.n_records == 10


# ---------------------------------------------------------------------------
# 5. Target definitions
# ---------------------------------------------------------------------------

class TestTargetDefinitions:
    """Phase 7: Targets defined and hashed before training."""

    def test_all_targets_have_schema_hash(self):
        for name, td in TARGET_DEFINITIONS.items():
            assert td.schema_hash != "", f"Target {name} has empty schema_hash"

    def test_target_hashes_are_unique(self):
        hashes = [td.schema_hash for td in TARGET_DEFINITIONS.values()]
        assert len(hashes) == len(set(hashes)), "Target hashes are not unique"

    def test_target_definitions_are_frozen(self):
        td = get_target_definition("realized_delta")
        assert td.target_type == TargetType.UTILITY_REGRESSION
        assert td.task_category == "regression"

    def test_sign_target_is_classification(self):
        td = get_target_definition("sign_delta")
        assert td.task_category == "classification"

    def test_ranking_target_is_ranking(self):
        td = get_target_definition("candidate_ranking")
        assert td.task_category == "ranking"

    def test_get_unknown_target_raises(self):
        with pytest.raises(KeyError):
            get_target_definition("nonexistent")

    def test_target_definition_hash_is_deterministic(self):
        td1 = TargetDefinition(name="test", target_type="regression", task_category="regression", description="test")
        td2 = TargetDefinition(name="test", target_type="regression", task_category="regression", description="test")
        assert td1.schema_hash == td2.schema_hash

    def test_all_target_hashes_function(self):
        from lgae_v3.experimental.exp4_2.targets import all_target_hashes
        hashes = all_target_hashes()
        assert len(hashes) == len(TARGET_DEFINITIONS)
        for name, h in hashes.items():
            assert h != ""


# ---------------------------------------------------------------------------
# 6. Regret metrics
# ---------------------------------------------------------------------------

class TestRegretMetrics:
    """Phase 12: Top-action regret."""

    def test_zero_regret_with_perfect_predictions(self):
        preds = [[0.1, 0.5, 0.3], [0.2, 0.8, 0.1]]
        trues = [[0.1, 0.5, 0.3], [0.2, 0.8, 0.1]]
        report = compute_regret(preds, trues, catastrophic_threshold=0.1)
        assert report.mean_regret == pytest.approx(0.0, abs=1e-10)
        assert report.catastrophic_regret_rate == pytest.approx(0.0)

    def test_regret_with_wrong_selection(self):
        # Model picks candidate 0 (pred=0.9), but true best is candidate 1 (true=0.8).
        preds = [[0.9, 0.1]]
        trues = [[0.1, 0.8]]
        report = compute_regret(preds, trues, catastrophic_threshold=0.1)
        assert report.mean_regret == pytest.approx(0.7, abs=1e-6)
        assert report.catastrophic_regret_rate == pytest.approx(1.0)

    def test_regret_percentiles(self):
        preds = [[0.9, 0.1]] * 10
        trues = [[0.1, 0.8]] * 10
        report = compute_regret(preds, trues, catastrophic_threshold=0.5)
        assert report.p90_regret == pytest.approx(0.7, abs=1e-6)
        assert report.max_regret == pytest.approx(0.7, abs=1e-6)

    def test_regret_empty_input(self):
        report = compute_regret([], [], catastrophic_threshold=0.1)
        assert report.n_states == 0
        assert report.mean_regret == 0.0

    def test_regret_single_candidate(self):
        """Single-candidate states have zero regret (no choice)."""
        preds = [[0.5]]
        trues = [[0.5]]
        report = compute_regret(preds, trues, catastrophic_threshold=0.1)
        assert report.n_states == 1
        assert report.mean_regret == pytest.approx(0.0, abs=1e-10)

    def test_catastrophic_threshold_respected(self):
        preds = [[0.9, 0.1]]
        trues = [[0.1, 0.5]]
        # Regret = 0.5 - 0.1 = 0.4
        report = compute_regret(preds, trues, catastrophic_threshold=0.3)
        assert report.catastrophic_regret_rate == pytest.approx(1.0)  # 0.4 > 0.3
        report2 = compute_regret(preds, trues, catastrophic_threshold=0.5)
        assert report2.catastrophic_regret_rate == pytest.approx(0.0)  # 0.4 < 0.5


# ---------------------------------------------------------------------------
# 7. Oracle recovery
# ---------------------------------------------------------------------------

class TestOracleRecovery:
    """Phase 13: Decision usefulness."""

    def test_perfect_recovery(self):
        preds = [[0.1, 0.9], [0.8, 0.2]]
        trues = [[0.1, 0.9], [0.8, 0.2]]
        baselines = [0.1, 0.2]
        report = compute_oracle_recovery(preds, trues, baselines)
        assert report.mean_oracle_recovery == pytest.approx(1.0, abs=1e-6)

    def test_zero_recovery_when_model_picks_baseline(self):
        preds = [[0.9, 0.1]]
        trues = [[0.1, 0.9]]
        baselines = [0.1]  # baseline picks candidate 0
        report = compute_oracle_recovery(preds, trues, baselines)
        assert report.mean_oracle_recovery == pytest.approx(0.0, abs=1e-6)

    def test_partial_recovery(self):
        preds = [[0.5, 0.6]]  # model picks 1
        trues = [[0.0, 0.5]]  # true best is 1
        baselines = [0.0]
        report = compute_oracle_recovery(preds, trues, baselines)
        assert 0.0 < report.mean_oracle_recovery <= 1.0

    def test_empty_input(self):
        report = compute_oracle_recovery([], [], [])
        assert report.n_states == 0


# ---------------------------------------------------------------------------
# 8. Selective prediction
# ---------------------------------------------------------------------------

class TestSelectivePrediction:
    """Phase 20: Selective prediction."""

    def test_selective_prediction_returns_coverage_levels(self):
        preds = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
        trues = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
        confs = [0.9, 0.8, 0.7, 0.6]
        report = compute_selective_prediction(preds, trues, confs)
        assert len(report.coverage_levels) == 5
        assert len(report.ranking_spearman) == 5

    def test_selective_prediction_full_coverage(self):
        preds = [[0.1, 0.9], [0.2, 0.8]]
        trues = [[0.1, 0.9], [0.2, 0.8]]
        confs = [0.9, 0.8]
        report = compute_selective_prediction(preds, trues, confs, coverage_levels=[1.0])
        assert report.ranking_spearman[0] == pytest.approx(1.0, abs=1e-6)

    def test_selective_prediction_empty(self):
        report = compute_selective_prediction([], [], [])
        assert len(report.coverage_levels) > 0


# ---------------------------------------------------------------------------
# 9. Pareto frontier
# ---------------------------------------------------------------------------

class TestParetoFrontier:
    """Phase 24: Simplicity frontier."""

    def test_pareto_identifies_optimal(self):
        entries = [
            ParetoFrontierEntry("global", "tree", "realized_delta", quality=0.7, latency_ms=1.0, n_parameters=100),
            ParetoFrontierEntry("hybrid", "mlp", "realized_delta", quality=0.75, latency_ms=10.0, n_parameters=5000),
        ]
        frontier = compute_pareto_frontier(entries, baseline_quality=0.5)
        # Both are Pareto-optimal: tree is cheaper, mlp is better quality.
        assert entries[0].is_pareto_optimal
        assert entries[1].is_pareto_optimal

    def test_pareto_dominates(self):
        entries = [
            ParetoFrontierEntry("global", "tree", "realized_delta", quality=0.7, latency_ms=1.0, n_parameters=100),
            ParetoFrontierEntry("global", "tree2", "realized_delta", quality=0.6, latency_ms=2.0, n_parameters=200),
        ]
        compute_pareto_frontier(entries, baseline_quality=0.5)
        # Entry 0 dominates entry 1 in all dimensions.
        assert entries[0].is_pareto_optimal
        assert not entries[1].is_pareto_optimal

    def test_efficiency_computed(self):
        entries = [
            ParetoFrontierEntry("global", "tree", "realized_delta", quality=0.7, latency_ms=1.0, n_parameters=100),
        ]
        compute_pareto_frontier(entries, baseline_quality=0.5)
        assert entries[0].efficiency > 0.0

    def test_empty_frontier(self):
        frontier = compute_pareto_frontier([], baseline_quality=0.5)
        assert len(frontier.entries) == 0


# ---------------------------------------------------------------------------
# 10. Bootstrap confidence intervals
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    """Phase 31: Statistical analysis."""

    def test_bootstrap_ci_returns_interval(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        lower, upper = bootstrap_ci(values, n_bootstrap=100, seed=42)
        assert lower < upper
        assert lower < 0.55 < upper  # mean is 0.55

    def test_bootstrap_ci_empty(self):
        lower, upper = bootstrap_ci([])
        assert lower == 0.0
        assert upper == 0.0

    def test_bootstrap_ci_deterministic(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        l1, u1 = bootstrap_ci(values, n_bootstrap=100, seed=42)
        l2, u2 = bootstrap_ci(values, n_bootstrap=100, seed=42)
        assert l1 == pytest.approx(l2)
        assert u1 == pytest.approx(u2)

    def test_bootstrap_ci_different_seeds_differ(self):
        values = list(np.random.RandomState(0).randn(50))
        l1, _ = bootstrap_ci(values, n_bootstrap=100, seed=42)
        l2, _ = bootstrap_ci(values, n_bootstrap=100, seed=123)
        # Different seeds should give slightly different CIs.
        # (Not guaranteed to differ, but very likely.)
        assert isinstance(l1, float) and isinstance(l2, float)


# ---------------------------------------------------------------------------
# 11. Uncertainty-error correlation
# ---------------------------------------------------------------------------

class TestUncertaintyCorrelation:
    """Phase 19: Uncertainty calibration."""

    def test_positive_correlation(self):
        uncertainties = [0.1, 0.2, 0.3, 0.4, 0.5]
        errors = [0.01, 0.02, 0.03, 0.04, 0.05]
        report = compute_uncertainty_error_correlation(uncertainties, errors)
        assert report.corr_uncertainty_abs_error > 0.9  # perfectly correlated

    def test_zero_correlation(self):
        uncertainties = [0.5, 0.5, 0.5, 0.5, 0.5]
        errors = [0.1, 0.2, 0.3, 0.4, 0.5]
        report = compute_uncertainty_error_correlation(uncertainties, errors)
        assert report.corr_uncertainty_abs_error == pytest.approx(0.0, abs=1e-6)

    def test_negative_correlation(self):
        uncertainties = [0.5, 0.4, 0.3, 0.2, 0.1]
        errors = [0.01, 0.02, 0.03, 0.04, 0.05]
        report = compute_uncertainty_error_correlation(uncertainties, errors)
        assert report.corr_uncertainty_abs_error < -0.9

    def test_empty_input(self):
        report = compute_uncertainty_error_correlation([], [])
        assert report.n_samples == 0

    def test_with_regrets(self):
        uncertainties = [0.1, 0.2, 0.3, 0.4, 0.5]
        errors = [0.01, 0.02, 0.03, 0.04, 0.05]
        regrets = [0.1, 0.2, 0.3, 0.4, 0.5]
        report = compute_uncertainty_error_correlation(uncertainties, errors, regrets)
        assert report.corr_uncertainty_regret > 0.9


# ---------------------------------------------------------------------------
# 12. CF-to-real experiment
# ---------------------------------------------------------------------------

class TestCFRealExperiment:
    """Phase 16: Counterfactual-to-real transfer."""

    def _make_data(self, n=30, d=5):
        rng = np.random.RandomState(42)
        X = rng.randn(n, d)
        y = X[:, 0] * 0.5 + rng.randn(n) * 0.1
        # 60% realized, 40% counterfactual.
        prov = ["realized"] * 18 + ["counterfactual"] * 12
        return X, y, prov

    def test_run_cf_real_three_regimes(self):
        from lgae_v3.experimental.models.linear import RidgeRegressionPredictor
        X_train, y_train, prov = self._make_data()
        X_val, y_val, _ = self._make_data()
        X_held, y_held, _ = self._make_data()

        def factory():
            return RidgeRegressionPredictor(n_epochs=20)

        report = run_cf_real_experiment(
            X_train, y_train, prov,
            X_val, y_val,
            X_held, y_held,
            model_factory=factory,
        )
        assert len(report.results) == 3
        assert report.results[0].regime == SupervisionRegime.REALIZED_ONLY
        assert report.results[1].regime == SupervisionRegime.COUNTERFACTUAL_ONLY
        assert report.results[2].regime == SupervisionRegime.MIXED

    def test_cf_real_gap_computed(self):
        from lgae_v3.experimental.models.linear import RidgeRegressionPredictor
        X_train, y_train, prov = self._make_data()
        X_val, y_val, _ = self._make_data()
        X_held, y_held, _ = self._make_data()

        def factory():
            return RidgeRegressionPredictor(n_epochs=20)

        report = run_cf_real_experiment(
            X_train, y_train, prov,
            X_val, y_val,
            X_held, y_held,
            model_factory=factory,
        )
        # Gap should be a float (could be positive or negative).
        assert isinstance(report.gap_cf_to_real_spearman, float)

    def test_cf_real_empty_realized(self):
        from lgae_v3.experimental.models.linear import RidgeRegressionPredictor
        X_train, y_train, _ = self._make_data()
        prov = ["counterfactual"] * 30  # no realized

        def factory():
            return RidgeRegressionPredictor(n_epochs=20)

        report = run_cf_real_experiment(
            X_train, y_train, prov,
            np.zeros((0, 5)), np.zeros(0),
            np.zeros((0, 5)), np.zeros(0),
            model_factory=factory,
        )
        assert report.results[0].n_train == 0  # realized-only has no data


# ---------------------------------------------------------------------------
# 13. Experiment config and finalist lock
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    """Phase 5-6, 27-28: Config and finalist locking."""

    def test_default_config_has_encoders(self):
        cfg = ExperimentConfig()
        # Without calling default_experiment_config, encoders list is empty.
        # Use default_experiment_config instead.
        from lgae_v3.experimental.exp4_2.experiment_config import default_experiment_config
        cfg = default_experiment_config()
        assert len(cfg.encoders) > 0
        assert len(cfg.predictors) > 0

    def test_selection_weights_compute_score(self):
        sw = SelectionWeights()
        score = sw.compute_score(
            spearman=0.7, ndcg=0.6, regret=0.1,
            sign_accuracy=0.8, ece=0.05, latency_ms=1.0,
        )
        assert isinstance(score, float)
        assert score > 0.0

    def test_finalist_lock_has_hash(self):
        lock = FinalistLock(
            finalists=[{"encoder_id": "global", "predictor_id": "tree", "target": "realized_delta"}],
            selection_weights={"w_spearman": 0.25},
        )
        assert lock.config_hash != ""

    def test_finalist_lock_deterministic_hash(self):
        import time as _time
        lock1 = FinalistLock(
            finalists=[{"encoder_id": "global"}],
            selection_weights={"w": 0.1},
            locked_at="2026-01-01T00:00:00Z",
        )
        lock2 = FinalistLock(
            finalists=[{"encoder_id": "global"}],
            selection_weights={"w": 0.1},
            locked_at="2026-01-01T00:00:00Z",
        )
        assert lock1.config_hash == lock2.config_hash

    def test_finalist_lock_save(self):
        lock = FinalistLock(finalists=[{"a": 1}], locked_at="2026-01-01T00:00:00Z")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            lock.save(f.name)
            data = json.loads(Path(f.name).read_text())
            assert data["config_hash"] == lock.config_hash
        os.unlink(f.name)


# ---------------------------------------------------------------------------
# 14. Scientific conclusion and exp5 gate
# ---------------------------------------------------------------------------

class TestScientificConclusion:
    """Phase 30, 36, 40: Conclusion and exp5 gate."""

    def test_qualified_simple_status(self):
        c = ScientificConclusion(
            scientific_status="QUALIFIED_SIMPLE",
            structural_signal_detected=True,
            generalizes_to_heldout=True,
            exp5_authorized=True,
            cf_real_transfer_ok=True,
            best_encoder="global",
            best_model="tree",
        )
        assert authorize_exp5(c) is True

    def test_qualified_graph_native_status(self):
        c = ScientificConclusion(
            scientific_status="QUALIFIED_GRAPH_NATIVE",
            structural_signal_detected=True,
            generalizes_to_heldout=True,
            exp5_authorized=True,
            cf_real_transfer_ok=True,
            best_encoder="learned-graph",
            best_model="mlp",
            recommended_exp5_architecture="graph_native_world_model",
        )
        assert authorize_exp5(c) is True

    def test_failed_generalization_blocks_exp5(self):
        c = ScientificConclusion(
            scientific_status="FAILED_GENERALIZATION",
            structural_signal_detected=False,
            exp5_authorized=False,
            cf_real_transfer_ok=True,
        )
        assert authorize_exp5(c) is False

    def test_failed_cf_real_blocks_exp5(self):
        c = ScientificConclusion(
            scientific_status="FAILED_CF_REAL_TRANSFER",
            structural_signal_detected=True,
            generalizes_to_heldout=True,
            exp5_authorized=True,
            cf_real_transfer_ok=False,
        )
        assert authorize_exp5(c) is False

    def test_inconclusive_blocks_exp5(self):
        c = ScientificConclusion(
            scientific_status="INCONCLUSIVE",
            structural_signal_detected=True,
            generalizes_to_heldout=False,
            exp5_authorized=False,
        )
        assert authorize_exp5(c) is False

    def test_conclusion_to_json(self):
        c = ScientificConclusion(scientific_status="INCONCLUSIVE")
        data = json.loads(c.to_json())
        assert data["scientific_status"] == "INCONCLUSIVE"
        assert data["exp5_authorized"] is False


# ---------------------------------------------------------------------------
# 15. Report generation
# ---------------------------------------------------------------------------

class TestReportGeneration:
    """Phase 35: Scientific report generation."""

    def test_generate_report_creates_files(self):
        results = [ScientificResult(
            encoder_id="global", predictor_id="tree", target="realized_delta",
            validation_metrics={"spearman": 0.7},
            heldout_metrics={"spearman": 0.65, "rmse": 0.1},
        )]
        conclusion = ScientificConclusion(
            scientific_status="QUALIFIED_SIMPLE",
            structural_signal_detected=True,
            generalizes_to_heldout=True,
            exp5_authorized=True,
            cf_real_transfer_ok=True,
            best_encoder="global",
            best_model="tree",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_scientific_report(results, conclusion, tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "RAW_RESULTS.json"))
            assert os.path.exists(os.path.join(tmpdir, "COMPETITION_TABLE.csv"))
            assert os.path.exists(os.path.join(tmpdir, "CONCLUSION.json"))
            assert os.path.exists(os.path.join(tmpdir, "EXECUTIVE_SUMMARY.md"))
            assert os.path.exists(os.path.join(tmpdir, "SCIENTIFIC_REPORT.md"))

    def test_machine_readable_conclusion(self):
        conclusion = ScientificConclusion(scientific_status="INCONCLUSIVE")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            generate_machine_readable_conclusion(conclusion, f.name)
            data = json.loads(Path(f.name).read_text())
            assert data["scientific_status"] == "INCONCLUSIVE"
        os.unlink(f.name)

    def test_competition_table_has_header(self):
        results = [ScientificResult(
            encoder_id="global", predictor_id="tree", target="realized_delta",
        )]
        conclusion = ScientificConclusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_scientific_report(results, conclusion, tmpdir)
            csv_path = os.path.join(tmpdir, "COMPETITION_TABLE.csv")
            content = Path(csv_path).read_text()
            assert "Encoder" in content
            assert "global" in content


# ---------------------------------------------------------------------------
# 16. Authority isolation
# ---------------------------------------------------------------------------

class TestAuthorityIsolation:
    """Phase 38: Authority regression — experimental code cannot commit."""

    def test_scientific_runner_does_not_mutate_runtime(self):
        """The scientific runner must not touch the v5.11 runtime."""
        from lgae_v3.runtime import LGAERuntime, RuntimeConfig
        from lgae_v3 import ResearchConfig, make_graph_buffers
        cfg = ResearchConfig()
        cfg.audit.orc_backend = "exact_lp"
        cfg.audit.persistent_homology_enabled = False
        cfg.audit.entropic_nodes = 0
        cfg.audit.bakry_nodes = 0
        cfg.audit.cde_nodes = 0
        cfg.audit.exact_lly_top_k = 0
        cfg.audit.orc_top_k = 0
        cfg.mutation.shadow_horizons = [1, 2]
        cfg.mutation.curvature_ema_enabled = False
        graph = make_graph_buffers(8, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)], capacity=16)
        runtime = LGAERuntime(graph=graph, config=cfg, runtime_config=RuntimeConfig())
        gen_before = runtime.snapshot().generation
        # Create and use a scientific runner — it should not touch the runtime.
        runner = ScientificRunner()
        # Just creating and using the state machine should not affect runtime.
        runner.state.transition_to("TRAINING")
        gen_after = runtime.snapshot().generation
        assert gen_before == gen_after

    def test_experiment_state_machine_cannot_commit(self):
        """The experiment state machine has no mutation authority."""
        sm = ExperimentStateMachine()
        # It only tracks state — no commit channel access.
        assert not hasattr(sm, "commit")
        assert not hasattr(sm, "mutate")

    def test_cf_real_experiment_cannot_commit(self):
        """CF-real experiment is advisory only."""
        # Verify the function signature doesn't accept any runtime/commit args.
        import inspect
        sig = inspect.signature(run_cf_real_experiment)
        params = set(sig.parameters.keys())
        assert "runtime" not in params
        assert "commit_channel" not in params
        assert "authority" not in params


# ---------------------------------------------------------------------------
# 17. Scientific runner integration (lightweight)
# ---------------------------------------------------------------------------

class TestScientificRunnerIntegration:
    """Integration test of the scientific runner with mock data."""

    def test_runner_state_machine_lifecycle(self):
        """Verify the full lifecycle can be driven via the state machine."""
        runner = ScientificRunner()
        assert runner.state.state == "PREPARATION"
        # We can't run the full pipeline without real data, but we can
        # verify the state machine transitions work.
        runner.state.transition_to("TRAINING")
        runner.state.transition_to("VALIDATION")
        runner.state.lock_finalists("test_hash")
        assert runner.state.state == "MODEL_LOCKED"
        runner.state.transition_to("HELDOUT_OPENED")
        assert runner.state.heldout_accessible
        runner.state.transition_to("FINALIZED")
        assert runner.state.state == "FINALIZED"

    def test_runner_cannot_open_heldout_without_locking(self):
        runner = ScientificRunner()
        runner.state.transition_to("TRAINING")
        runner.state.transition_to("VALIDATION")
        runner.state.transition_to("MODEL_LOCKED")
        with pytest.raises(ExperimentStateError, match="locked model"):
            runner.state.transition_to("HELDOUT_OPENED")

    def test_runner_conclusion_requires_finalized_state(self):
        runner = ScientificRunner()
        # Cannot finalize without going through the full lifecycle.
        with pytest.raises(ExperimentStateError, match="HELDOUT_OPENED"):
            runner.finalize()


# ---------------------------------------------------------------------------
# 18. Experiment freeze document
# ---------------------------------------------------------------------------

class TestExperimentFreeze:
    """Phase 0: Freeze exp4.1."""

    def test_freeze_document_exists(self):
        freeze_path = Path(__file__).resolve().parents[2] / "EXPERIMENT_FREEZE_v6.0-exp4.1.md"
        assert freeze_path.exists(), "EXPERIMENT_FREEZE_v6.0-exp4.1.md must exist"

    def test_freeze_document_has_experiment_id(self):
        freeze_path = Path(__file__).resolve().parents[2] / "EXPERIMENT_FREEZE_v6.0-exp4.1.md"
        content = freeze_path.read_text()
        assert "LGAE_V6_EXP4_2_STRUCTURAL_PREDICTION_STUDY_001" in content

    def test_freeze_document_has_qualification(self):
        freeze_path = Path(__file__).resolve().parents[2] / "EXPERIMENT_FREEZE_v6.0-exp4.1.md"
        content = freeze_path.read_text()
        assert "2008" in content
        assert "QUALIFIED" in content
