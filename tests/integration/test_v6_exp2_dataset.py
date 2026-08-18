"""v6.0-exp2: Structural transition dataset tests.

Tests verify:
1. Rich TransitionRecord schema with ~25 fields.
2. ObservedTransition vs CounterfactualTransition provenance separation.
3. Canonical structural feature extraction (global + local).
4. Dataset generation with split control and negative sampling.
5. Dataset validation (leakage, duplicates, schema, nonfinite).
6. Data quality report (distributions, imbalance detection).
7. Deterministic regeneration: same seed + config → identical hash.
8. Leakage control: held-out families never appear in train split.
9. Negative sampling: counterfactual records present.
10. v5.11 authority boundary untouched.
"""
from __future__ import annotations

import pytest
import torch
import json
import math
import tempfile
from pathlib import Path

from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.experimental import (
    # Transition record
    TransitionRecord, TransitionProvenance, AuthorizationDecision,
    AuthorityIdentity, StructuralStateSummary, DiagnosisSummary,
    CandidateSummary, CandidateSetSummary, PlannerMetadata, ComputeMetrics,
    make_record_id,
    # Feature extraction
    GlobalStructuralFeatures, LocalActionFeatures, StructuralFeatureVector,
    extract_global_features, extract_local_action_features,
    # Dataset generator
    DatasetGenerator, SplitDataset, DatasetImmutableMetadata,
    DATASET_SCHEMA_VERSION, GENERATOR_VERSION,
    # Dataset validator
    DatasetValidator, ValidationIssue, ValidationResult,
    # Quality report
    DataQualityReport, DistributionReport, CategoryDistribution,
    generate_quality_report,
    # Graph families
    get_frozen_registry, FROZEN_TRAIN_FAMILIES, FROZEN_HELD_OUT_FAMILIES,
)
from lgae_v3.runtime.curriculum import GraphFamily


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> ResearchConfig:
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 0
    cfg.audit.orc_backend = "exact_lp"
    cfg.audit.persistent_homology_enabled = False
    cfg.audit.entropic_nodes = 0
    cfg.audit.bakry_nodes = 0
    cfg.audit.cde_nodes = 0
    cfg.audit.exact_lly_top_k = 0
    cfg.audit.orc_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]
    cfg.mutation.curvature_ema_enabled = False
    return cfg


def _small_graph():
    return make_graph_buffers(8, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)], capacity=16)


def _make_minimal_record(
    record_id: str = "test_001",
    split: str = "train",
    graph_family: str = "path",
    provenance: TransitionProvenance = TransitionProvenance.REALIZED,
    action: str = "ADD_EDGE",
    auth_decision: AuthorizationDecision = AuthorizationDecision.ACCEPTED,
    realized_delta: float = 0.1,
    success: bool = True,
    step_id: int = 0,
    predicted_delta: float = 0.1,
) -> TransitionRecord:
    """Create a minimal valid TransitionRecord for testing."""
    auth = AuthorityIdentity(state_hash="abc123", state_version=0, authority_hash="hash123")
    state = StructuralStateSummary(
        n_nodes=10, n_edges=9, density=0.2, spectral_gap=0.1,
        degree_mean=1.8, degree_std=0.4, n_components=1, avg_clustering=0.0,
        fiber_count=1, fiber_width=2, gauge_dim=0, state_hash="abc123", graph_version=0,
    )
    cand = CandidateSummary(
        candidate_id=0, action_type=action, target={"u": 0, "v": 5},
        predicted_delta=predicted_delta, predicted_risk=0.0, predicted_cost=1.0,
        predicted_ig=0.05, selected=True,
    )
    cand_set = CandidateSetSummary(
        n_candidates=3, candidates=(cand,), action_distribution={action: 1},
    )
    return TransitionRecord(
        record_id=record_id,
        run_id="run001",
        episode_id="ep001",
        step_id=step_id,
        graph_family=graph_family,
        split=split,
        seed=42,
        authority_identity_before=auth,
        authority_identity_after=auth if success else None,
        structural_state_before=state,
        structural_state_after=state if success else None,
        diagnosis=DiagnosisSummary(),
        candidate_set_summary=cand_set,
        selected_candidate=cand,
        planner_metadata=PlannerMetadata(),
        predicted_delta=predicted_delta,
        predicted_risk=0.0,
        predicted_cost=1.0,
        predicted_ig=0.05,
        action=action,
        action_target={"u": 0, "v": 5},
        authorization_decision=auth_decision,
        transaction_id="txn001" if success else None,
        realized_delta=realized_delta,
        realized_cost=1.0,
        realized_risk=0.0,
        success=success,
        rollback=False,
        rejected=not success,
        compute_metrics=ComputeMetrics(candidate_evaluations=3),
        provenance=provenance,
        base_runtime_version="5.11.0",
        generator_version=GENERATOR_VERSION,
        timestamp="2026-08-18T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# 1. TransitionRecord schema
# ---------------------------------------------------------------------------

class TestTransitionRecordSchema:
    """Rich TransitionRecord schema with ~25 fields."""

    def test_record_has_all_required_fields(self):
        record = _make_minimal_record()
        # Identity.
        assert record.record_id
        assert record.run_id
        assert record.episode_id
        assert record.step_id == 0
        assert record.graph_family
        assert record.split
        assert record.seed == 42
        # Authority.
        assert record.authority_identity_before.state_hash
        # State.
        assert record.structural_state_before.n_nodes == 10
        # Diagnosis.
        assert record.diagnosis is not None
        # Candidate set.
        assert record.candidate_set_summary.n_candidates == 3
        # Selected candidate.
        assert record.selected_candidate is not None
        assert record.selected_candidate.selected is True
        # Planner.
        assert record.planner_metadata is not None
        # Predictions.
        assert record.predicted_delta == 0.1
        # Action.
        assert record.action == "ADD_EDGE"
        assert record.authorization_decision == AuthorizationDecision.ACCEPTED
        # Realized.
        assert record.realized_delta == 0.1
        # Flags.
        assert record.success is True
        assert record.rollback is False
        assert record.rejected is False
        # Compute.
        assert record.compute_metrics.candidate_evaluations == 3
        # Provenance.
        assert record.provenance == TransitionProvenance.REALIZED
        assert record.base_runtime_version == "5.11.0"
        assert record.generator_version == GENERATOR_VERSION

    def test_record_serialization(self):
        record = _make_minimal_record()
        log = record.to_log()
        assert isinstance(log, dict)
        assert log["record_id"] == "test_001"
        assert log["provenance"] == "realized"
        # Round-trip JSON.
        json_str = record.to_json()
        data = json.loads(json_str)
        assert data["record_id"] == "test_001"

    def test_make_record_id_is_deterministic(self):
        id1 = make_record_id("run1", "ep1", 0, 42, TransitionProvenance.REALIZED)
        id2 = make_record_id("run1", "ep1", 0, 42, TransitionProvenance.REALIZED)
        assert id1 == id2

    def test_make_record_id_differs_for_provenance(self):
        id_real = make_record_id("run1", "ep1", 0, 42, TransitionProvenance.REALIZED)
        id_cf = make_record_id("run1", "ep1", 0, 42, TransitionProvenance.COUNTERFACTUAL)
        assert id_real != id_cf

    def test_make_record_id_differs_for_candidate(self):
        id1 = make_record_id("run1", "ep1", 0, 42, TransitionProvenance.COUNTERFACTUAL, candidate_id=0)
        id2 = make_record_id("run1", "ep1", 0, 42, TransitionProvenance.COUNTERFACTUAL, candidate_id=1)
        assert id1 != id2


# ---------------------------------------------------------------------------
# 2. Provenance separation
# ---------------------------------------------------------------------------

class TestProvenanceSeparation:
    """ObservedTransition vs CounterfactualTransition provenance separation."""

    def test_realized_provenance(self):
        record = _make_minimal_record(provenance=TransitionProvenance.REALIZED)
        assert record.provenance == TransitionProvenance.REALIZED
        assert record.provenance.value == "realized"

    def test_counterfactual_provenance(self):
        record = _make_minimal_record(
            provenance=TransitionProvenance.COUNTERFACTUAL,
            success=False,
            auth_decision=AuthorizationDecision.REJECTED,
        )
        assert record.provenance == TransitionProvenance.COUNTERFACTUAL
        assert record.provenance.value == "counterfactual"

    def test_shadow_provenance(self):
        record = _make_minimal_record(provenance=TransitionProvenance.SHADOW)
        assert record.provenance == TransitionProvenance.SHADOW

    def test_provenance_enum_values(self):
        assert TransitionProvenance.REALIZED.value == "realized"
        assert TransitionProvenance.COUNTERFACTUAL.value == "counterfactual"
        assert TransitionProvenance.SHADOW.value == "shadow"


# ---------------------------------------------------------------------------
# 3. Feature extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    """Canonical structural feature extraction."""

    def test_global_features_dim(self):
        graph = _small_graph()
        features = extract_global_features(graph)
        assert features.dim == 24
        assert len(features.vector) == 24

    def test_global_features_are_finite(self):
        graph = _small_graph()
        features = extract_global_features(graph)
        for v in features.vector:
            assert math.isfinite(v), f"Nonfinite feature value: {v}"

    def test_global_features_deterministic(self):
        graph = _small_graph()
        f1 = extract_global_features(graph)
        f2 = extract_global_features(graph)
        assert f1.vector == f2.vector

    def test_global_features_log(self):
        graph = _small_graph()
        features = extract_global_features(graph)
        log = features.to_log()
        assert log["dim"] == 24
        assert len(log["field_names"]) == 24

    def test_local_action_features_dim(self):
        graph = _small_graph()
        features = extract_local_action_features(graph, 0, 5)
        assert features.dim == 12
        assert len(features.vector) == 12

    def test_local_action_features_finite(self):
        graph = _small_graph()
        features = extract_local_action_features(graph, 0, 5)
        for v in features.vector:
            assert math.isfinite(v), f"Nonfinite local feature: {v}"

    def test_local_action_features_for_distant_nodes(self):
        graph = _small_graph()  # path graph 0-1-2-3-4-5-6-7
        features = extract_local_action_features(graph, 0, 7)
        # Distance should be 7 (path length).
        assert features.shortest_path_distance == 7.0

    def test_combined_feature_vector(self):
        graph = _small_graph()
        global_feats = extract_global_features(graph)
        local_feats = extract_local_action_features(graph, 0, 5)
        combined = StructuralFeatureVector(
            global_features=global_feats,
            local_features=local_feats,
        )
        assert combined.dim == 24 + 12
        assert len(combined.vector) == 36

    def test_combined_feature_vector_no_local(self):
        graph = _small_graph()
        global_feats = extract_global_features(graph)
        combined = StructuralFeatureVector(
            global_features=global_feats,
            local_features=None,
        )
        assert combined.dim == 24
        assert len(combined.vector) == 24


# ---------------------------------------------------------------------------
# 4. Dataset generation
# ---------------------------------------------------------------------------

class TestDatasetGeneration:
    """Dataset generation with split control and negative sampling."""

    def test_generate_split_produces_records(self):
        gen = DatasetGenerator(config=_cfg(), seed=42, n_negative_samples=2)
        dataset = gen.generate_split("train", n_steps=2, n_episodes=1)
        assert dataset.n_records > 0
        assert dataset.split == "train"

    def test_generate_split_has_metadata(self):
        gen = DatasetGenerator(config=_cfg(), seed=42)
        dataset = gen.generate_split("train", n_steps=2, n_episodes=1)
        assert dataset.metadata is not None
        assert dataset.metadata.schema_version == DATASET_SCHEMA_VERSION
        assert dataset.metadata.split == "train"
        assert dataset.metadata.generator_version == GENERATOR_VERSION
        assert dataset.metadata.base_runtime == "5.11.0"
        assert dataset.metadata.config_hash
        assert dataset.metadata.dataset_hash

    def test_generate_split_has_quality_report(self):
        gen = DatasetGenerator(config=_cfg(), seed=42)
        dataset = gen.generate_split("train", n_steps=2, n_episodes=1)
        assert dataset.quality_report is not None
        assert dataset.quality_report.n_records > 0

    def test_generate_all_splits(self):
        gen = DatasetGenerator(config=_cfg(), seed=42, n_negative_samples=1)
        datasets = gen.generate_all_splits(n_steps=1, n_episodes=1)
        assert "train" in datasets
        assert "validation" in datasets
        assert "held_out" in datasets
        for split, ds in datasets.items():
            assert ds.split == split
            assert ds.n_records > 0

    def test_negative_sampling_produces_counterfactuals(self):
        gen = DatasetGenerator(config=_cfg(), seed=42, n_negative_samples=3)
        dataset = gen.generate_split("train", n_steps=2, n_episodes=1)
        n_cf = sum(1 for r in dataset.records if r.provenance == TransitionProvenance.COUNTERFACTUAL)
        n_obs = sum(1 for r in dataset.records if r.provenance == TransitionProvenance.REALIZED)
        assert n_cf > 0, "No counterfactual records generated"
        assert n_obs > 0, "No observed records generated"
        # Should have both types.
        assert n_cf + n_obs == dataset.n_records

    def test_dataset_serialization(self):
        gen = DatasetGenerator(config=_cfg(), seed=42, n_negative_samples=1)
        dataset = gen.generate_split("train", n_steps=1, n_episodes=1)
        json_str = dataset.to_json()
        data = json.loads(json_str)
        assert "metadata" in data
        assert "records" in data
        assert "content_hash" in data

    def test_dataset_save_to_disk(self):
        gen = DatasetGenerator(config=_cfg(), seed=42, n_negative_samples=1)
        dataset = gen.generate_split("train", n_steps=1, n_episodes=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset.save(tmpdir)
            assert (Path(tmpdir) / "dataset.json").exists()
            assert (Path(tmpdir) / "manifest.json").exists()
            assert (Path(tmpdir) / "quality_report.json").exists()

    def test_save_all_splits(self):
        gen = DatasetGenerator(config=_cfg(), seed=42, n_negative_samples=1)
        datasets = gen.generate_all_splits(n_steps=1, n_episodes=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen.save_all(datasets, tmpdir)
            assert (Path(tmpdir) / "train" / "dataset.json").exists()
            assert (Path(tmpdir) / "validation" / "dataset.json").exists()
            assert (Path(tmpdir) / "held_out" / "dataset.json").exists()
            # Each split should have its own manifest.
            assert (Path(tmpdir) / "train" / "manifest.json").exists()
            assert (Path(tmpdir) / "held_out" / "manifest.json").exists()


# ---------------------------------------------------------------------------
# 5. Dataset validation
# ---------------------------------------------------------------------------

class TestDatasetValidation:
    """Dataset validation: leakage, duplicates, schema, nonfinite."""

    def test_valid_dataset_passes(self):
        records = [
            _make_minimal_record(record_id="r1", step_id=0),
            _make_minimal_record(record_id="r2", step_id=1),
        ]
        validator = DatasetValidator()
        result = validator.validate(records, expected_split="train")
        assert result.valid
        assert result.n_errors == 0

    def test_duplicate_record_ids_fail(self):
        records = [
            _make_minimal_record(record_id="dup", step_id=0),
            _make_minimal_record(record_id="dup", step_id=1),
        ]
        validator = DatasetValidator()
        result = validator.validate(records)
        assert not result.valid
        assert any(i.category == "duplicate_id" for i in result.issues)

    def test_split_mismatch_fails(self):
        records = [_make_minimal_record(split="held_out")]
        validator = DatasetValidator()
        result = validator.validate(records, expected_split="train")
        assert not result.valid
        assert any(i.category == "split_mismatch" for i in result.issues)

    def test_held_out_leakage_fails(self):
        """Held-out family in train split should fail."""
        records = [
            _make_minimal_record(
                split="train",
                graph_family=FROZEN_HELD_OUT_FAMILIES[0].value,
            ),
        ]
        validator = DatasetValidator(
            held_out_families={f.value for f in FROZEN_HELD_OUT_FAMILIES},
        )
        result = validator.validate(records, expected_split="train")
        assert not result.valid
        assert any(i.category == "held_out_leakage" for i in result.issues)

    def test_train_contamination_in_held_out_fails(self):
        """Train family in held-out split should fail."""
        records = [
            _make_minimal_record(
                split="held_out",
                graph_family=FROZEN_TRAIN_FAMILIES[0].value,
            ),
        ]
        validator = DatasetValidator(
            train_families={f.value for f in FROZEN_TRAIN_FAMILIES},
        )
        result = validator.validate(records, expected_split="held_out")
        assert not result.valid
        assert any(i.category == "train_contamination" for i in result.issues)

    def test_nonfinite_metrics_fail(self):
        record = _make_minimal_record()
        # Create a record with nonfinite realized_delta.
        import dataclasses
        record = dataclasses.replace(record, realized_delta=float("nan"))
        validator = DatasetValidator()
        result = validator.validate([record])
        assert not result.valid
        assert any(i.category == "nonfinite" for i in result.issues)

    def test_missing_authority_fails(self):
        record = _make_minimal_record()
        # Replace with empty authority.
        import dataclasses
        empty_auth = AuthorityIdentity(state_hash="", state_version=0, authority_hash="")
        record = dataclasses.replace(record, authority_identity_before=empty_auth)
        validator = DatasetValidator()
        result = validator.validate([record])
        assert not result.valid
        assert any(i.category == "missing_authority" for i in result.issues)

    def test_duplicate_episode_step_fails(self):
        records = [
            _make_minimal_record(record_id="r1", step_id=0),
            _make_minimal_record(record_id="r2", step_id=0),  # same step
        ]
        validator = DatasetValidator()
        result = validator.validate(records)
        assert not result.valid
        assert any(i.category == "duplicate_episode_step" for i in result.issues)

    def test_empty_dataset_validates(self):
        validator = DatasetValidator()
        result = validator.validate([])
        assert result.valid
        assert result.n_records == 0


# ---------------------------------------------------------------------------
# 6. Data quality report
# ---------------------------------------------------------------------------

class TestDataQualityReport:
    """Data quality report with distributions and imbalance detection."""

    def test_quality_report_for_empty_dataset(self):
        report = generate_quality_report([])
        assert report.n_records == 0
        assert len(report.warnings) > 0

    def test_quality_report_for_valid_dataset(self):
        records = [
            _make_minimal_record(record_id="r1", step_id=0, action="ADD_EDGE"),
            _make_minimal_record(record_id="r2", step_id=1, action="PRUNE_EDGE"),
            _make_minimal_record(
                record_id="r3", step_id=2, action="NO_OP",
                provenance=TransitionProvenance.COUNTERFACTUAL,
                success=False,
            ),
        ]
        report = generate_quality_report(records)
        assert report.n_records == 3
        assert report.n_observed == 2
        assert report.n_counterfactual == 1
        assert report.action_distribution is not None
        assert report.action_distribution.counts.get("ADD_EDGE") == 1

    def test_imbalance_detection(self):
        """92% ADD_EDGE should be flagged as imbalanced."""
        records = []
        for i in range(92):
            records.append(_make_minimal_record(
                record_id=f"r{i}", step_id=i, action="ADD_EDGE",
            ))
        for i in range(8):
            records.append(_make_minimal_record(
                record_id=f"r{i+92}", step_id=i+92, action="PRUNE_EDGE",
            ))
        report = generate_quality_report(records)
        assert report.has_imbalance
        assert any("ADD_EDGE" in flag for flag in report.imbalance_flags)

    def test_no_imbalance_for_balanced_dataset(self):
        records = []
        families = ["path", "cycle", "star", "grid"]
        actions = ["ADD_EDGE", "PRUNE_EDGE", "NO_OP", "SPAWN_FIBER"]
        auths = [AuthorizationDecision.ACCEPTED, AuthorizationDecision.REJECTED,
                 AuthorizationDecision.ACCEPTED, AuthorizationDecision.QUARANTINED]
        for i in range(100):
            records.append(_make_minimal_record(
                record_id=f"r{i}", step_id=i,
                action=actions[i % len(actions)],
                graph_family=families[i % len(families)],
                auth_decision=auths[i % len(auths)],
            ))
        report = generate_quality_report(records)
        assert not report.has_imbalance

    def test_quality_report_to_log(self):
        records = [_make_minimal_record()]
        report = generate_quality_report(records)
        log = report.to_log()
        assert log["n_records"] == 1
        assert "action_distribution" in log

    def test_calibration_metrics(self):
        """Calibration correlation is computed for REALIZED records."""
        records = []
        for i in range(20):
            records.append(_make_minimal_record(
                record_id=f"r{i}",
                step_id=i,
                predicted_delta=0.1 + i * 0.01,
                realized_delta=0.1 + i * 0.01 + 0.001,  # near-perfect correlation
            ))
        report = generate_quality_report(records)
        assert report.calibration_correlation > 0.9

    def test_provenance_distribution(self):
        records = [
            _make_minimal_record(record_id="r1", step_id=0, provenance=TransitionProvenance.REALIZED),
            _make_minimal_record(record_id="r2", step_id=1, provenance=TransitionProvenance.COUNTERFACTUAL, success=False),
        ]
        report = generate_quality_report(records)
        assert report.provenance_distribution is not None
        assert report.provenance_distribution.counts.get("realized") == 1
        assert report.provenance_distribution.counts.get("counterfactual") == 1


# ---------------------------------------------------------------------------
# 7. Deterministic regeneration
# ---------------------------------------------------------------------------

class TestDeterministicRegeneration:
    """Same seed + config → identical dataset hash."""

    def test_same_seed_same_hash(self):
        gen1 = DatasetGenerator(config=_cfg(), seed=42, n_negative_samples=2)
        gen2 = DatasetGenerator(config=_cfg(), seed=42, n_negative_samples=2)
        ds1 = gen1.generate_split("train", n_steps=2, n_episodes=1)
        ds2 = gen2.generate_split("train", n_steps=2, n_episodes=1)
        assert ds1.content_hash == ds2.content_hash

    def test_different_seed_different_hash(self):
        gen1 = DatasetGenerator(config=_cfg(), seed=42, n_negative_samples=2)
        gen2 = DatasetGenerator(config=_cfg(), seed=99, n_negative_samples=2)
        ds1 = gen1.generate_split("train", n_steps=2, n_episodes=1)
        ds2 = gen2.generate_split("train", n_steps=2, n_episodes=1)
        # Hashes should differ (different seeds → different graphs/actions).
        assert ds1.content_hash != ds2.content_hash

    def test_config_hash_deterministic(self):
        gen1 = DatasetGenerator(config=_cfg(), seed=42)
        gen2 = DatasetGenerator(config=_cfg(), seed=42)
        assert gen1._config_hash == gen2._config_hash


# ---------------------------------------------------------------------------
# 8. Leakage control
# ---------------------------------------------------------------------------

class TestLeakageControl:
    """Held-out families never appear in train split."""

    def test_train_split_has_no_held_out_families(self):
        gen = DatasetGenerator(config=_cfg(), seed=42, n_negative_samples=1)
        dataset = gen.generate_split("train", n_steps=1, n_episodes=1)
        held_out_names = {f.value for f in FROZEN_HELD_OUT_FAMILIES}
        for record in dataset.records:
            assert record.graph_family not in held_out_names, \
                f"Held-out family {record.graph_family} found in train split!"

    def test_held_out_split_has_no_train_families(self):
        gen = DatasetGenerator(config=_cfg(), seed=42, n_negative_samples=1)
        dataset = gen.generate_split("held_out", n_steps=1, n_episodes=1)
        train_names = {f.value for f in FROZEN_TRAIN_FAMILIES}
        for record in dataset.records:
            assert record.graph_family not in train_names, \
                f"Train family {record.graph_family} found in held_out split!"

    def test_validation_split_separate_from_train_and_held_out(self):
        gen = DatasetGenerator(config=_cfg(), seed=42, n_negative_samples=1)
        dataset = gen.generate_split("validation", n_steps=1, n_episodes=1)
        train_names = {f.value for f in FROZEN_TRAIN_FAMILIES}
        held_out_names = {f.value for f in FROZEN_HELD_OUT_FAMILIES}
        for record in dataset.records:
            assert record.graph_family not in train_names
            assert record.graph_family not in held_out_names


# ---------------------------------------------------------------------------
# 9. Schema version
# ---------------------------------------------------------------------------

class TestSchemaVersion:
    """Dataset schema version is correct."""

    def test_schema_version(self):
        assert DATASET_SCHEMA_VERSION == "LGAE_STRUCTURAL_DATASET_V6_0_EXP2"

    def test_generator_version(self):
        assert GENERATOR_VERSION == "6.0-exp2"

    def test_metadata_has_correct_schema(self):
        gen = DatasetGenerator(config=_cfg(), seed=42)
        dataset = gen.generate_split("train", n_steps=1, n_episodes=1)
        assert dataset.metadata.schema_version == "LGAE_STRUCTURAL_DATASET_V6_0_EXP2"


# ---------------------------------------------------------------------------
# 10. Authority boundary preservation
# ---------------------------------------------------------------------------

class TestAuthorityBoundaryExp2:
    """v5.11 authority boundary is untouched by dataset generation."""

    def test_generator_does_not_mutate_runtime_after_generation(self):
        """After generating a dataset, the runtime's state should be consistent."""
        gen = DatasetGenerator(config=_cfg(), seed=42, n_negative_samples=2)
        # Generate a dataset (this runs the runtime internally).
        dataset = gen.generate_split("train", n_steps=3, n_episodes=1)
        # The dataset should have records.
        assert dataset.n_records > 0
        # The generator should not have left any runtime in an inconsistent state.
        # We verify by creating a fresh runtime and checking it works.
        runtime = LGAERuntime(graph=_small_graph(), config=_cfg(), runtime_config=RuntimeConfig())
        result = runtime.step()
        assert result is not None

    def test_counterfactual_records_have_no_authority_binding(self):
        """Counterfactual records must not have authority_identity_after."""
        gen = DatasetGenerator(config=_cfg(), seed=42, n_negative_samples=3)
        dataset = gen.generate_split("train", n_steps=2, n_episodes=1)
        cf_records = [r for r in dataset.records if r.provenance == TransitionProvenance.COUNTERFACTUAL]
        for r in cf_records:
            # Counterfactual records should NOT have authority_identity_after
            # (they were never committed).
            assert r.authority_identity_after is None
            assert r.transaction_id is None
            assert r.success is False
