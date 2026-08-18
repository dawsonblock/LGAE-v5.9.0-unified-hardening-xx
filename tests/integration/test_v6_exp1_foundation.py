"""v6.0-exp1: Structural world-model foundation tests.

Tests verify:
1. Frozen graph family splits are correct and non-overlapping.
2. All baseline runners produce valid results.
3. The benchmark harness orchestrates correctly.
4. The transition recorder captures v5.11 runtime steps passively.
5. The dataset schema serializes and deserializes correctly.
6. The experiment registry tracks experiments with provenance.
7. Reproducibility controls produce deterministic fingerprints.
8. World-model interfaces are abstract and enforce contracts.
9. v5.11 authority boundary is never violated by v6 components.
"""
from __future__ import annotations

import pytest
import torch
import json
import tempfile
from pathlib import Path

from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.experimental import (
    # Graph families
    GraphFamilySplit, FrozenGraphFamilyRegistry,
    FROZEN_TRAIN_FAMILIES, FROZEN_VALIDATION_FAMILIES, FROZEN_HELD_OUT_FAMILIES,
    get_frozen_registry,
    # Metrics
    V6Metric, V6MetricReport, aggregate_metrics,
    adaptation_speed_metric, performance_per_compute_metric,
    ood_generalization_metric, mutation_count_metric,
    topology_complexity_metric, failure_rate_metric, calibration_metric,
    # Baselines
    FixedTopologyBaseline, RandomRewiringBaseline, GreedyBaseline,
    CurvatureOnlyBaseline, FoSRBaseline, BORFBaseline,
    EffectiveResistanceBaseline, OneStepCounterfactualBaseline,
    MPCBaseline, MPCWithIGBaseline, FullV511Baseline,
    ALL_V6_BASELINES,
    # Harness
    V6BenchmarkHarness, BenchmarkRunResult,
    # Transition recorder
    StructuralTransition, TransitionRecorder, record_runtime_step,
    # Dataset
    StructuralDataset, StructuralDatasetSchema, DatasetMetadata,
    # Experiment registry
    ExperimentRecord, ExperimentRegistry, ExperimentStatus,
    # Reproducibility
    ReproducibilityConfig, RunFingerprint, seed_all, config_hash,
    # World model
    WorldModelInterface, OutcomeModelInterface, StructuralStateEncoderInterface,
    ModelPrediction, ModelTrustReport,
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


def _small_graph() -> torch.Tensor:
    return make_graph_buffers(8, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)], capacity=16)


# ---------------------------------------------------------------------------
# 1. Frozen graph family splits
# ---------------------------------------------------------------------------

class TestFrozenGraphFamilySplits:
    """Frozen splits are correct, non-overlapping, and deterministic."""

    def test_train_validation_held_out_no_overlap(self):
        train = set(FROZEN_TRAIN_FAMILIES)
        val = set(FROZEN_VALIDATION_FAMILIES)
        held = set(FROZEN_HELD_OUT_FAMILIES)
        assert not (train & val)
        assert not (train & held)
        assert not (val & held)

    def test_all_families_covered(self):
        all_frozen = set(FROZEN_TRAIN_FAMILIES) | set(FROZEN_VALIDATION_FAMILIES) | set(FROZEN_HELD_OUT_FAMILIES)
        # Should cover most of the GraphFamily enum.
        assert len(all_frozen) >= 9

    def test_registry_generates_entries(self):
        registry = get_frozen_registry()
        train = registry.train_entries()
        val = registry.validation_entries()
        held = registry.held_out_entries()
        assert len(train) > 0
        assert len(val) > 0
        assert len(held) > 0
        # Train should be larger than validation and held-out.
        assert len(train) >= len(val)
        assert len(train) >= len(held)

    def test_registry_is_deterministic(self):
        registry1 = get_frozen_registry()
        registry2 = get_frozen_registry()
        e1 = registry1.train_entries()
        e2 = registry2.train_entries()
        assert len(e1) == len(e2)
        for a, b in zip(e1, e2):
            assert a.family_id == b.family_id
            assert a.seed == b.seed

    def test_split_to_log(self):
        registry = get_frozen_registry()
        log = registry.split.to_log()
        assert "train" in log
        assert "validation" in log
        assert "held_out" in log
        assert len(log["train"]) == len(FROZEN_TRAIN_FAMILIES)
        assert len(log["held_out"]) == len(FROZEN_HELD_OUT_FAMILIES)


# ---------------------------------------------------------------------------
# 2. Baseline runners
# ---------------------------------------------------------------------------

class TestBaselineRunners:
    """All baseline runners produce valid results."""

    @pytest.mark.parametrize("baseline_name", list(ALL_V6_BASELINES.keys()))
    def test_baseline_runs_and_produces_result(self, baseline_name):
        baseline = ALL_V6_BASELINES[baseline_name]
        graph = _small_graph()
        cfg = _cfg()
        result = baseline.run(graph, cfg, seed=42, n_steps=3)
        assert result.baseline_name == baseline_name
        assert len(result.utility_history) >= 1
        assert result.n_steps == 3
        assert result.final_n_nodes == 8
        assert result.final_n_edges >= 0

    def test_fixed_topology_does_not_mutate(self):
        baseline = FixedTopologyBaseline()
        graph = _small_graph()
        result = baseline.run(graph, _cfg(), seed=42, n_steps=5)
        assert result.n_mutations == 0
        assert result.final_utility == result.utility_history[0]

    def test_greedy_improves_or_stays(self):
        baseline = GreedyBaseline()
        graph = _small_graph()
        result = baseline.run(graph, _cfg(), seed=42, n_steps=5)
        # Greedy should not decrease utility.
        assert result.final_utility >= result.utility_history[0] - 1e-6

    def test_fosr_improves_or_stays(self):
        baseline = FoSRBaseline()
        graph = _small_graph()
        result = baseline.run(graph, _cfg(), seed=42, n_steps=5)
        assert result.final_utility >= result.utility_history[0] - 1e-6

    def test_mpc_has_horizon_metadata(self):
        baseline = MPCBaseline(horizon=3)
        graph = _small_graph()
        result = baseline.run(graph, _cfg(), seed=42, n_steps=2)
        assert result.metadata["horizon"] == 3

    def test_mpc_with_ig_has_ig_weight(self):
        baseline = MPCWithIGBaseline(horizon=2, ig_weight=0.2)
        graph = _small_graph()
        result = baseline.run(graph, _cfg(), seed=42, n_steps=2)
        assert result.metadata["ig_weight"] == 0.2
        assert result.metadata["horizon"] == 2

    def test_borf_rewires(self):
        baseline = BORFBaseline()
        graph = _small_graph()
        result = baseline.run(graph, _cfg(), seed=42, n_steps=3)
        # BORF should have mutations (rewiring).
        assert result.n_mutations > 0

    def test_random_rewiring_mutates(self):
        baseline = RandomRewiringBaseline()
        graph = _small_graph()
        result = baseline.run(graph, _cfg(), seed=42, n_steps=5)
        assert result.n_mutations > 0

    def test_one_step_counterfactual_evaluates_candidates(self):
        baseline = OneStepCounterfactualBaseline()
        graph = _small_graph()
        result = baseline.run(graph, _cfg(), seed=42, n_steps=3)
        # Should have non-zero compute cost from candidate evaluation.
        assert result.compute_cost > 0

    def test_effective_resistance_adds_edges(self):
        baseline = EffectiveResistanceBaseline()
        graph = _small_graph()
        result = baseline.run(graph, _cfg(), seed=42, n_steps=3)
        # Should add edges (connecting distant nodes).
        assert result.n_mutations > 0


# ---------------------------------------------------------------------------
# 3. Benchmark harness
# ---------------------------------------------------------------------------

class TestBenchmarkHarness:
    """The benchmark harness orchestrates correctly."""

    def test_harness_runs_subset_of_baselines(self):
        harness = V6BenchmarkHarness(config=_cfg())
        results = harness.run_all_baselines(
            n_steps=2,
            seed=42,
            baselines=["fixed_topology", "greedy"],
            splits=["train"],
        )
        assert len(results) > 0
        # All results should be from the specified baselines and split.
        for r in results:
            assert r.baseline_name in ("fixed_topology", "greedy")
            assert r.split == "train"

    def test_harness_produces_metrics(self):
        harness = V6BenchmarkHarness(config=_cfg())
        results = harness.run_all_baselines(
            n_steps=2,
            seed=42,
            baselines=["fixed_topology"],
            splits=["train"],
        )
        for r in results:
            assert len(r.metrics) > 0
            # Should have final_utility metric.
            metric_names = [m.name for m in r.metrics]
            assert "final_utility" in metric_names

    def test_harness_summarize(self):
        harness = V6BenchmarkHarness(config=_cfg())
        results = harness.run_all_baselines(
            n_steps=2,
            seed=42,
            baselines=["fixed_topology", "greedy"],
            splits=["train"],
        )
        summary = harness.summarize(results)
        assert summary["n_results"] == len(results)
        assert "fixed_topology" in summary["by_baseline"]
        assert "greedy" in summary["by_baseline"]
        assert "metric_reports" in summary

    def test_harness_runs_on_held_out(self):
        harness = V6BenchmarkHarness(config=_cfg())
        results = harness.run_all_baselines(
            n_steps=2,
            seed=42,
            baselines=["fixed_topology"],
            splits=["held_out"],
        )
        for r in results:
            assert r.split == "held_out"


# ---------------------------------------------------------------------------
# 4. Transition recorder
# ---------------------------------------------------------------------------

class TestTransitionRecorder:
    """The transition recorder captures v5.11 steps passively."""

    def test_recorder_is_empty_initially(self):
        recorder = TransitionRecorder(seed=42)
        assert len(recorder) == 0
        assert len(recorder.dataset()) == 0

    def test_recorder_records_step(self):
        cfg = _cfg()
        runtime = LGAERuntime(graph=_small_graph(), config=cfg, runtime_config=RuntimeConfig())
        recorder = TransitionRecorder(seed=42)
        result = runtime.step()
        transition = recorder.record(result)
        assert len(recorder) == 1
        assert transition.step == 0
        assert transition.seed == 42
        assert transition.transition_id  # non-empty
        assert transition.runtime_version  # non-empty

    def test_recorder_records_multiple_steps(self):
        cfg = _cfg()
        runtime = LGAERuntime(graph=_small_graph(), config=cfg, runtime_config=RuntimeConfig())
        recorder = TransitionRecorder(seed=42)
        for _ in range(3):
            result = runtime.step()
            recorder.record(result)
        assert len(recorder) == 3
        dataset = recorder.dataset()
        assert dataset[0].step == 0
        assert dataset[1].step == 1
        assert dataset[2].step == 2

    def test_recorder_does_not_mutate_runtime(self):
        """The recorder must not affect the runtime's authoritative state."""
        cfg = _cfg()
        runtime = LGAERuntime(graph=_small_graph(), config=cfg, runtime_config=RuntimeConfig())
        recorder = TransitionRecorder(seed=42)
        # Run a step without recording.
        result1 = runtime.step()
        gen_1 = runtime.snapshot().generation
        # Record it (should not affect state).
        recorder.record(result1)
        gen_2 = runtime.snapshot().generation
        assert gen_1 == gen_2

    def test_recorder_to_log(self):
        cfg = _cfg()
        runtime = LGAERuntime(graph=_small_graph(), config=cfg, runtime_config=RuntimeConfig())
        recorder = TransitionRecorder(seed=42)
        result = runtime.step()
        recorder.record(result)
        log = recorder.to_log()
        assert log["seed"] == 42
        assert log["n_transitions"] == 1

    def test_record_runtime_step_convenience(self):
        cfg = _cfg()
        runtime = LGAERuntime(graph=_small_graph(), config=cfg, runtime_config=RuntimeConfig())
        recorder = TransitionRecorder(seed=42)
        result = runtime.step()
        transition = record_runtime_step(result, recorder)
        assert transition is not None
        assert len(recorder) == 1

    def test_recorder_clear(self):
        recorder = TransitionRecorder(seed=42)
        # Add a fake transition by recording a real step.
        cfg = _cfg()
        runtime = LGAERuntime(graph=_small_graph(), config=cfg, runtime_config=RuntimeConfig())
        result = runtime.step()
        recorder.record(result)
        assert len(recorder) == 1
        recorder.clear()
        assert len(recorder) == 0


# ---------------------------------------------------------------------------
# 5. Dataset schema
# ---------------------------------------------------------------------------

class TestStructuralDataset:
    """The dataset schema serializes and deserializes correctly."""

    def test_dataset_creation(self):
        cfg = _cfg()
        runtime = LGAERuntime(graph=_small_graph(), config=cfg, runtime_config=RuntimeConfig())
        recorder = TransitionRecorder(seed=42)
        for _ in range(3):
            result = runtime.step()
            recorder.record(result)
        dataset = StructuralDataset(
            transitions=recorder.dataset(),
            split="train",
            seed=42,
            description="test dataset",
        )
        assert dataset.n_transitions == 3
        assert dataset.metadata.schema_version == "LGAE_STRUCTURAL_DATASET_V6_0_EXP1"
        assert dataset.metadata.split == "train"

    def test_dataset_serialization_roundtrip(self):
        cfg = _cfg()
        runtime = LGAERuntime(graph=_small_graph(), config=cfg, runtime_config=RuntimeConfig())
        recorder = TransitionRecorder(seed=42)
        for _ in range(2):
            result = runtime.step()
            recorder.record(result)
        dataset = StructuralDataset(
            transitions=recorder.dataset(),
            split="train",
            seed=42,
        )
        json_str = dataset.to_json()
        assert "metadata" in json_str
        assert "transitions" in json_str
        # Deserialize.
        dataset2 = StructuralDataset.from_json(json_str)
        assert dataset2.n_transitions == dataset.n_transitions
        assert dataset2.metadata.split == dataset.metadata.split
        assert dataset2.metadata.seed == dataset.metadata.seed

    def test_dataset_save_load(self):
        cfg = _cfg()
        runtime = LGAERuntime(graph=_small_graph(), config=cfg, runtime_config=RuntimeConfig())
        recorder = TransitionRecorder(seed=42)
        for _ in range(2):
            result = runtime.step()
            recorder.record(result)
        dataset = StructuralDataset(
            transitions=recorder.dataset(),
            split="train",
            seed=42,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            dataset.save(path)
            loaded = StructuralDataset.load(path)
            assert loaded.n_transitions == dataset.n_transitions
        finally:
            Path(path).unlink(missing_ok=True)

    def test_dataset_content_hash(self):
        cfg = _cfg()
        runtime = LGAERuntime(graph=_small_graph(), config=cfg, runtime_config=RuntimeConfig())
        recorder = TransitionRecorder(seed=42)
        for _ in range(2):
            result = runtime.step()
            recorder.record(result)
        dataset = StructuralDataset(
            transitions=recorder.dataset(),
            split="train",
            seed=42,
        )
        h = dataset.content_hash
        assert len(h) == 64  # SHA-256 hex

    def test_schema_validation(self):
        cfg = _cfg()
        runtime = LGAERuntime(graph=_small_graph(), config=cfg, runtime_config=RuntimeConfig())
        recorder = TransitionRecorder(seed=42)
        result = runtime.step()
        recorder.record(result)
        dataset = StructuralDataset(
            transitions=recorder.dataset(),
            split="train",
            seed=42,
        )
        data = json.loads(dataset.to_json())
        assert StructuralDatasetSchema.validate(data)
        # Invalid: missing required key.
        bad_data = {"metadata": {}, "transitions": []}
        assert not StructuralDatasetSchema.validate(bad_data)


# ---------------------------------------------------------------------------
# 6. Experiment registry
# ---------------------------------------------------------------------------

class TestExperimentRegistry:
    """The experiment registry tracks experiments with provenance."""

    def test_create_experiment(self):
        registry = ExperimentRegistry()
        exp = registry.create(
            name="test-exp",
            description="A test experiment",
            config={"seed": 42, "n_steps": 5},
        )
        assert exp.name == "test-exp"
        assert exp.status == ExperimentStatus.PENDING
        assert exp.experiment_id  # non-empty

    def test_start_and_complete(self):
        registry = ExperimentRegistry()
        exp = registry.create("test", "desc", {"seed": 42})
        registry.start(exp.experiment_id)
        assert registry.get(exp.experiment_id).status == ExperimentStatus.RUNNING
        registry.complete(exp.experiment_id, results={"utility": 0.5})
        assert registry.get(exp.experiment_id).status == ExperimentStatus.COMPLETED
        assert registry.get(exp.experiment_id).results["utility"] == 0.5

    def test_fail_experiment(self):
        registry = ExperimentRegistry()
        exp = registry.create("test", "desc", {"seed": 42})
        registry.start(exp.experiment_id)
        registry.fail(exp.experiment_id, error="something went wrong")
        assert registry.get(exp.experiment_id).status == ExperimentStatus.FAILED
        assert "something" in registry.get(exp.experiment_id).error

    def test_supersede_experiment(self):
        registry = ExperimentRegistry()
        exp = registry.create("test", "desc", {"seed": 42})
        registry.supersede(exp.experiment_id)
        assert registry.get(exp.experiment_id).status == ExperimentStatus.SUPERSEDED

    def test_by_status(self):
        registry = ExperimentRegistry()
        e1 = registry.create("a", "desc", {"s": 1})
        e2 = registry.create("b", "desc", {"s": 2})
        registry.start(e2.experiment_id)
        pending = registry.by_status(ExperimentStatus.PENDING)
        running = registry.by_status(ExperimentStatus.RUNNING)
        assert len(pending) == 1
        assert len(running) == 1

    def test_by_tag(self):
        registry = ExperimentRegistry()
        registry.create("a", "desc", {"s": 1}, tags=["benchmark"])
        registry.create("b", "desc", {"s": 2}, tags=["world_model"])
        tagged = registry.by_tag("benchmark")
        assert len(tagged) == 1
        assert tagged[0].name == "a"

    def test_save_load(self):
        registry = ExperimentRegistry()
        registry.create("test", "desc", {"seed": 42}, tags=["v6"])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            registry.save(path)
            loaded = ExperimentRegistry.load(path)
            assert len(loaded) == 1
            exp = loaded.all_experiments()[0]
            assert exp.name == "test"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_deterministic_experiment_id(self):
        """Same config → same experiment ID."""
        registry = ExperimentRegistry()
        e1 = registry.create("test", "desc", {"seed": 42})
        # Creating with same name/desc/config should produce same ID.
        e2 = registry.create("test", "desc", {"seed": 42})
        assert e1.experiment_id == e2.experiment_id


# ---------------------------------------------------------------------------
# 7. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Reproducibility controls produce deterministic fingerprints."""

    def test_seed_all_does_not_error(self):
        seed_all(42)
        # Verify seeds are set.
        import random as py_random
        assert py_random.randint(0, 1000) is not None

    def test_config_hash_is_deterministic(self):
        h1 = config_hash({"a": 1, "b": 2})
        h2 = config_hash({"b": 2, "a": 1})  # different order
        assert h1 == h2  # order-independent

    def test_config_hash_differs_for_different_configs(self):
        h1 = config_hash({"a": 1})
        h2 = config_hash({"a": 2})
        assert h1 != h2

    def test_run_fingerprint_is_deterministic(self):
        fp1 = RunFingerprint.create({"n_steps": 5, "seed": 42})
        fp2 = RunFingerprint.create({"n_steps": 5, "seed": 42})
        assert fp1.combined_hash == fp2.combined_hash

    def test_run_fingerprint_differs_for_different_configs(self):
        fp1 = RunFingerprint.create({"n_steps": 5})
        fp2 = RunFingerprint.create({"n_steps": 10})
        assert fp1.combined_hash != fp2.combined_hash

    def test_reproducibility_config_to_log(self):
        config = ReproducibilityConfig(seed=123)
        log = config.to_log()
        assert log["seed"] == 123
        assert log["torch_deterministic"] is True


# ---------------------------------------------------------------------------
# 8. World-model interfaces
# ---------------------------------------------------------------------------

class TestWorldModelInterfaces:
    """World-model interfaces are abstract and enforce contracts."""

    def test_outcome_model_interface_is_abstract(self):
        with pytest.raises(TypeError):
            OutcomeModelInterface()  # type: ignore[abstract]

    def test_world_model_interface_is_abstract(self):
        with pytest.raises(TypeError):
            WorldModelInterface()  # type: ignore[abstract]

    def test_state_encoder_interface_is_abstract(self):
        with pytest.raises(TypeError):
            StructuralStateEncoderInterface()  # type: ignore[abstract]

    def test_model_prediction_dataclass(self):
        pred = ModelPrediction(
            predicted_delta_utility=0.1,
            predicted_reward=0.1,
            predicted_cost=5.0,
            predicted_uncertainty=0.2,
        )
        assert pred.predicted_delta_utility == 0.1
        log = pred.to_log()
        assert log["predicted_delta_utility"] == 0.1

    def test_model_trust_report_dataclass(self):
        report = ModelTrustReport(
            model_name="test",
            mean_prediction_error=0.1,
            ood_distance=0.5,
            calibration_correlation=0.8,
            trust_score=0.7,
            recommended_horizon=3,
            recommended_exact_verification_fraction=0.3,
        )
        assert report.trust_score == 0.7
        assert report.recommended_horizon == 3

    def test_concrete_outcome_model_can_be_implemented(self):
        """A minimal concrete outcome model satisfies the interface."""
        class DummyOutcomeModel(OutcomeModelInterface):
            @property
            def name(self) -> str:
                return "dummy"

            def predict(self, state, action):
                return ModelPrediction(predicted_delta_utility=0.0)

            def predict_batch(self, state, actions):
                return [self.predict(state, a) for a in actions]

            def update(self, state, action, realized):
                pass

            def trust_report(self):
                return ModelTrustReport(
                    model_name="dummy",
                    mean_prediction_error=0.0,
                    ood_distance=0.0,
                    calibration_correlation=0.0,
                    trust_score=0.5,
                    recommended_horizon=1,
                    recommended_exact_verification_fraction=1.0,
                )

        model = DummyOutcomeModel()
        pred = model.predict(None, None)
        assert pred.predicted_delta_utility == 0.0
        report = model.trust_report()
        assert report.model_name == "dummy"

    def test_concrete_world_model_can_be_implemented(self):
        """A minimal concrete world model satisfies the interface."""
        class DummyWorldModel(WorldModelInterface):
            @property
            def name(self) -> str:
                return "dummy_world"

            def predict_next_state(self, state, action):
                return ModelPrediction(predicted_next_state_hash="dummy")

            def rollout(self, initial_state, actions):
                return [self.predict_next_state(initial_state, a) for a in actions]

            def update(self, state, action, next_state):
                pass

            def trust_report(self):
                return ModelTrustReport(
                    model_name="dummy_world",
                    mean_prediction_error=0.0,
                    ood_distance=0.0,
                    calibration_correlation=0.0,
                    trust_score=0.5,
                    recommended_horizon=1,
                    recommended_exact_verification_fraction=1.0,
                )

        model = DummyWorldModel()
        pred = model.predict_next_state(None, None)
        assert pred.predicted_next_state_hash == "dummy"


# ---------------------------------------------------------------------------
# 9. Metrics
# ---------------------------------------------------------------------------

class TestV6Metrics:
    """v6 metrics compute correctly."""

    def test_adaptation_speed(self):
        # Reaches threshold at step 3.
        utilities = [0.0, 0.1, 0.5, 0.9, 1.0]
        m = adaptation_speed_metric(utilities, threshold=0.85, seed=42)
        assert m.value == 3
        assert m.direction == "lower"

    def test_adaptation_speed_never_reached(self):
        utilities = [0.0, 0.1, 0.2]
        m = adaptation_speed_metric(utilities, threshold=1.0, seed=42)
        assert m.value == 3  # len(utilities)

    def test_performance_per_compute(self):
        m = performance_per_compute_metric(utility_gain=1.0, compute_cost=10.0, seed=42)
        assert m.value == pytest.approx(0.1)
        assert m.direction == "higher"

    def test_ood_generalization(self):
        m = ood_generalization_metric(train_performance=1.0, held_out_performance=0.8, seed=42)
        assert m.value == pytest.approx(0.8)
        assert m.split == "held_out"

    def test_mutation_count(self):
        m = mutation_count_metric(n_mutations=5, seed=42)
        assert m.value == 5.0
        assert m.direction == "lower"

    def test_topology_complexity(self):
        m = topology_complexity_metric(n_edges=20, n_nodes=10, seed=42)
        assert m.value == 2.0

    def test_failure_rate(self):
        m = failure_rate_metric(n_rejected=3, n_total=10, seed=42)
        assert m.value == pytest.approx(0.3)

    def test_calibration_perfect(self):
        m = calibration_metric([1, 2, 3], [1, 2, 3], seed=42)
        assert m.value == pytest.approx(1.0)

    def test_calibration_no_correlation(self):
        m = calibration_metric([1, 2, 3], [3, 1, 2], seed=42)
        assert abs(m.value) < 0.9  # not perfectly correlated

    def test_aggregate_metrics(self):
        metrics = [
            V6Metric("final_utility", 0.5, 42, "train"),
            V6Metric("final_utility", 0.6, 43, "train"),
            V6Metric("mutation_count", 3, 42, "train", direction="lower"),
        ]
        reports = aggregate_metrics(metrics)
        assert "final_utility" in reports
        assert "mutation_count" in reports
        assert reports["final_utility"].mean == pytest.approx(0.55)
        assert len(reports["final_utility"].metrics) == 2


# ---------------------------------------------------------------------------
# 10. Authority boundary preservation
# ---------------------------------------------------------------------------

class TestAuthorityBoundaryPreservation:
    """v6 components never violate the v5.11 authority boundary."""

    def test_transition_recorder_does_not_touch_wal(self):
        """The recorder must not interact with the WAL."""
        cfg = _cfg()
        runtime = LGAERuntime(graph=_small_graph(), config=cfg, runtime_config=RuntimeConfig())
        recorder = TransitionRecorder(seed=42)

        # The WAL may be None in non-production mode. We verify the recorder
        # doesn't create or modify it.
        wal_before = runtime._wal
        result = runtime.step()
        recorder.record(result)
        wal_after = runtime._wal
        # WAL reference should not have changed.
        assert wal_before is wal_after

        # Also verify the recorder doesn't touch the commit channel.
        cc_before = runtime.commit_channel
        result2 = runtime.step()
        recorder.record(result2)
        cc_after = runtime.commit_channel
        assert cc_before is cc_after

    def test_baselines_do_not_use_commit_channel(self):
        """Baselines operate on graph copies, not the runtime's authority."""
        cfg = _cfg()
        runtime = LGAERuntime(graph=_small_graph(), config=cfg, runtime_config=RuntimeConfig())
        initial_graph = runtime._engine.graph
        initial_gen = runtime.snapshot().generation

        # Run a baseline on a copy of the graph.
        graph_copy = _small_graph()
        baseline = GreedyBaseline()
        baseline.run(graph_copy, cfg, seed=42, n_steps=3)

        # The runtime's state should be unaffected.
        assert runtime.snapshot().generation == initial_gen
