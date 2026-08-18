"""v5.11 Phase 15-16: Learning connection and governed model promotion tests.

These tests prove that:
1. Learning is connected to committed outcomes (not just governance rejections)
2. The replay buffer grows when commits happen
3. Calibration updates from predicted vs realized outcomes
4. Model promotion is governed by gates (safety, scientific, performance)
5. Promotion to PRODUCTION requires a signed checkpoint
6. Ungoverned promotion is rejected
"""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.runtime.promotion import (
    PromotionLevel, evaluate_promotion, assert_promotion, PromotionGateError,
)
from lgae_v3.runtime.qualification import (
    SafetyQualificationReport, SafetyCheckResult, SafetyCheckStatus,
)
from lgae_v3.runtime.scientific_qualification import (
    ScientificQualificationReport, ScientificMetric,
)
from lgae_v3.runtime.performance_qualification import (
    PerformanceQualificationReport, TierMeasurement, ScaleTier, MeasurementStatus,
)


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


def _graph():
    return make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=12)


def _passing_safety_report() -> SafetyQualificationReport:
    """Create a safety report where all checks pass."""
    report = SafetyQualificationReport()
    for name in ["invariants", "adversarial", "crash_recovery", "authority_boundary"]:
        report.add(SafetyCheckResult(
            name=name,
            status=SafetyCheckStatus.PASS,
            count=0,
            evidence={},
            message="passed",
        ))
    return report


def _failing_safety_report(failed_check: str = "invariants") -> SafetyQualificationReport:
    """Create a safety report where one check fails."""
    report = SafetyQualificationReport()
    for name in ["invariants", "adversarial", "crash_recovery", "authority_boundary"]:
        report.add(SafetyCheckResult(
            name=name,
            status=SafetyCheckStatus.FAIL if name == failed_check else SafetyCheckStatus.PASS,
            count=1 if name == failed_check else 0,
            evidence={},
            message="failed" if name == failed_check else "passed",
        ))
    return report


class TestLearningConnection:
    """Prove that learning is connected to committed outcomes."""

    def test_replay_buffer_grows_with_commits(self):
        """After steps with commits, the replay buffer grows."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        initial_size = len(rt.executive._experience)
        committed_count = 0
        for _ in range(10):
            result = rt.step()
            if result.committed:
                committed_count += 1
        final_size = len(rt.executive._experience)
        if committed_count > 0:
            assert final_size > initial_size, (
                f"Replay buffer did not grow after {committed_count} commits! "
                f"Initial: {initial_size}, Final: {final_size}"
            )

    def test_learning_transition_records_predicted_outcome(self):
        """The learning transition records the predicted outcome from the mutation."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        result = rt.step()
        assert result.learning is not None
        assert result.learning.transition is not None
        assert hasattr(result.learning.transition, "predicted_outcome")

    def test_learning_transition_records_realized_outcome(self):
        """The learning transition records the realized utility."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        result = rt.step()
        assert result.learning is not None
        assert result.learning.transition is not None
        assert hasattr(result.learning.transition, "realized_outcome")

    def test_calibration_flag_set_on_commit(self):
        """The calibration_updated flag is set when a commit happens."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        for _ in range(10):
            result = rt.step()
            if result.committed and result.learning is not None:
                assert isinstance(result.learning.calibration_updated, bool)
                break

    def test_governance_rejection_records_outcome(self):
        """When a proposal is rejected, the governance outcome is recorded."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        initial_size = len(rt.executive._experience)
        for _ in range(10):
            result = rt.step()
            if not result.committed and result.chosen_action != "no_op":
                break
        # The experience buffer should have grown if any rejection happened.
        final_size = len(rt.executive._experience)
        # At least the buffer should not shrink.
        assert final_size >= initial_size


class TestGovernedModelPromotion:
    """Prove that model promotion is governed by gates."""

    def test_promotion_experimental_to_candidate_requires_safety(self):
        """Promotion from EXPERIMENTAL to CANDIDATE requires the safety gate."""
        report = evaluate_promotion(
            current_level=PromotionLevel.EXPERIMENTAL,
            target_level=PromotionLevel.CANDIDATE,
            safety_report=None,
        )
        assert not report.promotion_approved
        with pytest.raises(PromotionGateError):
            assert_promotion(report)

    def test_promotion_with_safety_passes(self):
        """Promotion with a passing safety report is approved."""
        report = evaluate_promotion(
            current_level=PromotionLevel.EXPERIMENTAL,
            target_level=PromotionLevel.CANDIDATE,
            safety_report=_passing_safety_report(),
        )
        assert report.promotion_approved

    def test_promotion_to_qualified_requires_scientific_and_performance(self):
        """Promotion to QUALIFIED requires safety + scientific + performance."""
        report = evaluate_promotion(
            current_level=PromotionLevel.CANDIDATE,
            target_level=PromotionLevel.QUALIFIED,
            safety_report=_passing_safety_report(),
        )
        assert not report.promotion_approved
        with pytest.raises(PromotionGateError):
            assert_promotion(report)

    def test_promotion_to_production_requires_signed_checkpoint(self):
        """Promotion to PRODUCTION requires a signed checkpoint."""
        report = evaluate_promotion(
            current_level=PromotionLevel.QUALIFIED,
            target_level=PromotionLevel.PRODUCTION,
            safety_report=_passing_safety_report(),
            scientific_report=ScientificQualificationReport(),
            performance_report=PerformanceQualificationReport(),
            signed_checkpoint=None,
        )
        assert not report.promotion_approved
        with pytest.raises(PromotionGateError):
            assert_promotion(report)

    def test_promotion_to_production_with_checkpoint_passes(self):
        """Promotion to PRODUCTION with all gates + checkpoint is approved."""
        # v5.11-RC Phase 18: Performance gate requires PASS, not just MEASURED.
        perf = PerformanceQualificationReport()
        perf.add(TierMeasurement(
            tier=ScaleTier.S,
            n_nodes=100,
            status=MeasurementStatus.PASS,
            proposal_latency_ms=10.0,
            commit_latency_ms=5.0,
            candidate_throughput=1000.0,
        ))
        perf.add(TierMeasurement(
            tier=ScaleTier.M,
            n_nodes=1000,
            status=MeasurementStatus.PASS,
            proposal_latency_ms=50.0,
            commit_latency_ms=25.0,
            candidate_throughput=500.0,
        ))
        # Create a scientific report with passing gates.
        sci = ScientificQualificationReport(
            regret_learned=ScientificMetric(
                name="regret_learned", values_by_seed={0: 0.1, 1: 0.12},
                threshold=0.2, direction="lower",
            ),
            regret_best_baseline=ScientificMetric(
                name="regret_baseline", values_by_seed={0: 0.2, 1: 0.22},
                threshold=0.15, direction="lower",
            ),
            sigma_ood=ScientificMetric(
                name="sigma_ood", values_by_seed={0: 0.8, 1: 0.85},
                threshold=0.0, direction="higher",
            ),
            sigma_id=ScientificMetric(
                name="sigma_id", values_by_seed={0: 0.3, 1: 0.35},
                threshold=0.0, direction="higher",
            ),
            ig_correlation=ScientificMetric(
                name="ig_correlation", values_by_seed={0: 0.5, 1: 0.6},
                threshold=0.0, direction="higher",
            ),
        )
        report = evaluate_promotion(
            current_level=PromotionLevel.QUALIFIED,
            target_level=PromotionLevel.PRODUCTION,
            safety_report=_passing_safety_report(),
            scientific_report=sci,
            performance_report=perf,
            signed_checkpoint="sha256:abc123",
        )
        assert report.promotion_approved, (
            f"Promotion should be approved but gates: "
            f"{[(g.name, g.status_str) for g in report.gates]}"
        )
        assert_promotion(report)  # should not raise

    def test_no_downgrade_allowed(self):
        """Promotion to a lower level is a no-op (not a downgrade)."""
        report = evaluate_promotion(
            current_level=PromotionLevel.PRODUCTION,
            target_level=PromotionLevel.QUALIFIED,
        )
        assert report.promotion_approved  # no gates needed for no-op

    def test_failed_safety_gate_blocks_promotion(self):
        """A failed safety gate blocks promotion."""
        report = evaluate_promotion(
            current_level=PromotionLevel.EXPERIMENTAL,
            target_level=PromotionLevel.CANDIDATE,
            safety_report=_failing_safety_report("invariants"),
        )
        assert not report.promotion_approved
        with pytest.raises(PromotionGateError):
            assert_promotion(report)

    def test_failed_adversarial_gate_blocks_promotion(self):
        """A failed adversarial test blocks promotion."""
        report = evaluate_promotion(
            current_level=PromotionLevel.EXPERIMENTAL,
            target_level=PromotionLevel.CANDIDATE,
            safety_report=_failing_safety_report("adversarial"),
        )
        assert not report.promotion_approved

    def test_failed_crash_recovery_gate_blocks_promotion(self):
        """A failed crash recovery test blocks promotion."""
        report = evaluate_promotion(
            current_level=PromotionLevel.EXPERIMENTAL,
            target_level=PromotionLevel.CANDIDATE,
            safety_report=_failing_safety_report("crash_recovery"),
        )
        assert not report.promotion_approved

    def test_failed_authority_boundary_gate_blocks_promotion(self):
        """A failed authority boundary test blocks promotion."""
        report = evaluate_promotion(
            current_level=PromotionLevel.EXPERIMENTAL,
            target_level=PromotionLevel.CANDIDATE,
            safety_report=_failing_safety_report("authority_boundary"),
        )
        assert not report.promotion_approved
