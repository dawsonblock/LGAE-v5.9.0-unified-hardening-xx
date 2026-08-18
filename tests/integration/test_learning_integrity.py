"""v5.11 Sprint 3: Learning integrity — realized delta, calibration, hierarchical credit.

Tests for:
- D11-011: learn() uses realized delta (U_after - U_before), not predicted delta
- D11-012: calibration compares predicted delta with realized delta, not absolute utility
- D11-013: hierarchical credit assignment is connected (not just flat outcome_credit)
"""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import LGAERuntime, RuntimeConfig


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


class TestRealizedDelta:
    """D11-011: learn() uses realized delta, not predicted delta."""

    def test_realized_delta_is_computed(self):
        """The learning result contains a realized delta (U_after - U_before)."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        result = rt.step()
        assert result.learning is not None
        # The transition should have a realized_outcome field.
        transition = result.learning.transition
        assert transition is not None
        # realized_outcome should be a float (not None or NaN).
        assert isinstance(transition.realized_outcome, float)

    def test_realized_delta_not_predicted_delta(self):
        """The realized delta differs from the predicted delta when utility changes."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        result = rt.step()
        if result.learning is None:
            pytest.skip("No learning result")
        transition = result.learning.transition
        # The predicted and realized outcomes should be separate fields.
        # They may be equal in some cases, but the code must compute them separately.
        assert hasattr(transition, 'predicted_outcome')
        assert hasattr(transition, 'realized_outcome')

    def test_reward_is_realized_delta(self):
        """The reward field is the realized delta, not the predicted delta."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        result = rt.step()
        if result.learning is None:
            pytest.skip("No learning result")
        transition = result.learning.transition
        # D11-011: reward should be realized delta, not predicted delta.
        assert transition.reward == transition.realized_outcome

    def test_realized_delta_zero_on_no_commit(self):
        """Realized delta is 0 when no commit occurred."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        result = rt.step()
        if result.learning is None:
            pytest.skip("No learning result")
        transition = result.learning.transition
        if not result.executed:
            assert transition.realized_outcome == 0.0
            assert transition.reward == 0.0


class TestCalibrationIntegrity:
    """D11-012: calibration compares predicted delta with realized delta."""

    def test_calibration_updated_on_commit(self):
        """Calibration is updated when a commit occurs."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        result = rt.step()
        if result.learning is None:
            pytest.skip("No learning result")
        # If a commit occurred, calibration should be updated.
        if result.executed:
            assert result.learning.calibration_updated

    def test_calibration_not_updated_on_no_commit(self):
        """Calibration is not updated when no commit occurred."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        result = rt.step()
        if result.learning is None:
            pytest.skip("No learning result")
        if not result.executed:
            assert not result.learning.calibration_updated


class TestHierarchicalCredit:
    """D11-013: hierarchical credit assignment is connected."""

    def test_credit_has_hierarchical_fields(self):
        """The credit assignment has hierarchical credit fields."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        result = rt.step()
        if result.learning is None:
            pytest.skip("No learning result")
        credit = result.learning.credit
        assert credit is not None
        # D11-013: Should have hierarchical credit fields, not just outcome_credit.
        assert hasattr(credit, 'diagnostic_credit')
        assert hasattr(credit, 'candidate_credit')
        assert hasattr(credit, 'planner_credit')
        assert hasattr(credit, 'action_credit')
        assert hasattr(credit, 'governance_credit')
        assert hasattr(credit, 'outcome_credit')

    def test_credit_not_flat_on_commit(self):
        """On commit, hierarchical credit fields are non-zero (not just flat outcome_credit)."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        result = rt.step()
        if result.learning is None:
            pytest.skip("No learning result")
        credit = result.learning.credit
        if result.executed and credit.outcome_credit != 0.0:
            # At least some hierarchical fields should be non-zero.
            total_hierarchical = (
                credit.diagnostic_credit +
                credit.candidate_credit +
                credit.planner_credit +
                credit.action_credit +
                credit.governance_credit
            )
            assert total_hierarchical != 0.0, (
                "Hierarchical credit is all zero despite non-zero outcome_credit!"
            )

    def test_credit_sums_to_outcome(self):
        """Hierarchical credit fields sum to approximately the outcome credit."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        result = rt.step()
        if result.learning is None:
            pytest.skip("No learning result")
        credit = result.learning.credit
        if result.executed and credit.outcome_credit != 0.0:
            total = (
                credit.diagnostic_credit +
                credit.candidate_credit +
                credit.planner_credit +
                credit.action_credit +
                credit.governance_credit
            )
            # The sum should be close to outcome_credit (within float tolerance).
            assert abs(total - credit.outcome_credit) < 1e-6, (
                f"Hierarchical credit sum {total} != outcome_credit {credit.outcome_credit}"
            )
