"""Information-directed exploration (Phase 13).

Information-directed sampling balances exploration and exploitation by
estimating the *information gain* (IG) of each candidate action: how much
will taking this action reduce our uncertainty about the optimal structural
policy?

The core formula is:

  IG(a) = rho(sigma_pred, sigma_post) * H(reduction)

where:
  - sigma_pred is the predicted uncertainty before the action
  - sigma_post is the predicted uncertainty after the action
  - rho is the correlation between predicted and realized information gain

Actions with high IG are preferred when the policy is uncertain. This
complements the learned score (which captures expected utility) by adding
an exploration bonus.

The implementation supports:
  - Thompson sampling via ensemble disagreement
  - UCB-style bonuses from epistemic uncertainty
  - Information gain from posterior variance reduction
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class InformationGainEstimate:
    """Information gain estimate for one candidate action."""
    candidate_id: str
    predicted_ig: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    exploration_bonus: float
    total_score: float  # learned_score + exploration_bonus

    def to_log(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "predicted_ig": float(self.predicted_ig),
            "epistemic_uncertainty": float(self.epistemic_uncertainty),
            "aleatoric_uncertainty": float(self.aleatoric_uncertainty),
            "exploration_bonus": float(self.exploration_bonus),
            "total_score": float(self.total_score),
        }


@dataclass(slots=True)
class InformationDirectedReport:
    """Report of information-directed exploration for one step."""
    estimates: list[InformationGainEstimate] = field(default_factory=list)
    chosen_candidate_id: str = ""
    ig_correlation: float = 0.0  # rho(predicted_IG, realized_IG)

    def to_log(self) -> dict[str, Any]:
        return {
            "estimate_count": len(self.estimates),
            "chosen_candidate_id": self.chosen_candidate_id,
            "ig_correlation": float(self.ig_correlation),
            "estimates": [e.to_log() for e in self.estimates],
        }


def ensemble_disagreement_ig(
    *,
    candidate_ids: list[str],
    ensemble_scores: Tensor,  # [N_candidates, N_ensemble_members]
    learned_scores: Tensor,  # [N_candidates]
    exploration_weight: float = 1.0,
) -> list[InformationGainEstimate]:
    """Estimate IG from ensemble disagreement (Thompson sampling style).

    The epistemic uncertainty is the variance across ensemble members.
    The predicted IG is proportional to this variance: candidates with
    high disagreement are expected to yield more information.
    """
    if ensemble_scores.numel() == 0:
        return []
    n_candidates, n_ensemble = ensemble_scores.shape
    # Epistemic uncertainty: std across ensemble members.
    epistemic = ensemble_scores.std(dim=1) if n_ensemble > 1 else torch.zeros(n_candidates)
    # Aleatoric uncertainty: intrinsic noise (estimated as 0 without more info).
    aleatoric = torch.zeros(n_candidates)
    # Predicted IG: proportional to epistemic uncertainty.
    predicted_ig = epistemic * exploration_weight
    # Exploration bonus: exploration_weight * epistemic_uncertainty.
    exploration_bonus = exploration_weight * epistemic
    # Total score: learned + bonus.
    total = learned_scores + exploration_bonus

    estimates: list[InformationGainEstimate] = []
    for i in range(n_candidates):
        estimates.append(InformationGainEstimate(
            candidate_id=candidate_ids[i],
            predicted_ig=float(predicted_ig[i]),
            epistemic_uncertainty=float(epistemic[i]),
            aleatoric_uncertainty=float(aleatoric[i]),
            exploration_bonus=float(exploration_bonus[i]),
            total_score=float(total[i]),
        ))
    return estimates


def ucb_ig(
    *,
    candidate_ids: list[str],
    mean_scores: Tensor,  # [N_candidates]
    uncertainty: Tensor,  # [N_candidates]
    exploration_weight: float = 2.0,
) -> list[InformationGainEstimate]:
    """UCB-style information gain: total = mean + weight * uncertainty."""
    n = len(candidate_ids)
    bonus = exploration_weight * uncertainty
    total = mean_scores + bonus
    return [
        InformationGainEstimate(
            candidate_id=candidate_ids[i],
            predicted_ig=float(uncertainty[i]),
            epistemic_uncertainty=float(uncertainty[i]),
            aleatoric_uncertainty=0.0,
            exploration_bonus=float(bonus[i]),
            total_score=float(total[i]),
        )
        for i in range(n)
    ]


def posterior_variance_reduction_ig(
    *,
    candidate_ids: list[str],
    prior_variance: Tensor,  # [N_candidates]
    expected_posterior_variance: Tensor,  # [N_candidates]
    learned_scores: Tensor,  # [N_candidates]
    exploration_weight: float = 1.0,
) -> list[InformationGainEstimate]:
    """IG from expected posterior variance reduction.

    IG(a) = prior_var(a) - expected_post_var(a)
    Candidates that maximally reduce variance have the highest IG.
    """
    ig = prior_variance - expected_posterior_variance
    ig = torch.clamp(ig, min=0.0)  # IG is non-negative
    bonus = exploration_weight * ig
    total = learned_scores + bonus
    return [
        InformationGainEstimate(
            candidate_id=candidate_ids[i],
            predicted_ig=float(ig[i]),
            epistemic_uncertainty=float(prior_variance[i]),
            aleatoric_uncertainty=0.0,
            exploration_bonus=float(bonus[i]),
            total_score=float(total[i]),
        )
        for i in range(len(candidate_ids))
    ]


def compute_ig_correlation(
    predicted_ig: list[float],
    realized_ig: list[float],
) -> float:
    """Compute Pearson correlation between predicted and realized IG.

    A positive correlation means the IG estimates are informative.
    The scientific gate (Phase 47) requires rho > 0 on every seed.
    """
    if len(predicted_ig) < 2 or len(realized_ig) < 2:
        return 0.0
    if len(predicted_ig) != len(realized_ig):
        return 0.0
    pred = torch.tensor(predicted_ig)
    real = torch.tensor(realized_ig)
    if pred.std() < 1e-10 or real.std() < 1e-10:
        return 0.0
    return float(torch.corrcoef(torch.stack([pred, real]))[0, 1].item())


def select_information_directed(
    estimates: list[InformationGainEstimate],
) -> str:
    """Select the candidate with the highest total score (learned + IG bonus)."""
    if not estimates:
        return ""
    best = max(estimates, key=lambda e: e.total_score)
    return best.candidate_id
