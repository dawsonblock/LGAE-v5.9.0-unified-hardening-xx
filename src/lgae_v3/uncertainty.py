"""Calibrated uncertainty for structural mutations.

v5.1.1 fixes two critical defects from v5.1.0:
- uncertainty estimation never swaps weights into the authoritative executive;
- conformal order statistics follow the split-conformal finite-sample rule.

The ensemble members are independent network copies. They can be updated with
bootstrap-style supervised outcomes through :meth:`update` so uncertainty stays
anchored to the current structural task instead of frozen near initialization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy
import math
import random

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .executive import StructuralExecutive, ActionProposal


@dataclass
class UncertaintyEstimate:
    mean: float
    std: float
    lcb: float
    ucb: float
    calibration_error: float = 0.0
    method: str = "bootstrap_ensemble"
    metadata: dict[str, Any] = field(default_factory=dict)


class EnsembleUncertainty:
    """Bootstrap ensemble around the structural value model.

    Each member is a real independent module. Calling ``estimate`` is strictly
    read-only with respect to the authoritative executive. ``update`` trains
    members on observed outcomes with independent bootstrap masks.
    """

    def __init__(
        self,
        executive: StructuralExecutive,
        ensemble_size: int = 5,
        beta: float = 1.0,
        lr: float = 1e-3,
        init_noise: float = 0.02,
        bootstrap_probability: float = 0.8,
    ):
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive")
        self.executive = executive
        self.ensemble_size = int(ensemble_size)
        self.beta = float(beta)
        self.bootstrap_probability = float(bootstrap_probability)
        self.members: list[torch.nn.Module] = []
        self.optimizers: list[torch.optim.Optimizer] = []
        for _ in range(self.ensemble_size):
            member = copy.deepcopy(executive.network)
            with torch.no_grad():
                for p in member.parameters():
                    p.add_(float(init_noise) * torch.randn_like(p))
            self.members.append(member)
            self.optimizers.append(torch.optim.Adam(member.parameters(), lr=lr))

    @property
    def ensemble(self) -> list[dict[str, Tensor]]:
        """Backward-compatible state view used by older tests/tools."""
        return [
            {k: v.detach().clone() for k, v in m.state_dict().items()}
            for m in self.members
        ]

    def estimate(self, observation_vec: Tensor, action_idx: int) -> UncertaintyEstimate:
        predictions: list[float] = []
        for member in self.members:
            member.eval()
            with torch.no_grad():
                preds = member(observation_vec)
                predictions.append(float(preds["delta_u"][action_idx].item()))

        mean = float(np.mean(predictions))
        std = float(np.std(predictions))
        lcb = mean - self.beta * std
        ucb = mean + self.beta * std
        return UncertaintyEstimate(
            mean=mean,
            std=std,
            lcb=lcb,
            ucb=ucb,
            method="ensemble",
            metadata={"backend": "bootstrap_ensemble", "ensemble_size": self.ensemble_size, "predictions": predictions},
        )

    def update(
        self,
        observation_vec: Tensor,
        action_idx: int,
        target_delta_u: float,
        *,
        cost_target: float | None = None,
        risk_target: float | None = None,
        ig_target: float | None = None,
        update_delta: bool = True,
    ) -> int:
        """Train ensemble members on one observed structural outcome.

        Returns the number of members updated. The independent Bernoulli masks
        create a lightweight online bootstrap posterior approximation.
        """
        updated = 0
        target = torch.as_tensor(target_delta_u, dtype=observation_vec.dtype, device=observation_vec.device)
        for member, optimizer in zip(self.members, self.optimizers):
            if random.random() > self.bootstrap_probability:
                continue
            member.train()
            preds = member(observation_vec)
            loss = torch.zeros((), dtype=target.dtype, device=target.device)
            if update_delta:
                loss = loss + F.mse_loss(preds["delta_u"][action_idx], target)
            if cost_target is not None:
                loss = loss + 0.25 * F.mse_loss(
                    preds["cost"][action_idx],
                    torch.as_tensor(cost_target, dtype=target.dtype, device=target.device),
                )
            if risk_target is not None:
                loss = loss + 0.25 * F.mse_loss(
                    preds["risk"][action_idx],
                    torch.as_tensor(risk_target, dtype=target.dtype, device=target.device),
                )
            if ig_target is not None:
                loss = loss + 0.25 * F.mse_loss(
                    preds["ig"][action_idx],
                    torch.as_tensor(ig_target, dtype=target.dtype, device=target.device),
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updated += 1
        return updated


class ConformalCalibrator:
    """Online split-conformal residual calibrator.

    For ``n`` calibration residuals and miscoverage ``alpha``, the finite-sample
    order statistic is ``ceil((n+1)(1-alpha))`` (clipped to ``n``), not divided
    by ``n`` before converting to an index.
    """

    def __init__(self, alpha: float = 0.1, max_residuals: int = 4096):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie in (0,1)")
        self.alpha = float(alpha)
        self.max_residuals = int(max_residuals)
        self._residuals: list[float] = []
        self._quantile: float | None = None

    def _recompute(self) -> float:
        if not self._residuals:
            self._quantile = 0.0
            return 0.0
        vals = sorted(float(x) for x in self._residuals)
        n = len(vals)
        k = min(n, max(1, int(math.ceil((n + 1) * (1.0 - self.alpha)))))
        self._quantile = vals[k - 1]
        return self._quantile

    def calibrate(self, predicted: list[float], actual: list[float]) -> float:
        if len(predicted) != len(actual):
            raise ValueError(
                f"predicted and actual must have same length: {len(predicted)} vs {len(actual)}"
            )
        self._residuals = [abs(float(p) - float(a)) for p, a in zip(predicted, actual)][-self.max_residuals:]
        return self._recompute()

    def update(self, predicted: float, actual: float) -> float:
        self._residuals.append(abs(float(predicted) - float(actual)))
        if len(self._residuals) > self.max_residuals:
            del self._residuals[: len(self._residuals) - self.max_residuals]
        return self._recompute()

    @property
    def calibrated(self) -> bool:
        return bool(self._residuals)

    @property
    def quantile(self) -> float | None:
        return self._quantile

    def interval(self, prediction: float) -> tuple[float, float]:
        if self._quantile is None:
            return (float(prediction), float(prediction))
        return (float(prediction) - self._quantile, float(prediction) + self._quantile)

    def lcb(self, prediction: float, beta: float = 1.0) -> float:
        lower, _ = self.interval(prediction)
        return lower


def uncertainty_gated_decision(
    proposal: ActionProposal,
    uncertainty: UncertaintyEstimate,
    lcb_threshold: float = 0.0,
    quarantine_uncertainty: float = 0.5,
    *,
    conformal_interval: tuple[float, float] | None = None,
) -> str:
    """Return ``accept``, ``quarantine`` or ``reject``.

    When a conformal interval is available it is combined conservatively with
    the ensemble interval: the lower bound is the minimum of both lower bounds
    and the upper bound is the maximum of both upper bounds.
    """
    lcb = float(uncertainty.lcb)
    ucb = float(uncertainty.ucb)
    if conformal_interval is not None:
        lcb = min(lcb, float(conformal_interval[0]))
        ucb = max(ucb, float(conformal_interval[1]))
    sigma = float(uncertainty.std)

    if lcb > lcb_threshold:
        return "accept" if sigma < quarantine_uncertainty else "quarantine"
    if ucb > lcb_threshold:
        return "quarantine"
    return "reject"
