"""Delta-state dynamics model for exp5.2.

Instead of predicting z_{t+1} directly, predicts:
    Δz = F(z_t, a_t)
then reconstructs:
    z_{t+1} = z_t + Δz

This is a better inductive bias for local graph mutations:
- ADD_EDGE changes structural statistics incrementally
- The model learns the *effect* of an action, not the absolute next state
- Delta targets are more similar across graph families than absolute states

Supports both absolute-state and delta-state modes for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json
import math
import numpy as np

from .state_encoding import NORM_STATE_DIM, NORM_ACTION_DIM, norm_state_action_schema_hash


class DeltaDynamicsModel:
    """Dynamics model with delta-state prediction.

    Modes:
    - "delta": predict Δz = F(z_t, a_t), reconstruct z_{t+1} = z_t + Δz
    - "absolute": predict z_{t+1} = F(z_t, a_t) directly (baseline)

    The delta mode is the primary exp5.2 contribution.
    """

    model_type = "delta_dynamics"
    version = "v6.0-exp5.2"
    deterministic = True

    def __init__(
        self,
        *,
        mode: str = "delta",  # "delta" or "absolute"
        lr: float = 0.01,
        n_epochs: int = 200,
        seed: int = 42,
        regularization: float = 1e-3,
        state_dim: int = NORM_STATE_DIM,
        action_dim: int = NORM_ACTION_DIM,
    ) -> None:
        self.mode = mode
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.seed = int(seed)
        self.regularization = float(regularization)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self._A: np.ndarray | None = None  # (state_dim, state_dim)
        self._B: np.ndarray | None = None  # (state_dim, action_dim)
        self._c: np.ndarray | None = None  # (state_dim,)
        self._fitted = False
        self._n_samples = 0
        self._target_mean: np.ndarray | None = None  # for delta normalization

    @property
    def model_id(self) -> str:
        return f"{self.model_type}-{self.version}-{self.mode}"

    @property
    def requires_fit(self) -> bool:
        return True

    @property
    def n_parameters(self) -> int:
        if self._A is None:
            return 0
        return self._A.size + self._B.size + self._c.size

    @property
    def schema_hash(self) -> str:
        return norm_state_action_schema_hash()

    def _compute_targets(
        self,
        z_t: np.ndarray,
        z_next: np.ndarray,
    ) -> np.ndarray:
        """Compute training targets based on mode."""
        if self.mode == "delta":
            return z_next - z_t
        return z_next

    def fit(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
        z_next: np.ndarray,
        *,
        split: str = "train",
    ) -> None:
        if split != "train":
            raise ValueError(f"Can only fit on train split, got '{split}'.")

        n = len(z_t)
        if n == 0:
            self._A = np.eye(self.state_dim)
            self._B = np.zeros((self.state_dim, self.action_dim))
            self._c = np.zeros(self.state_dim)
            self._fitted = True
            return

        targets = self._compute_targets(z_t, z_next)

        # Store target mean for normalization (delta mode only).
        if self.mode == "delta":
            self._target_mean = targets.mean(axis=0)
        else:
            self._target_mean = None

        # Ridge regression: targets = [z_t, a_t, 1] @ W
        X = np.hstack([z_t, a_t, np.ones((n, 1))])
        lam = self.regularization
        XtX = X.T @ X + lam * np.eye(X.shape[1])
        XtY = X.T @ targets
        W = np.linalg.solve(XtX, XtY)

        self._A = W[:self.state_dim, :].T
        self._B = W[self.state_dim:self.state_dim + self.action_dim, :].T
        self._c = W[-1, :]
        self._fitted = True
        self._n_samples = n

    def predict_raw(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
    ) -> np.ndarray:
        """Predict the raw target (delta or absolute)."""
        if not self._fitted:
            return np.zeros(self.state_dim)
        return self._A @ z_t + self._B @ a_t + self._c

    def predict(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
    ) -> np.ndarray:
        """Predict next state (reconstructing from delta if needed)."""
        raw = self.predict_raw(z_t, a_t)
        if self.mode == "delta":
            return z_t + raw
        return raw

    def predict_batch(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
    ) -> np.ndarray:
        """Batch prediction."""
        if z_t.ndim == 1:
            z_t = z_t[np.newaxis, :]
        if a_t.ndim == 1:
            a_t = a_t[np.newaxis, :]
        if not self._fitted:
            return z_t.copy()
        raw = z_t @ self._A.T + a_t @ self._B.T + self._c
        if self.mode == "delta":
            return z_t + raw
        return raw

    def predict_raw_batch(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
    ) -> np.ndarray:
        """Batch raw target prediction (for uncertainty)."""
        if z_t.ndim == 1:
            z_t = z_t[np.newaxis, :]
        if a_t.ndim == 1:
            a_t = a_t[np.newaxis, :]
        if not self._fitted:
            return np.zeros((len(z_t), self.state_dim))
        return z_t @ self._A.T + a_t @ self._B.T + self._c

    def freeze(self) -> None:
        pass

    def get_state(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "A": self._A.tolist() if self._A is not None else None,
            "B": self._B.tolist() if self._B is not None else None,
            "c": self._c.tolist() if self._c is not None else None,
            "fitted": self._fitted,
            "n_samples": self._n_samples,
            "target_mean": self._target_mean.tolist() if self._target_mean is not None else None,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.mode = state.get("mode", "delta")
        self._A = np.array(state["A"]) if state.get("A") else None
        self._B = np.array(state["B"]) if state.get("B") else None
        self._c = np.array(state["c"]) if state.get("c") else None
        self._fitted = state.get("fitted", False)
        self._n_samples = state.get("n_samples", 0)
        tm = state.get("target_mean")
        self._target_mean = np.array(tm) if tm else None

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "version": self.version,
            "mode": self.mode,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "lr": self.lr,
            "n_epochs": self.n_epochs,
            "seed": self.seed,
            "regularization": self.regularization,
        }


# ---------------------------------------------------------------------------
# Ensemble delta dynamics (family-bootstrap).
# ---------------------------------------------------------------------------

class FamilyBootstrapEnsemble:
    """Ensemble where each member trains on a different subset of graph families.

    This is more meaningful than row bootstrapping for measuring
    structural extrapolation uncertainty. If TEST-B produces strong
    member disagreement, it indicates the test family is OOD relative
    to the training distribution of some members.

    Members:
        member 0: train on all families
        member 1: omit family[0]
        member 2: omit family[1]
        ...
        member N: omit family[N-1]
    """

    model_type = "family_bootstrap_ensemble"
    version = "v6.0-exp5.2"
    deterministic = True

    def __init__(
        self,
        *,
        mode: str = "delta",
        n_members: int = 5,
        lr: float = 0.01,
        n_epochs: int = 200,
        seed: int = 42,
        regularization: float = 1e-3,
    ) -> None:
        self.mode = mode
        self.n_members = int(n_members)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.seed = int(seed)
        self.regularization = float(regularization)
        self._members: list[DeltaDynamicsModel] = []
        self._fitted = False
        self._n_samples = 0

    @property
    def n_parameters(self) -> int:
        return sum(m.n_parameters for m in self._members)

    def fit_with_family_split(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
        z_next: np.ndarray,
        family_labels: list[str],
        *,
        split: str = "train",
    ) -> None:
        """Fit ensemble members, each omitting one family.

        Args:
            z_t, a_t, z_next: Training data.
            family_labels: Graph family name for each sample.
            split: Must be "train".
        """
        if split != "train":
            raise ValueError(f"Can only fit on train split, got '{split}'.")

        unique_families = sorted(set(family_labels))
        n_families = len(unique_families)

        self._members = []

        # Member 0: train on all data.
        m0 = DeltaDynamicsModel(
            mode=self.mode, lr=self.lr, n_epochs=self.n_epochs,
            seed=self.seed, regularization=self.regularization,
        )
        m0.fit(z_t, a_t, z_next, split="train")
        self._members.append(m0)

        # Members 1..N: each omits one family.
        for i, fam in enumerate(unique_families[:self.n_members - 1]):
            mask = np.array([f != fam for f in family_labels])
            if mask.sum() < 10:
                continue  # skip if too few samples
            m = DeltaDynamicsModel(
                mode=self.mode, lr=self.lr, n_epochs=self.n_epochs,
                seed=self.seed + i * 1000, regularization=self.regularization,
            )
            m.fit(z_t[mask], a_t[mask], z_next[mask], split="train")
            self._members.append(m)

        self._fitted = True
        self._n_samples = len(z_t)

    def predict(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
    ) -> np.ndarray:
        """Mean prediction across ensemble."""
        if not self._fitted or not self._members:
            return z_t.copy()
        preds = np.array([m.predict(z_t, a_t) for m in self._members])
        return preds.mean(axis=0)

    def predict_batch(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
    ) -> np.ndarray:
        if not self._fitted or not self._members:
            return z_t.copy()
        preds = np.array([m.predict_batch(z_t, a_t) for m in self._members])
        return preds.mean(axis=0)

    def predict_uncertainty_batch(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
    ) -> np.ndarray:
        """Per-prediction uncertainty from family-bootstrap disagreement."""
        if not self._fitted or not self._members:
            return np.zeros(len(z_t))
        preds = np.array([m.predict_batch(z_t, a_t) for m in self._members])
        stds = preds.std(axis=0).mean(axis=1)
        return stds

    def freeze(self) -> None:
        pass

    def get_state(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "n_members": len(self._members),
            "members": [m.get_state() for m in self._members],
            "fitted": self._fitted,
            "n_samples": self._n_samples,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.mode = state.get("mode", "delta")
        self._fitted = state.get("fitted", False)
        self._n_samples = state.get("n_samples", 0)
        self._members = []
        for ms in state.get("members", []):
            m = DeltaDynamicsModel(mode=self.mode)
            m.set_state(ms)
            self._members.append(m)

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "version": self.version,
            "mode": self.mode,
            "n_members": self.n_members,
            "state_dim": NORM_STATE_DIM,
            "action_dim": NORM_ACTION_DIM,
        }


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GeneralizationMetrics:
    """Metrics for cross-family generalization evaluation."""
    one_step_r2: float = 0.0
    one_step_rmse: float = 0.0
    one_step_delta_r2: float = 0.0  # R² on delta targets
    one_step_delta_rmse: float = 0.0
    spearman: float = 0.0
    n_samples: int = 0
    per_feature_nrmse: list[float] = field(default_factory=list)
    mean_uncertainty: float = 0.0
    calibration_corr: float = 0.0
    calibration_spearman: float = 0.0

    def to_log(self) -> dict[str, Any]:
        return {
            "one_step_r2": float(self.one_step_r2),
            "one_step_rmse": float(self.one_step_rmse),
            "one_step_delta_r2": float(self.one_step_delta_r2),
            "one_step_delta_rmse": float(self.one_step_delta_rmse),
            "spearman": float(self.spearman),
            "n_samples": int(self.n_samples),
            "per_feature_nrmse": [float(x) for x in self.per_feature_nrmse],
            "mean_uncertainty": float(self.mean_uncertainty),
            "calibration_corr": float(self.calibration_corr),
            "calibration_spearman": float(self.calibration_spearman),
        }


def compute_generalization_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    *,
    predicted_delta: np.ndarray | None = None,
    actual_delta: np.ndarray | None = None,
    uncertainties: np.ndarray | None = None,
) -> GeneralizationMetrics:
    """Compute generalization metrics with per-feature NRMSE."""
    if len(predicted) == 0:
        return GeneralizationMetrics()

    diff = predicted - actual
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((actual - actual.mean(axis=0)) ** 2))
    r2 = max(-10.0, min(1.0, 1.0 - ss_res / max(ss_tot, 1e-10)))

    # Per-feature NRMSE.
    feat_std = np.std(actual, axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    norm_diff = diff / feat_std
    per_feat_nrmse = [float(np.sqrt(np.mean(norm_diff[:, j] ** 2)))
                      for j in range(actual.shape[1])]

    # Delta metrics.
    delta_r2 = 0.0
    delta_rmse = 0.0
    if predicted_delta is not None and actual_delta is not None:
        d_diff = predicted_delta - actual_delta
        delta_rmse = float(np.sqrt(np.mean(d_diff ** 2)))
        d_ss_res = float(np.sum(d_diff ** 2))
        d_ss_tot = float(np.sum((actual_delta - actual_delta.mean(axis=0)) ** 2))
        delta_r2 = max(-10.0, min(1.0, 1.0 - d_ss_res / max(d_ss_tot, 1e-10)))

    # Spearman correlation (simplified).
    spearman = 0.0
    try:
        from scipy.stats import spearmanr
        flat_pred = predicted.mean(axis=1)
        flat_actual = actual.mean(axis=1)
        if len(flat_pred) > 1:
            sp, _ = spearmanr(flat_pred, flat_actual)
            spearman = float(sp) if not math.isnan(sp) else 0.0
    except Exception:
        pass

    # Calibration.
    cal_corr = 0.0
    cal_sp = 0.0
    if uncertainties is not None and len(uncertainties) == len(predicted):
        errors = np.sqrt(np.sum(diff ** 2, axis=1))
        if len(uncerts := uncertainties) > 1 and np.std(uncerts) > 1e-10 and np.std(errors) > 1e-10:
            cal_corr = float(np.corrcoef(uncerts, errors)[0, 1])
        try:
            from scipy.stats import spearmanr
            if len(uncerts) > 1 and np.std(uncerts) > 1e-10:
                sp, _ = spearmanr(uncerts, errors)
                cal_sp = float(sp) if not math.isnan(sp) else 0.0
        except Exception:
            pass

    return GeneralizationMetrics(
        one_step_r2=r2,
        one_step_rmse=rmse,
        one_step_delta_r2=delta_r2,
        one_step_delta_rmse=delta_rmse,
        spearman=spearman,
        n_samples=len(predicted),
        per_feature_nrmse=per_feat_nrmse,
        mean_uncertainty=float(np.mean(uncertainties)) if uncertainties is not None else 0.0,
        calibration_corr=cal_corr,
        calibration_spearman=cal_sp,
    )
