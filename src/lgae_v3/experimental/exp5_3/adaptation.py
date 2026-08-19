"""Component-wise adaptation for exp5.3.

Instead of retraining the entire model with k adaptation samples,
adapt only specific components:

1. bias-only: adapt only the bias term b
   Δz = Wz + Ba + b_adapted
   This tests whether the global dynamics are correct but need a
   family-specific offset.

2. scale+offset: adapt output scale s and offset b
   Δz = s * (Wz + Ba) + b_adapted
   This tests whether the global dynamics need rescaling.

3. low-rank: adapt a low-rank correction UV^T
   Δz = Wz + Ba + b + U(V^T [z;a])
   This tests whether a small structural correction is sufficient.

4. full: retrain the entire model on global + adaptation data.
   This is the baseline (current approach).

If bias-only adaptation works with 5-25 samples, the architecture is:
    Δz = F_θ(z, a) + C_φ_G(z, a)
where F_θ is the global prior and C_φ_G is a tiny local correction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from ..exp5_2.dynamics import DeltaDynamicsModel


class ComponentAdapter:
    """Base class for component-wise adaptation."""

    def __init__(self, base_model: DeltaDynamicsModel) -> None:
        self.base_model = base_model
        self._fitted = False

    def fit(self, z_t: np.ndarray, a_t: np.ndarray, z_next: np.ndarray) -> None:
        """Fit the adaptation component on local data."""
        raise NotImplementedError

    def predict(self, z_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        """Predict next state using adapted model."""
        raise NotImplementedError

    def predict_delta(self, z_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        """Predict delta using adapted model."""
        raise NotImplementedError

    @property
    def adaptation_type(self) -> str:
        return "base"


class BiasOnlyAdapter(ComponentAdapter):
    """Adapt only the bias term.

    Δz = Wz + Ba + b + b_local
    where b_local is learned from adaptation data.
    """

    @property
    def adaptation_type(self) -> str:
        return "bias_only"

    def fit(self, z_t: np.ndarray, a_t: np.ndarray, z_next: np.ndarray) -> None:
        # Compute residuals from base model.
        base_delta = self.base_model.predict_raw_batch(z_t, a_t)
        actual_delta = z_next - z_t
        residual = actual_delta - base_delta
        # Bias = mean residual.
        self._bias = residual.mean(axis=0)
        self._fitted = True

    def predict_delta(self, z_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        base = self.base_model.predict_raw_batch(z_t, a_t)
        if self._fitted:
            return base + self._bias
        return base

    def predict(self, z_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        return z_t + self.predict_delta(z_t, a_t)


class ScaleOffsetAdapter(ComponentAdapter):
    """Adapt output scale and offset.

    Δz = s * (Wz + Ba + b) + b_local
    where s is a per-dimension scale and b_local is an offset.
    """

    @property
    def adaptation_type(self) -> str:
        return "scale_offset"

    def fit(self, z_t: np.ndarray, a_t: np.ndarray, z_next: np.ndarray) -> None:
        base_delta = self.base_model.predict_raw_batch(z_t, a_t)
        actual_delta = z_next - z_t
        # Linear regression: actual = s * base + b_local
        # Solve per-dimension.
        n, d = actual_delta.shape
        self._scale = np.ones(d)
        self._offset = np.zeros(d)
        for j in range(d):
            x = base_delta[:, j]
            y = actual_delta[:, j]
            if np.std(x) > 1e-10:
                # OLS: y = s*x + b
                s = np.cov(x, y)[0, 1] / np.var(x)
                b = y.mean() - s * x.mean()
                self._scale[j] = s
                self._offset[j] = b
            else:
                self._offset[j] = y.mean()
        self._fitted = True

    def predict_delta(self, z_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        base = self.base_model.predict_raw_batch(z_t, a_t)
        if self._fitted:
            return self._scale * base + self._offset
        return base

    def predict(self, z_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        return z_t + self.predict_delta(z_t, a_t)


class LowRankAdapter(ComponentAdapter):
    """Adapt a low-rank correction.

    Δz = Wz + Ba + b + U(V^T [z; a])
    where U is (d, r) and V is (state_dim+action_dim, r), r is small.
    """

    def __init__(self, base_model: DeltaDynamicsModel, rank: int = 2) -> None:
        super().__init__(base_model)
        self.rank = rank

    @property
    def adaptation_type(self) -> str:
        return f"low_rank_r{self.rank}"

    def fit(self, z_t: np.ndarray, a_t: np.ndarray, z_next: np.ndarray) -> None:
        base_delta = self.base_model.predict_raw_batch(z_t, a_t)
        actual_delta = z_next - z_t
        residual = actual_delta - base_delta  # (n, d)

        # Fit U, V via SVD of the residual mapped to input space.
        X = np.hstack([z_t, a_t])  # (n, state_dim+action_dim)
        n, d = residual.shape

        if n < self.rank + 1:
            # Not enough samples for low-rank; fall back to bias-only.
            self._U = np.zeros((d, self.rank))
            self._V = np.zeros((X.shape[1], self.rank))
            self._bias = residual.mean(axis=0)
            self._use_bias = True
        else:
            # Solve: residual ≈ X @ V @ U^T
            # Use reduced-rank regression.
            # X^T residual = (X^T X) V U^T
            # Simplified: SVD of X^T @ residual
            M = X.T @ residual  # (input_dim, d)
            U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
            r = min(self.rank, len(S))
            self._V = U_svd[:, :r]  # (input_dim, r)
            self._U = (Vt[:r, :]).T * S[:r]  # (d, r)
            self._bias = residual.mean(axis=0) - X.mean(axis=0) @ self._V @ self._U.T
            self._use_bias = True

        self._fitted = True

    def predict_delta(self, z_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        base = self.base_model.predict_raw_batch(z_t, a_t)
        if not self._fitted:
            return base
        X = np.hstack([z_t, a_t])
        correction = X @ self._V @ self._U.T
        if self._use_bias:
            correction = correction + self._bias
        return base + correction

    def predict(self, z_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        return z_t + self.predict_delta(z_t, a_t)


class FullRetrainAdapter(ComponentAdapter):
    """Full retrain: combine global + adaptation data and retrain."""

    def __init__(
        self,
        base_model: DeltaDynamicsModel,
        global_z_t: np.ndarray,
        global_a_t: np.ndarray,
        global_z_next: np.ndarray,
    ) -> None:
        super().__init__(base_model)
        self.global_z_t = global_z_t
        self.global_a_t = global_a_t
        self.global_z_next = global_z_next
        self._adapted_model: DeltaDynamicsModel | None = None

    @property
    def adaptation_type(self) -> str:
        return "full_retrain"

    def fit(self, z_t: np.ndarray, a_t: np.ndarray, z_next: np.ndarray) -> None:
        combined_z = np.vstack([self.global_z_t, z_t])
        combined_a = np.vstack([self.global_a_t, a_t])
        combined_zn = np.vstack([self.global_z_next, z_next])
        self._adapted_model = DeltaDynamicsModel(
            mode=self.base_model.mode,
            regularization=self.base_model.regularization,
            seed=self.base_model.seed,
            state_dim=self.base_model.state_dim,
            action_dim=self.base_model.action_dim,
        )
        self._adapted_model.fit(combined_z, combined_a, combined_zn, split="train")
        self._fitted = True

    def predict_delta(self, z_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        if self._adapted_model is None:
            return self.base_model.predict_raw_batch(z_t, a_t)
        return self._adapted_model.predict_raw_batch(z_t, a_t)

    def predict(self, z_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        if self._adapted_model is None:
            return self.base_model.predict(z_t, a_t)
        return self._adapted_model.predict(z_t, a_t)


def create_adapter(
    adaptation_type: str,
    base_model: DeltaDynamicsModel,
    *,
    global_z_t: np.ndarray | None = None,
    global_a_t: np.ndarray | None = None,
    global_z_next: np.ndarray | None = None,
    rank: int = 2,
) -> ComponentAdapter:
    """Create an adapter of the specified type."""
    if adaptation_type == "bias_only":
        return BiasOnlyAdapter(base_model)
    elif adaptation_type == "scale_offset":
        return ScaleOffsetAdapter(base_model)
    elif adaptation_type.startswith("low_rank"):
        return LowRankAdapter(base_model, rank=rank)
    elif adaptation_type == "full_retrain":
        if global_z_t is None:
            raise ValueError("full_retrain requires global data")
        return FullRetrainAdapter(base_model, global_z_t, global_a_t, global_z_next)
    elif adaptation_type == "none":
        return BiasOnlyAdapter(base_model)  # no-op adapter
    else:
        raise ValueError(f"Unknown adaptation type: {adaptation_type}")
