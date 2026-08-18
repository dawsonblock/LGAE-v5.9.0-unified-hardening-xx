"""Lightweight probe models for representation informativeness benchmark.

For each encoder, train identical lightweight predictors for:
- sign(ΔU) — logistic regression
- mutation success — logistic regression
- risk bucket — logistic regression
- relative candidate rank — linear regression on rank

Use simple models (logistic/linear regression) to test the representation,
not model capacity.

Example matrix:

    Encoder          | ΔU sign | Candidate rank | Risk  | Dim
    -----------------+---------+----------------+-------+-----
    Minimal          | baseline| baseline       | base  | ~5
    Global           | …       | …              | …     | 24
    Global+Local     | …       | …              | …     | 36
    Geometry         | …       | …              | …     | …
    Spectral         | …       | …              | …     | …
    Learned graph    | …       | …              | …     | 64
    Hybrid           | …       | …              | …     | …
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import math
import numpy as np

from .protocol import StateActionRepresentation


@dataclass(slots=True)
class ProbeResult:
    """Result of a single probe model on a single task."""
    encoder_id: str
    task: str
    metric: str  # "accuracy", "mse", "spearman"
    value: float
    n_samples: int
    n_features: int
    baseline_value: float = 0.0  # baseline (e.g., majority class accuracy)

    @property
    def gain_over_baseline(self) -> float:
        return self.value - self.baseline_value

    def to_log(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "task": self.task,
            "metric": self.metric,
            "value": float(self.value),
            "n_samples": int(self.n_samples),
            "n_features": int(self.n_features),
            "baseline_value": float(self.baseline_value),
            "gain_over_baseline": float(self.gain_over_baseline),
        }


@dataclass(slots=True)
class EncoderProbeReport:
    """Probe results for a single encoder across all tasks."""
    encoder_id: str
    encoder_dimension: int
    results: list[ProbeResult] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "encoder_dimension": int(self.encoder_dimension),
            "results": [r.to_log() for r in self.results],
        }


class LogisticProbe:
    """Simple logistic regression probe using numpy.

    No sklearn dependency — uses gradient descent on logistic loss.
    """

    def __init__(self, lr: float = 0.01, n_epochs: int = 100, seed: int = 42) -> None:
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, d = X.shape
        rng = np.random.RandomState(self.seed)
        self.weights = rng.randn(d) * 0.01
        self.bias = 0.0
        for _ in range(self.n_epochs):
            logits = X @ self.weights + self.bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            grad = (probs - y) / n
            self.weights -= self.lr * (X.T @ grad)
            self.bias -= self.lr * float(np.mean(grad))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.zeros(len(X))
        logits = X @ self.weights + self.bias
        return (1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30))) > 0.5).astype(float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.zeros(len(X))
        logits = X @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


class LinearProbe:
    """Simple linear regression probe using numpy."""

    def __init__(self, lr: float = 0.01, n_epochs: int = 100, seed: int = 42) -> None:
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, d = X.shape
        rng = np.random.RandomState(self.seed)
        self.weights = rng.randn(d) * 0.01
        self.bias = 0.0
        for _ in range(self.n_epochs):
            pred = X @ self.weights + self.bias
            grad = (pred - y) / n
            self.weights -= self.lr * (X.T @ grad)
            self.bias -= self.lr * float(np.mean(grad))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.zeros(len(X))
        return X @ self.weights + self.bias


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation."""
    if len(x) < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-10 or ry.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def run_probe_benchmark(
    representations: list[StateActionRepresentation],
    targets_delta_u: list[float],
    targets_success: list[bool],
    targets_risk: list[float],
    encoder_id: str = "",
) -> EncoderProbeReport:
    """Run all probe tasks on a set of representations.

    Args:
        representations: List of StateActionRepresentation from one encoder.
        targets_delta_u: Realized ΔU for each transition.
        targets_success: Success flag for each transition.
        targets_risk: Realized risk for each transition.
        encoder_id: Encoder identifier.

    Returns:
        EncoderProbeReport with results for all tasks.
    """
    if not representations:
        return EncoderProbeReport(encoder_id=encoder_id, encoder_dimension=0)

    X = np.array([list(r.vector) for r in representations], dtype=np.float64)
    # Replace nonfinite.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n, d = X.shape

    results: list[ProbeResult] = []

    # Task 1: sign(ΔU) — logistic regression.
    y_sign = np.array([1.0 if du > 0 else 0.0 for du in targets_delta_u], dtype=np.float64)
    if n >= 10 and len(set(y_sign)) > 1:
        probe = LogisticProbe()
        probe.fit(X, y_sign)
        preds = probe.predict(X)
        acc = float(np.mean(preds == y_sign))
        baseline = float(max(np.mean(y_sign), 1 - np.mean(y_sign)))
        results.append(ProbeResult(
            encoder_id=encoder_id, task="sign_delta_u",
            metric="accuracy", value=acc, n_samples=n, n_features=d,
            baseline_value=baseline,
        ))

    # Task 2: mutation success — logistic regression.
    y_success = np.array([1.0 if s else 0.0 for s in targets_success], dtype=np.float64)
    if n >= 10 and len(set(y_success)) > 1:
        probe = LogisticProbe()
        probe.fit(X, y_success)
        preds = probe.predict(X)
        acc = float(np.mean(preds == y_success))
        baseline = float(max(np.mean(y_success), 1 - np.mean(y_success)))
        results.append(ProbeResult(
            encoder_id=encoder_id, task="mutation_success",
            metric="accuracy", value=acc, n_samples=n, n_features=d,
            baseline_value=baseline,
        ))

    # Task 3: risk bucket (high risk vs low risk) — logistic regression.
    median_risk = float(np.median(targets_risk)) if targets_risk else 0.0
    y_risk = np.array([1.0 if r > median_risk else 0.0 for r in targets_risk], dtype=np.float64)
    if n >= 10 and len(set(y_risk)) > 1:
        probe = LogisticProbe()
        probe.fit(X, y_risk)
        preds = probe.predict(X)
        acc = float(np.mean(preds == y_risk))
        baseline = float(max(np.mean(y_risk), 1 - np.mean(y_risk)))
        results.append(ProbeResult(
            encoder_id=encoder_id, task="risk_bucket",
            metric="accuracy", value=acc, n_samples=n, n_features=d,
            baseline_value=baseline,
        ))

    # Task 4: relative candidate rank — linear regression with Spearman.
    y_delta = np.array(targets_delta_u, dtype=np.float64)
    if n >= 10 and y_delta.std() > 1e-10:
        probe = LinearProbe()
        probe.fit(X, y_delta)
        preds = probe.predict(X)
        spearman = _spearman_correlation(preds, y_delta)
        results.append(ProbeResult(
            encoder_id=encoder_id, task="candidate_rank",
            metric="spearman", value=spearman, n_samples=n, n_features=d,
            baseline_value=0.0,
        ))

    return EncoderProbeReport(
        encoder_id=encoder_id,
        encoder_dimension=d,
        results=results,
    )
