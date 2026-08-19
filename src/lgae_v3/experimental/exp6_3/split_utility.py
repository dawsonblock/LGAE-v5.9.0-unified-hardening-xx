"""Split utility: exact additive + learned non-additive bonus.

The key architectural fix from the exp6.3 information leakage audit:

    U(G) = U_additive(G) + U_bonus(G)

Where:
    U_additive(G) = -sum(w * ||z_u - z_v||^2)    [exact, analytical, O(E)]
    U_bonus(G) = lambda * max(0, threshold + 1 - n_components)  [non-additive]

During search:
    Q_hat(S, a) = delta_U_additive(S, a) + gamma * V_bonus_hat(S')

The exact utility_fn is used ONLY for:
- Training label generation (exact enumeration)
- Finalist replay and verification
- NOT for beam search scoring

This is the true separation between known mathematics and learned prediction.
"""
from __future__ import annotations

from typing import Any, Callable
import numpy as np
import torch

from ...types import GraphBuffers
from ...runtime.analytical_utility import AnalyticalUtilityOracle
from .delayed_tasks import _count_components


def compute_additive_utility(graph: GraphBuffers, z: torch.Tensor) -> float:
    """Exact additive utility: U = -sum(w * ||z_u - z_v||^2).

    This is O(E) and can be computed exactly without knowing
    global graph structure.
    """
    with torch.no_grad():
        src = graph.src[graph.valid]
        dst = graph.dst[graph.valid]
        w = graph.weight[graph.valid]
        if src.numel() == 0:
            return 0.0
        d = (z[src] - z[dst]).pow(2).sum(-1)
        return float(-(w * d).sum().item())


def compute_bonus(graph: GraphBuffers, z: torch.Tensor,
                  lambda_conn: float = 30.0, threshold: int = 1) -> float:
    """Non-additive bonus: lambda * max(0, threshold + 1 - n_components).

    This requires knowing the GLOBAL graph structure (component count).
    It CANNOT be computed from local edge deltas alone.
    This is the quantity that must be LEARNED during search.
    """
    n = int(graph.num_nodes)
    n_comp = _count_components(graph, n)
    bonus = max(0, threshold + 1 - n_comp)
    return lambda_conn * bonus


def compute_total_utility(graph: GraphBuffers, z: torch.Tensor,
                          lambda_conn: float = 30.0, threshold: int = 1) -> float:
    """Total utility = additive + bonus. Used for exact ground truth only."""
    u_add = compute_additive_utility(graph, z)
    u_bonus = compute_bonus(graph, z, lambda_conn, threshold)
    return u_add + u_bonus


def make_total_utility_fn(lambda_conn: float = 30.0, threshold: int = 1) -> Callable:
    """Create a total utility function for exact MPC ground truth."""
    return lambda g, z: compute_total_utility(g, z, lambda_conn, threshold)


# ---------------------------------------------------------------------------
# Bonus prediction model
# ---------------------------------------------------------------------------

class BonusPredictor:
    """Predicts the non-additive connectivity bonus from graph features.

    This is the TRUE learned component. It predicts:
        bonus(S') = lambda * max(0, threshold + 1 - n_components(S'))

    from structural features of S', WITHOUT computing n_components.

    Features:
    - density
    - degree statistics
    - additive utility (proxy for edge quality)
    - n_edges / n_nodes ratio

    The model must learn that bridging components reduces n_components,
    which increases the bonus. It cannot compute this directly.
    """

    def __init__(self, lambda_conn: float = 30.0, threshold: int = 1) -> None:
        self.lambda_conn = lambda_conn
        self.threshold = threshold
        self._model: Any = None
        self._fitted = False

    def extract_bonus_features(self, graph: GraphBuffers, z: torch.Tensor) -> np.ndarray:
        """Extract features for bonus prediction (no component counting)."""
        n = int(graph.num_nodes)
        valid = graph.valid.bool()
        n_edges = int(valid.sum().item())
        density = n_edges / max(n * (n - 1) / 2, 1)

        degrees = np.zeros(n)
        for i in range(graph.src.shape[0]):
            if valid[i]:
                s = int(graph.src[i].item())
                d = int(graph.dst[i].item())
                if s < n: degrees[s] += 1
                if d < n: degrees[d] += 1

        u_add = compute_additive_utility(graph, z)

        # Number of zero-degree nodes (proxy for isolated components).
        n_isolated = int(np.sum(degrees == 0))

        # Edge-to-node ratio (proxy for connectivity).
        edge_ratio = n_edges / max(n, 1)

        return np.array([
            n / 50.0, density, float(np.mean(degrees)) / 10.0,
            float(np.std(degrees)) / 10.0, float(np.max(degrees)) / 20.0,
            u_add / 100.0, n_isolated / max(n, 1), edge_ratio,
        ])

    def fit(self, graphs: list[GraphBuffers], z_list: list[torch.Tensor]) -> None:
        """Train on (graph, exact_bonus) pairs from exact enumeration."""
        from sklearn.linear_model import Ridge
        X = np.array([self.extract_bonus_features(g, z) for g, z in zip(graphs, z_list)])
        y = np.array([compute_bonus(g, z, self.lambda_conn, self.threshold)
                      for g, z in zip(graphs, z_list)])
        self._model = Ridge(alpha=1.0)
        self._model.fit(X, y)
        self._fitted = True

    def fit_from_records(self, records: list[dict]) -> None:
        """Train from value dataset records."""
        from sklearn.linear_model import Ridge
        X = np.array([r["state_features"] for r in records])
        y = np.array([r["exact_bonus"] for r in records])
        self._model = Ridge(alpha=1.0)
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, graph: GraphBuffers, z: torch.Tensor) -> float:
        """Predict bonus without computing n_components."""
        if not self._fitted:
            return 0.0
        x = self.extract_bonus_features(graph, z).reshape(1, -1)
        return float(self._model.predict(x)[0])

    @property
    def name(self) -> str:
        return "BonusPredictor_Ridge"


class ZeroBonusPredictor(BonusPredictor):
    """Always predicts zero bonus. This is the greedy baseline."""

    def predict(self, graph: GraphBuffers, z: torch.Tensor) -> float:
        return 0.0

    @property
    def name(self) -> str:
        return "ZeroBonus"
