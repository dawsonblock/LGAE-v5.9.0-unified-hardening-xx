"""Representation collision analysis.

Different structural situations should not collapse excessively into
identical encodings. Generate thousands of transitions and measure:

- exact representation collisions
- near-neighbor collision rate
- same representation / different outcome variance

If the handcrafted encoder maps very different graph states to the same
vector, that identifies where graph-native representations become justified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import math
import numpy as np
from collections import Counter

from .protocol import StateActionRepresentation


@dataclass(slots=True)
class CollisionReport:
    """Report on representation collisions."""
    encoder_id: str
    n_representations: int
    n_unique: int
    n_collisions: int
    collision_rate: float
    max_collision_group_size: int
    mean_outcome_variance_in_collision_groups: float
    near_neighbor_rate: float
    collision_groups: list[dict[str, Any]] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "n_representations": int(self.n_representations),
            "n_unique": int(self.n_unique),
            "n_collisions": int(self.n_collisions),
            "collision_rate": float(self.collision_rate),
            "max_collision_group_size": int(self.max_collision_group_size),
            "mean_outcome_variance_in_collision_groups": float(self.mean_outcome_variance_in_collision_groups),
            "near_neighbor_rate": float(self.near_neighbor_rate),
            "n_collision_groups": len(self.collision_groups),
            "collision_groups": self.collision_groups[:20],  # top 20
        }


def analyze_collisions(
    representations: list[StateActionRepresentation],
    outcomes: list[float],
    *,
    near_neighbor_threshold: float = 1e-4,
) -> CollisionReport:
    """Analyze representation collisions.

    Args:
        representations: List of StateActionRepresentation.
        outcomes: Realized outcomes (e.g., ΔU) for each representation.
        near_neighbor_threshold: L2 distance threshold for near-neighbor.

    Returns:
        CollisionReport with collision statistics.
    """
    n = len(representations)
    if n == 0:
        return CollisionReport(
            encoder_id="", n_representations=0, n_unique=0,
            n_collisions=0, collision_rate=0.0,
            max_collision_group_size=0,
            mean_outcome_variance_in_collision_groups=0.0,
            near_neighbor_rate=0.0,
        )

    encoder_id = representations[0].encoder_id if representations else ""

    # Exact collisions: representations with identical vectors.
    vec_strings = [",".join(f"{v:.10f}" for v in r.vector) for r in representations]
    counter = Counter(vec_strings)

    n_unique = len(counter)
    n_collisions = sum(count - 1 for count in counter.values() if count > 1)
    collision_rate = n_collisions / n if n > 0 else 0.0
    max_group_size = max(counter.values()) if counter else 0

    # Outcome variance within collision groups.
    group_outcomes: dict[str, list[float]] = {}
    for vec_str, outcome in zip(vec_strings, outcomes):
        if vec_str not in group_outcomes:
            group_outcomes[vec_str] = []
        group_outcomes[vec_str].append(outcome)

    group_variances = []
    collision_groups_info = []
    for vec_str, outs in group_outcomes.items():
        if len(outs) > 1:
            var = float(np.var(outs))
            group_variances.append(var)
            if len(collision_groups_info) < 20:
                collision_groups_info.append({
                    "group_size": len(outs),
                    "outcome_variance": var,
                    "outcome_mean": float(np.mean(outs)),
                })

    mean_var = float(np.mean(group_variances)) if group_variances else 0.0

    # Near-neighbor rate: fraction of pairs within threshold.
    near_neighbor_rate = 0.0
    if n > 1 and n <= 5000:  # limit for O(n²) computation
        X = np.array([list(r.vector) for r in representations], dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        # Sample pairs for speed.
        n_samples = min(1000, n * (n - 1) // 2)
        rng = np.random.RandomState(42)
        close_count = 0
        for _ in range(n_samples):
            i, j = rng.randint(0, n, 2)
            if i != j:
                dist = np.linalg.norm(X[i] - X[j])
                if dist < near_neighbor_threshold:
                    close_count += 1
        near_neighbor_rate = close_count / n_samples

    return CollisionReport(
        encoder_id=encoder_id,
        n_representations=n,
        n_unique=n_unique,
        n_collisions=n_collisions,
        collision_rate=collision_rate,
        max_collision_group_size=max_group_size,
        mean_outcome_variance_in_collision_groups=mean_var,
        near_neighbor_rate=near_neighbor_rate,
        collision_groups=collision_groups_info,
    )
