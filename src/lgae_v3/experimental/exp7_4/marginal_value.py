"""Per-node marginal value estimator for exp7.4.

For each optional node (Researcher, Critic, Verifier, Memory),
estimate the marginal value of including that node:

  ΔJ_n = J(with node n) - J(without node n)

This is learned from execution history. The estimator uses
k-nearest-neighbor regression on task embeddings to predict
the marginal value of each node for a new task.

Training data comes from shadow executions:
  - Run a task with the node included → J_with
  - Run the same task without the node → J_without
  - ΔJ = J_with - J_without

Then for a new task:
  - Embed the task
  - Find k nearest neighbors in training data
  - Predict ΔJ_n as weighted average of neighbors' ΔJ_n
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from collections import defaultdict

from .task_embedding import embed_task, cosine_similarity, nearest_neighbors


# Optional nodes that can be included or excluded.
OPTIONAL_NODES = ["researcher", "critic", "verifier", "memory"]


@dataclass
class MarginalValueSample:
    """A single sample of marginal value for a node."""
    task_input: str
    node: str
    j_with: float
    j_without: float
    delta_j: float
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


class MarginalValueEstimator:
    """Estimates per-node marginal value from execution history.

    Uses k-nearest-neighbor regression on task embeddings.
    No task labels — only text embeddings and observed outcomes.
    """

    def __init__(
        self,
        k: int = 5,
        min_samples: int = 3,
    ) -> None:
        self.k = k
        self.min_samples = min_samples
        self.samples_by_node: dict[str, list[MarginalValueSample]] = defaultdict(list)

    def add_sample(
        self,
        task_input: str,
        node: str,
        j_with: float,
        j_without: float,
    ) -> None:
        """Add a marginal value sample."""
        delta = j_with - j_without
        emb = embed_task(task_input).vector
        sample = MarginalValueSample(
            task_input=task_input,
            node=node,
            j_with=j_with,
            j_without=j_without,
            delta_j=delta,
            embedding=emb,
        )
        self.samples_by_node[node].append(sample)

    def predict_marginal_value(
        self,
        task_input: str,
        node: str,
    ) -> tuple[float, float]:
        """Predict the marginal value of including a node.

        Returns (predicted_delta_j, confidence).
        Confidence is based on number of neighbors and their agreement.
        """
        samples = self.samples_by_node.get(node, [])
        if len(samples) < self.min_samples:
            # Not enough data — default to neutral (0.0) with low confidence.
            return 0.0, 0.0

        query_emb = embed_task(task_input).vector

        # Find k nearest neighbors.
        candidate_embs = np.array([s.embedding for s in samples])
        neighbors = nearest_neighbors(query_emb, candidate_embs, k=self.k)

        if not neighbors:
            return 0.0, 0.0

        # Weighted average by similarity.
        weights = []
        values = []
        for idx, sim in neighbors:
            # Clamp similarity to non-negative.
            w = max(0.0, sim)
            weights.append(w)
            values.append(samples[idx].delta_j)

        total_weight = sum(weights)
        if total_weight < 1e-10:
            # All neighbors have zero similarity — use unweighted mean.
            return float(np.mean(values)), 0.3

        predicted = sum(w * v for w, v in zip(weights, values)) / total_weight

        # Confidence: based on neighbor count and weight concentration.
        confidence = min(1.0, len(neighbors) / self.k) * min(1.0, total_weight / len(neighbors))

        return predicted, confidence

    def predict_all_nodes(
        self,
        task_input: str,
    ) -> dict[str, dict]:
        """Predict marginal value for all optional nodes.

        Returns {node: {"delta_j": float, "confidence": float, "include": bool}}
        """
        results = {}
        for node in OPTIONAL_NODES:
            delta, conf = self.predict_marginal_value(task_input, node)
            # Include node if predicted marginal value is positive.
            include = delta > 0.0
            results[node] = {
                "delta_j": round(delta, 4),
                "confidence": round(conf, 4),
                "include": include,
            }
        return results

    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            node: {
                "n_samples": len(samples),
                "mean_delta_j": round(float(np.mean([s.delta_j for s in samples])), 4) if samples else 0.0,
                "std_delta_j": round(float(np.std([s.delta_j for s in samples])), 4) if samples else 0.0,
            }
            for node, samples in self.samples_by_node.items()
        }

    def get_training_data(self) -> dict[str, list[dict]]:
        """Get all training data for inspection."""
        return {
            node: [
                {
                    "task_input": s.task_input[:100],
                    "delta_j": round(s.delta_j, 4),
                    "j_with": round(s.j_with, 4),
                    "j_without": round(s.j_without, 4),
                }
                for s in samples
            ]
            for node, samples in self.samples_by_node.items()
        }
