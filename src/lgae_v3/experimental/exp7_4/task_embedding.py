"""Task embedding for exp7.4.

A frozen semantic embedding of task text. Not trainable end-to-end
initially — just a fixed representation that captures task semantics
well enough to distinguish task types without labels.

Uses a combination of:
  - Bag-of-words TF-IDF style features
  - Keyword category scores
  - Structural features (length, sentence count, etc.)

The embedding is deterministic and frozen — same input always
produces the same embedding. No training required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import hashlib
import re

from ..exp7_3.task_features import extract_features, TaskFeatures


# Keyword categories for semantic embedding.
CATEGORIES = {
    "factual": ["what", "who", "when", "where", "capital", "name", "define", "list"],
    "research": ["research", "synthesize", "sources", "gather", "analyze", "summary", "comprehensive", "information", "topic"],
    "coding": ["code", "function", "bug", "debug", "snippet", "program", "script", "compile", "syntax", "fix", "error"],
    "reasoning": ["solve", "reason", "logical", "steps", "multi-step", "derive", "infer", "deduce", "calculate", "prove"],
    "verification": ["verify", "check", "validate", "confirm", "assumption", "evidence", "proof", "test", "correct"],
    "memory": ["recall", "memory", "context", "previous", "stored", "remember", "history", "past"],
}


def _tokenize(text: str) -> list[str]:
    """Simple tokenization."""
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    return tokens


def _category_scores(tokens: list[str]) -> dict[str, float]:
    """Compute category scores for a tokenized text."""
    token_set = set(tokens)
    scores = {}
    for category, keywords in CATEGORIES.items():
        overlap = len(token_set & set(keywords))
        # Normalize by category size.
        scores[category] = overlap / max(len(keywords), 1)
    return scores


@dataclass
class TaskEmbedding:
    """Frozen semantic embedding of a task."""
    # Raw features
    manual_features: TaskFeatures = field(default_factory=TaskFeatures)
    # Category scores (6 dimensions)
    category_scores: dict[str, float] = field(default_factory=dict)
    # Dense embedding vector (16 dimensions)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float32))

    @property
    def vector(self) -> np.ndarray:
        """Full representation: manual features + embedding."""
        manual = np.array(self.manual_features.to_vector(), dtype=np.float32)
        return np.concatenate([manual, self.embedding])

    @property
    def dim(self) -> int:
        return len(self.vector)

    def to_dict(self) -> dict:
        return {
            "manual_features": self.manual_features.to_dict(),
            "category_scores": {k: round(v, 4) for k, v in self.category_scores.items()},
            "embedding_dim": len(self.embedding),
            "vector_dim": self.dim,
        }


def embed_task(task_input: str) -> TaskEmbedding:
    """Create a frozen semantic embedding of a task.

    The embedding combines:
    1. Manual structural features (from exp7.3)
    2. Category scores (keyword-based semantic dimensions)
    3. Hash-based dense features (for discriminability)
    """
    # Manual features.
    manual = extract_features(task_input)

    # Category scores.
    tokens = _tokenize(task_input)
    cat_scores = _category_scores(tokens)

    # Dense embedding: combine category scores with hash features.
    # This gives a 16-dimensional embedding.
    embedding = np.zeros(16, dtype=np.float32)

    # First 6 dimensions: category scores.
    categories = sorted(CATEGORIES.keys())
    for i, cat in enumerate(categories):
        embedding[i] = cat_scores.get(cat, 0.0)

    # Next 4 dimensions: structural features.
    embedding[6] = min(1.0, manual.n_tokens / 30.0)
    embedding[7] = manual.complexity_score
    embedding[8] = manual.estimated_difficulty
    embedding[9] = manual.avg_word_length / 10.0

    # Last 6 dimensions: hash-based features for discriminability.
    # These help distinguish tasks that have similar category scores
    # but different specific content.
    for i in range(6):
        hash_input = f"{task_input}:{i}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        embedding[10 + i] = (hash_val % 1000) / 1000.0

    return TaskEmbedding(
        manual_features=manual,
        category_scores=cat_scores,
        embedding=embedding,
    )


def embed_batch(task_inputs: list[str]) -> np.ndarray:
    """Embed a batch of tasks into a matrix."""
    embeddings = [embed_task(t).vector for t in task_inputs]
    return np.array(embeddings, dtype=np.float32)


def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors."""
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(np.dot(e1, e2) / (norm1 * norm2))


def nearest_neighbors(
    query: np.ndarray,
    candidates: np.ndarray,
    k: int = 5,
) -> list[tuple[int, float]]:
    """Find k nearest neighbors of query in candidates.

    Returns list of (index, similarity) pairs, sorted by similarity.
    """
    if len(candidates) == 0:
        return []
    sims = []
    for i, c in enumerate(candidates):
        sim = cosine_similarity(query, c)
        sims.append((i, sim))
    sims.sort(key=lambda x: -x[1])
    return sims[:k]
