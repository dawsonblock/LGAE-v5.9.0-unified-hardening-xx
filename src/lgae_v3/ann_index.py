"""ANN-backed geometric neighbor index.

v5.1.1 makes this module conform to :class:`NeighborIndex` and fixes self-neighbor
and padding semantics. The pure-NumPy fallback is explicitly a random-projection
candidate index; it is not mislabeled as HNSW.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .neighbor_index import NeighborIndex, KNNGraphResult, recall_at_k


class FAISSIndex:
    def __init__(self, dim: int, nlist: int = 100, nprobe: int = 10):
        try:
            import faiss
        except ImportError as e:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu") from e
        self.faiss = faiss
        self.dim = int(dim)
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.index = None
        self._data: np.ndarray | None = None

    def build(self, data: np.ndarray) -> None:
        data = np.ascontiguousarray(data.astype(np.float32, copy=False))
        N = data.shape[0]
        if N < 1000:
            self.index = self.faiss.IndexFlatL2(self.dim)
        else:
            nlist = min(self.nlist, max(1, N // 20))
            quantizer = self.faiss.IndexFlatL2(self.dim)
            self.index = self.faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            self.index.train(data)
            self.index.nprobe = min(self.nprobe, nlist)
        self.index.add(data)
        self._data = data.copy()

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")
        return self.index.search(np.ascontiguousarray(query.astype(np.float32, copy=False)), int(k))

    def refresh(self, data: np.ndarray) -> None:
        self.build(data)


class RandomProjectionANN:
    """Dependency-free random-projection candidate index.

    This is a coarse partition ANN backend, not HNSW. It partitions the latent
    cloud by random projections and exact-reranks candidates within the best
    few partitions.
    """
    def __init__(self, dim: int, n_partitions: int = 16, max_leaf_size: int = 50):
        self.dim = int(dim)
        self.n_partitions = int(n_partitions)
        self.max_leaf_size = int(max_leaf_size)
        self._data: np.ndarray | None = None
        self._partitions: list[np.ndarray] = []
        self._projection: np.ndarray | None = None

    def build(self, data: np.ndarray) -> None:
        data = np.asarray(data)
        N = data.shape[0]
        self._data = data.copy()
        self._projection = None
        if N <= self.max_leaf_size:
            self._partitions = [np.arange(N)]
            return
        rng = np.random.RandomState(42)
        self._projection = rng.randn(self.dim, self.n_partitions).astype(data.dtype)
        projections = data @ self._projection
        best_partition = np.argmax(projections, axis=1)
        self._partitions = []
        # Preserve partition IDs so query projection indices stay meaningful.
        for p in range(self.n_partitions):
            self._partitions.append(np.where(best_partition == p)[0])

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        if self._data is None:
            raise RuntimeError("Index not built. Call build() first.")
        Nq = query.shape[0]
        all_distances = np.full((Nq, k), np.inf, dtype=np.float32)
        all_indices = np.full((Nq, k), -1, dtype=np.int64)
        for i in range(Nq):
            q = query[i]
            if self._projection is not None:
                q_proj = q @ self._projection
                top_partitions = np.argsort(-q_proj)[: min(4, self.n_partitions)]
            else:
                top_partitions = np.array([0])
            valid = [self._partitions[int(p)] for p in top_partitions if len(self._partitions[int(p)])]
            candidates = np.unique(np.concatenate(valid)) if valid else np.arange(len(self._data))
            diff = self._data[candidates] - q
            dists = np.sum(diff * diff, axis=1)
            k_actual = min(int(k), len(candidates))
            if k_actual == 0:
                continue
            top_idx = np.argpartition(dists, k_actual - 1)[:k_actual]
            order = np.argsort(dists[top_idx])
            all_distances[i, :k_actual] = dists[top_idx][order]
            all_indices[i, :k_actual] = candidates[top_idx][order]
        return all_distances, all_indices

    def refresh(self, data: np.ndarray) -> None:
        self.build(data)


# Backward-compatible import name; documentation no longer calls this HNSW.
HNSWIndexNumpy = RandomProjectionANN


class ANNNeighborIndex(NeighborIndex):
    """Approximate candidate search followed by exact metric reranking."""
    def __init__(
        self,
        dim: int,
        n_candidates: int = 96,
        n_final: int = 32,
        backend: str = "auto",
        refresh_interval: int = 100,
        drift_threshold: float = 0.1,
    ):
        self.dim = int(dim)
        self.n_candidates = int(n_candidates)
        self.n_final = int(n_final)
        self.backend = backend
        self.refresh_interval = int(refresh_interval)
        self.drift_threshold = float(drift_threshold)
        self._index: Any | None = None
        self._step = 0
        self._data: np.ndarray | None = None
        self._z_ref: Tensor | None = None
        self._cache_generation = 0
        self._cache_dirty = True
        self._bound_graph_version: int | None = None
        self._dirty_reason: str | None = "unbuilt"

    def _select_backend(self) -> Any:
        if self.backend == "faiss":
            return FAISSIndex(self.dim)
        if self.backend in {"numpy", "random_projection"}:
            return RandomProjectionANN(self.dim)
        if self.backend == "auto":
            try:
                return FAISSIndex(self.dim)
            except ImportError:
                return RandomProjectionANN(self.dim)
        raise ValueError(f"Unknown backend: {self.backend}")

    def build(self, z: Tensor) -> None:
        data = z.detach().cpu().numpy().astype(np.float32)
        self._data = data.copy()
        self._z_ref = z.detach().cpu().clone()
        self._index = self._select_backend()
        self._index.build(data)
        self._step = 0
        self._cache_generation += 1
        self._cache_dirty = False
        self._dirty_reason = None

    def refresh(self, z: Tensor) -> None:
        self.build(z)

    def should_rebuild(self, z: Tensor, drift_threshold: float | None = None) -> bool:
        if self.cache_dirty or self._index is None or self._z_ref is None:
            return True
        threshold = self.drift_threshold if drift_threshold is None else float(drift_threshold)
        z_cpu = z.detach().cpu()
        if z_cpu.shape != self._z_ref.shape:
            return True
        denom = float(torch.linalg.vector_norm(self._z_ref).item())
        drift = float(torch.linalg.vector_norm(z_cpu - self._z_ref).item()) / max(denom, 1e-10)
        return drift > threshold or (self.refresh_interval > 0 and self._step >= self.refresh_interval)

    def search(self, z: Tensor, k: int | None = None) -> tuple[Tensor, Tensor]:
        """Backward-compatible return order: ``(distances, indices)``."""
        if self.should_rebuild(z):
            self.refresh(z)
        self._step += 1

        k_final = int(k or self.n_final)
        N = int(z.shape[0])
        if N <= 1 or k_final <= 0:
            return torch.empty((N, 0)), torch.empty((N, 0), dtype=torch.long)
        k_final = min(k_final, N - 1)
        # Ask for extra candidates so removing self does not consume one of k.
        k_search = min(N, max(k_final + 1, self.n_candidates + 1))
        data = z.detach().cpu().numpy().astype(np.float32)
        _, indices = self._index.search(data, k_search)

        final_distances = np.full((N, k_final), np.inf, dtype=np.float32)
        final_indices = np.full((N, k_final), -1, dtype=np.int64)
        for i in range(N):
            cand = indices[i]
            valid = cand >= 0
            cand = cand[valid]
            # When querying the indexed cloud itself, remove the exact self index.
            cand = cand[cand != i]
            if cand.size == 0:
                continue
            cand = np.unique(cand)
            diff = data[cand] - data[i]
            exact = np.sum(diff * diff, axis=1)
            k_valid = min(k_final, len(cand))
            top = np.argpartition(exact, k_valid - 1)[:k_valid]
            order = np.argsort(exact[top])
            final_distances[i, :k_valid] = exact[top][order]
            final_indices[i, :k_valid] = cand[top][order]

        return torch.from_numpy(final_distances), torch.from_numpy(final_indices)

    def query(self, z: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """NeighborIndex protocol return order: ``(indices, distances)``."""
        distances, indices = self.search(z, k)
        return indices, distances.sqrt()  # protocol uses Euclidean distance

    def build_knn_graph(self, z: Tensor, k: int | None = None, threshold: float | None = None) -> KNNGraphResult:
        k = min(int(k or self.n_final), max(0, int(z.shape[0]) - 1))
        distances, indices = self.search(z, k)
        N = z.shape[0]
        src_list: list[int] = []
        dst_list: list[int] = []
        weight_list: list[float] = []
        for i in range(N):
            for j_idx in range(k):
                j = int(indices[i, j_idx])
                d = float(distances[i, j_idx])
                if j < 0 or j == i or not np.isfinite(d):
                    continue
                if threshold is not None and d > threshold:
                    continue
                src_list.append(i)
                dst_list.append(j)
                weight_list.append(1.0 / (1.0 + d))
        if not src_list:
            return KNNGraphResult(torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), torch.zeros(0), N)
        return KNNGraphResult(
            torch.tensor(src_list, dtype=torch.long),
            torch.tensor(dst_list, dtype=torch.long),
            torch.tensor(weight_list, dtype=torch.float32),
            N,
        )

    def measure_recall(self, z: Tensor, k: int = 10, exact_index: NeighborIndex | None = None) -> float:
        if exact_index is None:
            from .neighbor_index import ExactChunkedKNN
            exact_index = ExactChunkedKNN()
            exact_index.build(z)
        exact_indices, _ = exact_index.query(z, k)
        ann_indices, _ = self.query(z, k)
        return recall_at_k(ann_indices, exact_indices, k)
