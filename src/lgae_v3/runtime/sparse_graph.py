"""Sparse-first graph representation (Phase 34).

The runtime's GraphBuffers uses dense tensors for edge storage. For large
sparse graphs (e.g. 10K nodes with 1K edges), this wastes memory and compute.

This module provides a sparse-first graph representation using CSR
(Compressed Sparse Row) format, which is memory-efficient and enables
fast neighbor lookups without building a full adjacency matrix.

CSR format:
  - row_ptr: [N+1] tensor, row_ptr[i] to row_ptr[i+1] is the range of neighbors
  - col_idx: [E] tensor, the neighbor indices
  - values: [E] tensor, optional edge weights

This is the standard format for sparse graph computation in scientific computing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class SparseGraph:
    """Sparse graph in CSR format."""
    row_ptr: Tensor  # [N+1] int64
    col_idx: Tensor  # [2E] int64 (both directions for undirected)
    values: Tensor | None = None  # [2E] float, optional edge weights
    num_nodes: int = 0

    @property
    def num_edges(self) -> int:
        return int(self.col_idx.shape[0]) // 2

    def neighbors(self, node: int) -> Tensor:
        """Get neighbors of a node. O(1) lookup."""
        start = int(self.row_ptr[node].item())
        end = int(self.row_ptr[node + 1].item())
        return self.col_idx[start:end]

    def degree(self, node: int) -> int:
        """Get degree of a node. O(1)."""
        start = int(self.row_ptr[node].item())
        end = int(self.row_ptr[node + 1].item())
        return end - start

    def to_log(self) -> dict[str, Any]:
        return {
            "num_nodes": int(self.num_nodes),
            "num_edges": int(self.num_edges),
            "row_ptr_size": int(self.row_ptr.shape[0]),
            "col_idx_size": int(self.col_idx.shape[0]),
            "has_values": self.values is not None,
        }


def build_sparse_graph(edge_index: Tensor, num_nodes: int) -> SparseGraph:
    """Build a SparseGraph from an edge index tensor.

    The edge index is [2, E] with src in row 0 and dst in row 1.
    The resulting CSR includes both directions (undirected).
    """
    if edge_index.numel() == 0:
        row_ptr = torch.zeros(num_nodes + 1, dtype=torch.long)
        col_idx = torch.zeros(0, dtype=torch.long)
        return SparseGraph(row_ptr=row_ptr, col_idx=col_idx, num_nodes=num_nodes)

    src = edge_index[0]
    dst = edge_index[1]
    # Add both directions.
    all_src = torch.cat([src, dst])
    all_dst = torch.cat([dst, src])
    # Sort by source node.
    order = torch.argsort(all_src)
    all_src = all_src[order]
    all_dst = all_dst[order]
    # Build row_ptr.
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    for s in all_src.tolist():
        row_ptr[s + 1] += 1
    row_ptr = torch.cumsum(row_ptr, dim=0)
    return SparseGraph(row_ptr=row_ptr, col_idx=all_dst, num_nodes=num_nodes)


def sparse_adjacency_matrix(sparse: SparseGraph) -> Tensor:
    """Build a sparse adjacency matrix from a SparseGraph.

    Returns a dense [N, N] tensor. For very large graphs, prefer using
    the SparseGraph directly.
    """
    n = sparse.num_nodes
    adj = torch.zeros(n, n, dtype=torch.float32)
    for i in range(n):
        neighbors = sparse.neighbors(i)
        for j in neighbors.tolist():
            adj[i, j] = 1.0
    return adj


def sparse_to_edge_index(sparse: SparseGraph) -> Tensor:
    """Convert a SparseGraph back to an edge index tensor [2, E]."""
    edges: list[tuple[int, int]] = []
    for i in range(sparse.num_nodes):
        neighbors = sparse.neighbors(i)
        for j in neighbors.tolist():
            if i < j:  # avoid duplicates
                edges.append((i, j))
    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).T
