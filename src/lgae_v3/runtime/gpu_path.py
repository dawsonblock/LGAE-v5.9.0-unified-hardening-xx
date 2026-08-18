"""GPU acceleration path (Phase 35).

Provides GPU-aware versions of hot-path operations. The runtime
automatically uses GPU when available and falls back to CPU when not.

Key operations accelerated:
  - batched message passing (sparse matmul)
  - batched feature computation
  - batched candidate scoring

The GPU path is opt-in: set ``device='cuda'`` or ``device='mps'`` (Apple
Silicon) to enable. All operations produce identical results to CPU;
the GPU path is purely a performance optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .graph_ops import build_adjacency_matrix


def get_device(device: str = "auto") -> torch.device:
    """Get the best available device.

    'auto' selects CUDA if available, then MPS (Apple Silicon), then CPU.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """Configuration for device placement."""
    device: torch.device
    pin_memory: bool = False
    async_copy: bool = True

    def to_log(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "pin_memory": bool(self.pin_memory),
            "async_copy": bool(self.async_copy),
        }


def move_to_device(tensor: Tensor, device: torch.device) -> Tensor:
    """Move a tensor to the specified device."""
    if tensor.device == device:
        return tensor
    return tensor.to(device)


def batched_message_passing(
    features: Tensor,  # [N, d] node features
    edge_index: Tensor,  # [2, E] edge index
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Batched message passing on GPU.

    Computes A @ features where A is the adjacency matrix.
    This is the core operation of graph neural networks.
    """
    if device is not None:
        features = move_to_device(features, device)
        edge_index = move_to_device(edge_index, device)
    n = features.shape[0]
    adj = build_adjacency_matrix(edge_index, n).to(features.device)
    # Normalize by degree (symmetric normalization).
    degrees = adj.sum(dim=1, keepdim=True)
    degrees = torch.clamp(degrees, min=1.0)
    adj_norm = adj / torch.sqrt(degrees * degrees.T)
    return adj_norm @ features


def batched_candidate_scoring(
    candidate_features: Tensor,  # [B, d] features for B candidates
    weights: Tensor,  # [d] scoring weights
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Score B candidates in parallel on GPU.

    Returns [B] tensor of scores.
    """
    if device is not None:
        candidate_features = move_to_device(candidate_features, device)
        weights = move_to_device(weights, device)
    return (candidate_features @ weights).squeeze(-1)


def batched_feature_computation(
    edge_index: Tensor,  # [2, E]
    node_features: Tensor,  # [N, d]
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Compute aggregated features for all nodes in parallel.

    For each node, computes the mean of its neighbors' features.
    """
    if device is not None:
        edge_index = move_to_device(edge_index, device)
        node_features = move_to_device(node_features, device)
    n = node_features.shape[0]
    adj = build_adjacency_matrix(edge_index, n).to(node_features.device)
    degrees = adj.sum(dim=1, keepdim=True)
    degrees = torch.clamp(degrees, min=1.0)
    # Mean aggregation.
    return (adj @ node_features) / degrees


def is_gpu_available() -> bool:
    """Check if any GPU device is available."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def device_info() -> dict[str, Any]:
    """Get information about available devices."""
    return {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": str(get_device("auto")),
    }
