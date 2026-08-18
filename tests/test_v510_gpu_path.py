"""v5.10 Phase 35: GPU acceleration path tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3.runtime import (
    get_device, DeviceConfig, move_to_device, batched_message_passing,
    batched_candidate_scoring, batched_feature_computation,
    is_gpu_available, device_info,
)


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(edges, dtype=torch.long).T


def test_get_device_auto():
    dev = get_device("auto")
    assert dev.type in ("cuda", "mps", "cpu")


def test_get_device_cpu():
    dev = get_device("cpu")
    assert dev.type == "cpu"


def test_device_config_to_log():
    config = DeviceConfig(device=torch.device("cpu"))
    log = config.to_log()
    assert log["device"] == "cpu"
    assert log["pin_memory"] is False


def test_move_to_device_same_device():
    t = torch.zeros(3)
    result = move_to_device(t, torch.device("cpu"))
    assert result is t  # no copy needed


def test_move_to_device_different_device():
    t = torch.zeros(3)
    result = move_to_device(t, torch.device("cpu"))
    assert result.device.type == "cpu"


def test_batched_message_passing():
    ei = _edge_index([(0, 1), (1, 2), (2, 0)])
    features = torch.tensor([[1.0], [2.0], [3.0]])
    result = batched_message_passing(features, ei)
    assert result.shape == (3, 1)


def test_batched_message_passing_with_device():
    ei = _edge_index([(0, 1), (1, 2)])
    features = torch.tensor([[1.0], [2.0], [3.0]])
    result = batched_message_passing(features, ei, device=torch.device("cpu"))
    assert result.shape == (3, 1)


def test_batched_candidate_scoring():
    candidates = torch.tensor([[1.0, 0.5], [0.3, 0.8], [0.9, 0.1]])
    weights = torch.tensor([0.6, 0.4])
    scores = batched_candidate_scoring(candidates, weights)
    assert scores.shape == torch.Size([3])


def test_batched_feature_computation():
    ei = _edge_index([(0, 1), (1, 2)])
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = batched_feature_computation(ei, features)
    assert result.shape == (3, 2)


def test_is_gpu_available():
    # Just check it doesn't crash.
    assert isinstance(is_gpu_available(), bool)


def test_device_info():
    info = device_info()
    assert "cuda_available" in info
    assert "mps_available" in info
    assert "current_device" in info
