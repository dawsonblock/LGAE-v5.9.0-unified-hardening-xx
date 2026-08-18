"""v5.10 Phase 16: Lie-group/manifold action tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3.runtime import (
    LieGroup, ManifoldAction, exponential_map, compose, inverse,
    make_so3_action, make_su2_action, make_gl_action,
)


def test_so3_identity_is_identity():
    action = make_so3_action([0.0, 0.0, 0.0])
    R = exponential_map(action)
    assert torch.allclose(R, torch.eye(3), atol=1e-5)


def test_so3_rotation_is_orthogonal():
    action = make_so3_action([0.1, 0.2, 0.3])
    R = exponential_map(action)
    # R^T R = I for orthogonal matrices.
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)


def test_so3_rotation_has_det_one():
    action = make_so3_action([0.5, -0.3, 0.2])
    R = exponential_map(action)
    det = torch.det(R)
    assert abs(det.item() - 1.0) < 1e-5


def test_so3_compose_with_inverse_is_identity():
    action = make_so3_action([0.1, 0.2, 0.3])
    R = exponential_map(action)
    R_inv = inverse(LieGroup.SO3, R)
    result = compose(LieGroup.SO3, R, R_inv)
    assert torch.allclose(result, torch.eye(3), atol=1e-5)


def test_su2_identity_is_identity_quaternion():
    action = make_su2_action([0.0, 0.0, 0.0])
    q = exponential_map(action)
    assert torch.allclose(q, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-5)


def test_su2_quaternion_has_unit_norm():
    action = make_su2_action([0.1, 0.2, 0.3])
    q = exponential_map(action)
    assert abs(q.norm().item() - 1.0) < 1e-5


def test_su2_compose_with_inverse_is_identity():
    action = make_su2_action([0.1, 0.2, 0.3])
    q = exponential_map(action)
    q_inv = inverse(LieGroup.SU2, q)
    result = compose(LieGroup.SU2, q, q_inv)
    assert torch.allclose(result, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-5)


def test_gl_identity_is_identity():
    A = torch.zeros(3, 3)
    action = make_gl_action(A, d=3)
    M = exponential_map(action)
    assert torch.allclose(M, torch.eye(3), atol=1e-5)


def test_gl_exponential_is_invertible():
    A = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]])
    action = make_gl_action(A, d=3)
    M = exponential_map(action)
    # exp(A) is always invertible.
    M_inv = inverse(LieGroup.GL, M)
    result = compose(LieGroup.GL, M, M_inv)
    assert torch.allclose(result, torch.eye(3), atol=1e-4)


def test_manifold_action_to_log():
    action = make_so3_action([0.1, 0.2, 0.3])
    log = action.to_log()
    assert log["group"] == "so3"
    assert log["dim"] == 3
    assert log["algebra_norm"] > 0


def test_so3_zero_rotation_is_identity():
    action = make_so3_action([0.0, 0.0, 0.0])
    R = exponential_map(action)
    assert torch.allclose(R, torch.eye(3), atol=1e-6)


def test_so3_2pi_rotation_is_identity():
    # Rotation by 2*pi around any axis should be identity.
    action = make_so3_action([2 * 3.14159265, 0.0, 0.0])
    R = exponential_map(action)
    assert torch.allclose(R, torch.eye(3), atol=1e-4)
