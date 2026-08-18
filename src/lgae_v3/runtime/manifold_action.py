"""Lie-group/manifold action abstraction (Phase 16).

Generalizes the runtime's action space from discrete graph mutations to
actions on a Lie group manifold. A ``ManifoldAction`` represents a smooth
transformation parameterized by Lie algebra elements, enabling:

  - continuous gauge transformations (not just discrete reweighting)
  - manifold-valued gradient descent on the fiber bundle
  - exponential map from Lie algebra to Lie group

Supported Lie groups:
  - SO(3): rotations in 3D (9-dim representation, 3-dim algebra)
  - SU(2): spin rotations (4-dim quaternion representation, 3-dim algebra)
  - GL(d): general linear group (d^2-dim representation, d^2-dim algebra)

The exponential map converts Lie algebra elements (tangent vectors) to
group elements (manifold points). For SO(3), this is the Rodrigues formula.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch import Tensor


class LieGroup(str, Enum):
    SO3 = "so3"       # 3D rotations
    SU2 = "su2"       # 2x2 unitary with det=1 (quaternions)
    GL = "gl"         # general linear group


@dataclass(frozen=True, slots=True)
class ManifoldAction:
    """An action on a Lie group manifold, parameterized by Lie algebra elements."""
    group: LieGroup
    algebra_element: Tensor  # Lie algebra element (tangent vector at identity)
    dim: int  # dimension of the group representation

    def to_log(self) -> dict[str, Any]:
        return {
            "group": self.group.value,
            "dim": int(self.dim),
            "algebra_norm": float(self.algebra_element.norm().item()),
        }


def _expm_so3(omega: Tensor) -> Tensor:
    """Exponential map for SO(3) via Rodrigues' formula.

    omega: [3] rotation vector (axis * angle)
    returns: [3, 3] rotation matrix
    """
    theta = omega.norm()
    if theta < 1e-10:
        return torch.eye(3, dtype=omega.dtype, device=omega.device)
    k = omega / theta
    K = torch.zeros(3, 3, dtype=omega.dtype, device=omega.device)
    K[0, 1] = -k[2]
    K[0, 2] = k[1]
    K[1, 0] = k[2]
    K[1, 2] = -k[0]
    K[2, 0] = -k[1]
    K[2, 1] = k[0]
    I = torch.eye(3, dtype=omega.dtype, device=omega.device)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def _expm_su2(omega: Tensor) -> Tensor:
    """Exponential map for SU(2) via quaternion representation.

    omega: [3] pure imaginary quaternion part
    returns: [4] quaternion (w, x, y, z)
    """
    theta = omega.norm()
    if theta < 1e-10:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=omega.dtype, device=omega.device)
    axis = omega / theta
    w = torch.cos(theta / 2)
    xyz = torch.sin(theta / 2) * axis
    return torch.cat([w.unsqueeze(0), xyz])


def _expm_gl(A: Tensor) -> Tensor:
    """Exponential map for GL(d) via matrix exponential (Taylor series).

    A: [d, d] matrix
    returns: [d, d] matrix exponential
    """
    d = A.shape[0]
    result = torch.eye(d, dtype=A.dtype, device=A.device)
    term = torch.eye(d, dtype=A.dtype, device=A.device)
    for i in range(1, 20):  # 20 terms is usually sufficient
        term = term @ A / i
        result = result + term
    return result


def exponential_map(action: ManifoldAction) -> Tensor:
    """Compute the exponential map from Lie algebra to Lie group.

    Returns the group element (matrix or quaternion) corresponding to the
    action's algebra element.
    """
    if action.group == LieGroup.SO3:
        return _expm_so3(action.algebra_element)
    elif action.group == LieGroup.SU2:
        return _expm_su2(action.algebra_element)
    elif action.group == LieGroup.GL:
        return _expm_gl(action.algebra_element)
    else:
        raise ValueError(f"unsupported Lie group: {action.group}")


def compose(group: LieGroup, a: Tensor, b: Tensor) -> Tensor:
    """Compose two group elements (group multiplication)."""
    if group == LieGroup.SO3:
        return a @ b
    elif group == LieGroup.SU2:
        # Quaternion multiplication: (w1, v1) * (w2, v2)
        w1, v1 = a[0], a[1:]
        w2, v2 = b[0], b[1:]
        w = w1 * w2 - torch.dot(v1, v2)
        v = w1 * v2 + w2 * v1 + torch.cross(v1, v2, dim=0)
        return torch.cat([w.unsqueeze(0), v])
    elif group == LieGroup.GL:
        return a @ b
    else:
        raise ValueError(f"unsupported Lie group: {group}")


def inverse(group: LieGroup, g: Tensor) -> Tensor:
    """Compute the inverse of a group element."""
    if group == LieGroup.SO3:
        return g.T  # orthogonal: inverse = transpose
    elif group == LieGroup.SU2:
        # Quaternion inverse: conjugate (negate vector part)
        return torch.cat([g[0:1], -g[1:]])
    elif group == LieGroup.GL:
        return torch.linalg.inv(g)
    else:
        raise ValueError(f"unsupported Lie group: {group}")


def make_so3_action(omega: list[float] | Tensor) -> ManifoldAction:
    """Create an SO(3) action from a rotation vector."""
    if not isinstance(omega, Tensor):
        omega = torch.tensor(omega, dtype=torch.float32)
    return ManifoldAction(group=LieGroup.SO3, algebra_element=omega, dim=3)


def make_su2_action(omega: list[float] | Tensor) -> ManifoldAction:
    """Create an SU(2) action from a rotation vector."""
    if not isinstance(omega, Tensor):
        omega = torch.tensor(omega, dtype=torch.float32)
    return ManifoldAction(group=LieGroup.SU2, algebra_element=omega, dim=4)


def make_gl_action(A: list[float] | Tensor, d: int) -> ManifoldAction:
    """Create a GL(d) action from a matrix."""
    if not isinstance(A, Tensor):
        A = torch.tensor(A, dtype=torch.float32).reshape(d, d)
    return ManifoldAction(group=LieGroup.GL, algebra_element=A, dim=d * d)
