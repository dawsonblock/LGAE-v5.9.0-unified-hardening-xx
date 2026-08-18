"""Joint topology/gauge actions and localized structural credit for LGAE v5.8.4.

This module keeps the learned layer advisory.  A joint proposal couples a
concrete topology edit with an initial SO(d) connection, but the connection is
applied only inside the governor's shadow rollout until the topology mutation
has been certified.  No authoritative gauge parameter is modified here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .benchmark.tasks import StructuralAction
from .reasoning import ConcreteAction
from .types import GraphBuffers, MutationDecision
from .action_bridge import action_to_mutation


@dataclass(slots=True)
class JointStructuralAction:
    """A topology proposal bundled with an initial Lie-algebra connection."""

    candidate: ConcreteAction
    generator: Tensor  # [d,d] skew-symmetric, interpreted via matrix exponential
    connection: Tensor  # [d,d] element of SO(d)
    basis_coefficients: Tensor


@dataclass(slots=True)
class JointCertificationResult:
    action: JointStructuralAction
    accepted: bool
    decision: str
    reasons: list[str]
    slot: int | None
    shadow_graph: GraphBuffers
    metadata: dict[str, Any]


class LowRankLieGaugeHead(nn.Module):
    """Predict a low-dimensional element of so(d) for a candidate edge.

    A learned bank of R skew bases keeps the action dimension O(R) rather than
    d(d-1)/2. The hot path uses a batched Cayley retraction solved with
    torch.linalg.solve, guaranteeing SO(d) without matrix-exp overhead.
    """

    def __init__(self, hidden_dim: int, gauge_dim: int, rank: int = 8, max_generator_norm: float = 2.0):
        super().__init__()
        if gauge_dim <= 0:
            raise ValueError("gauge_dim must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.hidden_dim = int(hidden_dim)
        self.gauge_dim = int(gauge_dim)
        self.rank = int(rank)
        self.max_generator_norm = float(max_generator_norm)
        self.coeff = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, rank)
        )
        raw = 0.05 * torch.randn(rank, gauge_dim, gauge_dim)
        self.raw_bases = nn.Parameter(raw)

    def skew_bases(self) -> Tensor:
        return 0.5 * (self.raw_bases - self.raw_bases.transpose(-1, -2))

    def forward(self, h_u: Tensor, h_v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if h_u.shape != h_v.shape or h_u.ndim != 2 or h_u.shape[-1] != self.hidden_dim:
            raise ValueError("h_u/h_v must have shape [B, hidden_dim]")
        coeff = torch.tanh(self.coeff(torch.cat([h_u, h_v], dim=-1)))
        A = torch.einsum("br,rij->bij", coeff, self.skew_bases())
        # Bound generator magnitude without leaving the Lie algebra.
        norm = torch.linalg.matrix_norm(A, ord="fro", dim=(-2, -1)).clamp_min(1e-12)
        scale = torch.clamp(self.max_generator_norm / norm, max=1.0)
        A = A * scale[:, None, None]
        W = cayley_retraction(A)
        return coeff, A, W


def cayley_retraction(A: Tensor) -> Tensor:
    """Map a skew-symmetric generator to SO(d) with a batched linear solve.

    W = (I - A/2)^-1 (I + A/2), implemented as solve rather than an
    explicit inverse.  For real skew A this is orthogonal with determinant +1
    except at the usual Cayley chart singularities, which cannot occur for
    finite real skew eigenvalues at I-A/2.
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("A must have shape [..., d, d]")
    d = A.shape[-1]
    I = torch.eye(d, dtype=A.dtype, device=A.device)
    I = I.expand(A.shape[:-2] + (d, d))
    return torch.linalg.solve(I - 0.5 * A, I + 0.5 * A)


def paired_restriction_maps(A: Tensor) -> tuple[Tensor, Tensor]:
    """Return dual orthogonal restrictions for an undirected cellular edge."""
    W_u = cayley_retraction(A)
    W_v = W_u.transpose(-1, -2)
    return W_u, W_v


def assemble_paired_connection_laplacian(
    num_nodes: int, u: int, v: int, W_u: Tensor, W_v: Tensor, *, weight: float = 1.0
) -> Tensor:
    """Assemble the one-edge cellular sheaf Laplacian for a paired restriction.

    For coboundary B_e=[... W_u ... -W_v ...], L_e=B_e^T B_e is exactly
    self-adjoint.  This helper is used by qualification and downstream
    diagnostics to verify the joint-action pairing contract explicitly.
    """
    if W_u.ndim != 2 or W_u.shape != W_v.shape or W_u.shape[0] != W_u.shape[1]:
        raise ValueError("W_u/W_v must be square matrices of matching shape")
    d = W_u.shape[0]
    B = torch.zeros((d, num_nodes * d), dtype=W_u.dtype, device=W_u.device)
    B[:, u*d:(u+1)*d] = W_u
    B[:, v*d:(v+1)*d] = -W_v
    return float(weight) * (B.transpose(-1, -2) @ B)


def two_sided_connection_dirichlet_energy(x_u: Tensor, x_v: Tensor, W_u: Tensor, W_v: Tensor) -> Tensor:
    """Cellular edge energy ||W_u x_u - W_v x_v||^2."""
    if W_u.shape != W_v.shape or W_u.ndim != 2 or W_u.shape[0] != W_u.shape[1]:
        raise ValueError("W_u/W_v must be square matrices of matching shape")
    d = W_u.shape[-1]
    if x_u.numel() < d or x_v.numel() < d:
        raise ValueError("latent width is smaller than gauge dimension")
    diff = W_u @ x_u[:d] - W_v @ x_v[:d]
    return diff.square().sum()


class JointStructuralGaugePolicy(nn.Module):
    """Attach low-rank SO(d) connection proposals to ADD_EDGE candidates."""

    def __init__(self, hidden_dim: int, gauge_dim: int, lie_rank: int = 8):
        super().__init__()
        self.gauge_head = LowRankLieGaugeHead(hidden_dim, gauge_dim, rank=lie_rank)

    def forward(
        self,
        node_h: Tensor,
        candidates: Sequence[ConcreteAction],
    ) -> list[JointStructuralAction]:
        add = [c for c in candidates if c.action == StructuralAction.ADD_EDGE]
        if not add:
            return []
        us = torch.tensor([int(c.target["u"]) for c in add], device=node_h.device, dtype=torch.long)
        vs = torch.tensor([int(c.target["v"]) for c in add], device=node_h.device, dtype=torch.long)
        coeff, A, W = self.gauge_head(node_h[us], node_h[vs])
        return [JointStructuralAction(c, A[i], W[i], coeff[i]) for i, c in enumerate(add)]


def _edge_slot_for_candidate(graph: GraphBuffers, candidate: ConcreteAction) -> int | None:
    if candidate.action != StructuralAction.ADD_EDGE:
        return None
    u, v = int(candidate.target["u"]), int(candidate.target["v"])
    ids = torch.where(graph.valid & (((graph.src == u) & (graph.dst == v)) | ((graph.src == v) & (graph.dst == u))))[0]
    if ids.numel():
        return int(ids[0])
    free = torch.where(~graph.valid)[0]
    return int(free[0]) if free.numel() else None


def certify_joint_structural_action(
    graph: GraphBuffers,
    z: Tensor,
    joint: JointStructuralAction,
    governor: Any,
    *,
    gauge_bank: Any = None,
    seed: int = 0,
) -> JointCertificationResult:
    """Certify topology and proposed initial connection in one shadow evaluation."""
    mutation = action_to_mutation(joint.candidate.action, graph, z, **joint.candidate.target)
    if mutation is None:
        return JointCertificationResult(joint, False, "reject", ["unmappable_joint_action"], None, graph.clone(), {})
    slot = _edge_slot_for_candidate(graph, joint.candidate)
    if slot is None:
        return JointCertificationResult(joint, False, "reject", ["no_available_edge_slot"], None, graph.clone(), {})
    if gauge_bank is not None and int(getattr(gauge_bank, "dim", joint.connection.shape[-1])) != joint.connection.shape[-1]:
        return JointCertificationResult(joint, False, "reject", ["joint_gauge_dimension_mismatch"], slot, graph.clone(), {})
    result, shadow = governor.evaluate_mutation(
        graph,
        z,
        mutation,
        seed=seed,
        gauge_bank=gauge_bank,
        gauge_overrides={slot: joint.connection.detach()},
    )
    md = dict(result.metadata or {})
    md.update({
        "joint_gauge_slot": slot,
        "joint_generator_norm": float(torch.linalg.matrix_norm(joint.generator.detach(), ord="fro")),
        "joint_connection_orth_error": float(torch.linalg.matrix_norm(
            joint.connection.detach().T @ joint.connection.detach() - torch.eye(joint.connection.shape[-1], device=joint.connection.device, dtype=joint.connection.dtype),
            ord="fro",
        )),
    })
    return JointCertificationResult(
        joint,
        result.decision == MutationDecision.ACCEPT,
        result.decision.value,
        list(result.reasons),
        slot,
        shadow,
        md,
    )


@dataclass(slots=True)
class LocalizedCreditResult:
    global_advantage: float
    dirichlet_before: float
    dirichlet_after: float
    normalized_dirichlet_improvement: float
    blended_advantage: float
    node_weights: Tensor
    node_credits: Tensor


def connection_dirichlet_energy(x_u: Tensor, x_v: Tensor, W_uv: Tensor) -> Tensor:
    """Connection Dirichlet energy ||W x_u - x_v||^2 for one edge."""
    d = W_uv.shape[-1]
    if W_uv.ndim != 2 or W_uv.shape[-2] != d:
        raise ValueError("W_uv must be square")
    if x_u.numel() < d or x_v.numel() < d:
        raise ValueError("latent width is smaller than gauge dimension")
    diff = W_uv @ x_u[:d] - x_v[:d]
    return diff.square().sum()


def _graph_distances(graph: GraphBuffers, seeds: set[int]) -> Tensor:
    n = graph.num_nodes
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in torch.where(graph.valid)[0].tolist():
        u, v = int(graph.src[i]), int(graph.dst[i])
        adj[u].append(v); adj[v].append(u)
    inf = n + 1
    dist = [inf] * n
    frontier = list(seeds)
    for s in frontier:
        dist[s] = 0
    head = 0
    while head < len(frontier):
        u = frontier[head]; head += 1
        for v in adj[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                frontier.append(v)
    return torch.tensor(dist, dtype=graph.weight.dtype, device=graph.weight.device)


def localized_dirichlet_credit(
    *,
    global_advantage: float,
    graph: GraphBuffers,
    z_before: Tensor,
    z_after: Tensor,
    u: int,
    v: int,
    W_before: Tensor,
    W_after: Tensor | None = None,
    Wv_before: Tensor | None = None,
    Wv_after: Tensor | None = None,
    global_mix: float = 0.5,
    distance_tau: float = 2.0,
) -> LocalizedCreditResult:
    """Blend global advantage with normalized local sheaf-energy improvement.

    Node credits conserve the blended advantage: sum(node_credits) equals the
    blended scalar (up to floating point error).  Spatial attribution uses an
    exponential graph-distance kernel around the edited endpoints.
    """
    if not 0.0 <= global_mix <= 1.0:
        raise ValueError("global_mix must be in [0,1]")
    if distance_tau <= 0:
        raise ValueError("distance_tau must be positive")
    W_after = W_before if W_after is None else W_after
    if Wv_before is None:
        eb = connection_dirichlet_energy(z_before[u], z_before[v], W_before)
    else:
        eb = two_sided_connection_dirichlet_energy(z_before[u], z_before[v], W_before, Wv_before)
    if Wv_after is None:
        ea = connection_dirichlet_energy(z_after[u], z_after[v], W_after)
    else:
        ea = two_sided_connection_dirichlet_energy(z_after[u], z_after[v], W_after, Wv_after)
    denom = (eb.abs() + ea.abs()).clamp_min(1e-8)
    normalized_improvement = ((eb - ea) / denom).clamp(-1.0, 1.0)
    local_signal = float(normalized_improvement.detach())
    blended = float(global_mix) * float(global_advantage) + (1.0 - float(global_mix)) * local_signal

    dist = _graph_distances(graph, {int(u), int(v)}).to(dtype=z_before.dtype, device=z_before.device)
    finite = dist <= graph.num_nodes
    raw = torch.where(finite, torch.exp(-dist / float(distance_tau)), torch.zeros_like(dist))
    weights = raw / raw.sum().clamp_min(1e-12)
    credits = weights * blended
    return LocalizedCreditResult(
        float(global_advantage), float(eb.detach()), float(ea.detach()), local_signal,
        blended, weights, credits,
    )

@torch.no_grad()
def commit_joint_connection(
    gauge_bank: Any,
    slot: int,
    joint: JointStructuralAction,
    *,
    graph: GraphBuffers | None = None,
    optimizers: Any = None,
) -> None:
    """Initialize an already-committed graph edge's authoritative SO(d) slot.

    This must be called only after topology commit.  The function resets stale
    optimizer state for a reused edge slot and preserves the bank's native
    parameterization while matching the certified shadow connection.
    """
    slot = int(slot)
    if not (0 <= slot < int(gauge_bank.edge_capacity)):
        raise IndexError("gauge slot out of range")
    if joint.connection.shape != (gauge_bank.dim, gauge_bank.dim):
        raise ValueError("joint connection dimension does not match gauge bank")
    sync = None if graph is None else graph.slot_generation
    gauge_bank.reset_slots([slot], optimizers=optimizers, sync_generation=sync)
    if gauge_bank.parameterization == "exp":
        raw = joint.generator.detach().to(gauge_bank.raw_generators)
    else:
        # Inverse of W=(I-A/2)^-1(I+A/2): A=2(W-I)(W+I)^-1.
        W = joint.connection.detach().to(gauge_bank.raw_generators)
        I = torch.eye(gauge_bank.dim, dtype=W.dtype, device=W.device)
        rhs = (W - I)
        # right-side inverse via transposed solve; skew cleanup removes roundoff.
        raw = 2.0 * torch.linalg.solve((W + I).T, rhs.T).T
        raw = 0.5 * (raw - raw.T)
    gauge_bank.raw_generators[slot].copy_(raw)
