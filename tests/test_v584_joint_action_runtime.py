import time
import torch

from lgae_v3 import (
    LowRankLieGaugeHead, cayley_retraction, paired_restriction_maps,
    assemble_paired_connection_laplacian, two_sided_connection_dirichlet_energy,
    localized_dirichlet_credit,
)
from lgae_v3.types import make_graph_buffers


def test_cayley_retraction_is_special_orthogonal_and_differentiable():
    torch.manual_seed(584)
    raw = torch.randn(8,4,4, requires_grad=True)
    A = 0.2 * (raw - raw.transpose(-1,-2))
    W = cayley_retraction(A)
    I = torch.eye(4).expand_as(W)
    assert torch.allclose(W.transpose(-1,-2) @ W, I, atol=2e-5)
    assert torch.all(torch.linalg.det(W) > 0.999)
    W.square().mean().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


def test_low_rank_head_uses_cayley_contract():
    torch.manual_seed(2)
    head = LowRankLieGaugeHead(16, 4, rank=3)
    hu = torch.randn(6,16); hv = torch.randn(6,16)
    _, A, W = head(hu,hv)
    assert torch.allclose(W, cayley_retraction(A), atol=1e-6)


def test_paired_restrictions_are_inverse_transposes():
    A = torch.tensor([[0.,-.4],[.4,0.]])
    Wu,Wv = paired_restriction_maps(A)
    assert torch.allclose(Wv, Wu.T, atol=1e-7)
    assert torch.allclose(Wu @ Wv, torch.eye(2), atol=1e-6)


def test_paired_connection_laplacian_is_self_adjoint_psd():
    A = torch.tensor([[0.,-.7],[.7,0.]])
    Wu,Wv = paired_restriction_maps(A)
    L = assemble_paired_connection_laplacian(3,0,2,Wu,Wv,weight=1.3)
    assert torch.allclose(L,L.T,atol=1e-7)
    assert float(torch.linalg.eigvalsh(L).min()) > -1e-6


def test_two_sided_dirichlet_flux_and_credit_conservation():
    g=make_graph_buffers(4,[(0,1,1.),(1,2,1.),(2,3,1.)],capacity=6)
    A=torch.tensor([[0.,-.3],[.3,0.]])
    Wu,Wv=paired_restriction_maps(A)
    zb=torch.randn(4,2); za=zb.clone(); za[1]=0.5*za[1]
    e=two_sided_connection_dirichlet_energy(zb[0],zb[1],Wu,Wv)
    assert e >= 0
    r=localized_dirichlet_credit(global_advantage=.35,graph=g,z_before=zb,z_after=za,u=0,v=1,
        W_before=Wu,W_after=Wu,Wv_before=Wv,Wv_after=Wv,global_mix=.6)
    assert torch.isclose(r.node_credits.sum(), torch.tensor(r.blended_advantage), atol=1e-6)
    assert torch.isclose(r.node_weights.sum(), torch.tensor(1.), atol=1e-6)


def test_batched_cayley_throughput_smoke():
    # Functional throughput guard, not a hardware-specific absolute SLA.
    torch.manual_seed(0)
    raw=torch.randn(64,8,8)
    A=0.1*(raw-raw.transpose(-1,-2))
    t0=time.perf_counter(); W=cayley_retraction(A); elapsed=time.perf_counter()-t0
    assert W.shape==(64,8,8)
    assert elapsed < 5.0
