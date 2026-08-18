import torch

from lgae_v3 import (
    JointStructuralGaugePolicy,
    certify_joint_structural_action,
    localized_dirichlet_credit,
    connection_dirichlet_energy,
)
from lgae_v3.reasoning import ConcreteAction
from lgae_v3.benchmark.tasks import StructuralAction
from lgae_v3.types import make_graph_buffers, MutationResult, MutationDecision
from lgae_v3.fibers import SOConnectionBank
from lgae_v3.governor import GeometryGovernor
from lgae_v3.config import LGAEConfig


def graph3():
    return make_graph_buffers(3, [(0, 1, 1.0)], capacity=4)


def test_low_rank_joint_policy_outputs_so_connection():
    torch.manual_seed(4)
    h = torch.randn(3, 16)
    c = ConcreteAction(StructuralAction.ADD_EDGE, {"u": 1, "v": 2, "weight": 1.0, "length": 1.0})
    p = JointStructuralGaugePolicy(hidden_dim=16, gauge_dim=4, lie_rank=3)
    out = p(h, [c])
    assert len(out) == 1
    A, W = out[0].generator, out[0].connection
    assert torch.allclose(A + A.T, torch.zeros_like(A), atol=1e-6)
    eye = torch.eye(4)
    assert torch.allclose(W.T @ W, eye, atol=1e-5)
    assert torch.det(W) > 0.999


def test_shadow_gauge_override_changes_gauge_diffusion_without_mutating_bank():
    g = graph3()
    cfg = LGAEConfig()
    cfg.fiber.gauge_dim = 2
    cfg.mutation.shadow_steps = 1
    cfg.mutation.shadow_eta = 0.2
    gov = GeometryGovernor(cfg)
    bank = SOConnectionBank(g.src.numel(), 2, parameterization="exp")
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.2, -0.1]])
    before = bank.raw_generators.detach().clone()
    base = gov.shadow_rollout(g, z, gauge_bank=bank)
    rot = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    changed = gov.shadow_rollout(g, z, gauge_bank=bank, gauge_overrides={0: rot})
    assert not torch.allclose(base, changed)
    assert torch.equal(before, bank.raw_generators.detach())


def test_joint_certification_passes_shadow_only_override():
    g = graph3(); z = torch.randn(3, 2)
    c = ConcreteAction(StructuralAction.ADD_EDGE, {"u": 1, "v": 2, "weight": 1.0, "length": 1.0})
    p = JointStructuralGaugePolicy(hidden_dim=8, gauge_dim=2, lie_rank=1)
    joint = p(torch.randn(3, 8), [c])[0]

    class StubGovernor:
        def __init__(self): self.overrides = None
        def evaluate_mutation(self, graph, z, mutation, **kwargs):
            self.overrides = kwargs.get("gauge_overrides")
            shadow = graph.clone(); md = mutation.apply(shadow)
            return MutationResult(MutationDecision.ACCEPT, [], metadata=md), shadow

    gov = StubGovernor()
    result = certify_joint_structural_action(g, z, joint, gov, gauge_bank=SOConnectionBank(g.src.numel(), 2))
    assert result.accepted
    assert result.slot is not None
    assert result.slot in gov.overrides
    assert g.edge_count == 1
    assert result.shadow_graph.edge_count == 2


def test_localized_dirichlet_credit_is_bounded_and_conservative():
    g = make_graph_buffers(4, [(0,1,1.0),(1,2,1.0),(2,3,1.0)], capacity=6)
    zb = torch.tensor([[1.,0.],[0.,1.],[0.2,0.1],[0.,0.]])
    za = zb.clone(); za[1] = torch.tensor([0.8, 0.1])
    W = torch.eye(2)
    r = localized_dirichlet_credit(
        global_advantage=0.4, graph=g, z_before=zb, z_after=za,
        u=0, v=1, W_before=W, global_mix=0.5, distance_tau=1.5,
    )
    assert -1.0 <= r.normalized_dirichlet_improvement <= 1.0
    assert torch.all(r.node_weights >= 0)
    assert torch.isclose(r.node_weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(r.node_credits.sum(), torch.tensor(r.blended_advantage), atol=1e-6)
    assert r.node_weights[0] > r.node_weights[3]


def test_connection_dirichlet_energy_zero_for_aligned_transport():
    x = torch.tensor([1.0, 2.0])
    assert float(connection_dirichlet_energy(x, x, torch.eye(2))) == 0.0


def test_commit_joint_connection_matches_certified_connection_for_cayley_bank():
    from lgae_v3 import commit_joint_connection
    g = graph3()
    c = ConcreteAction(StructuralAction.ADD_EDGE, {"u": 1, "v": 2, "weight": 1.0, "length": 1.0})
    policy = JointStructuralGaugePolicy(hidden_dim=8, gauge_dim=2, lie_rank=1)
    joint = policy(torch.randn(3, 8), [c])[0]
    bank = SOConnectionBank(g.src.numel(), 2, parameterization="cayley")
    # Simulate topology commit into the slot that the candidate would occupy.
    from lgae_v3.mutations import AddEdge
    md = AddEdge(1,2).apply(g)
    commit_joint_connection(bank, md["slot"], joint, graph=g)
    assert torch.allclose(bank.matrices()[md["slot"]], joint.connection, atol=2e-5)
    assert int(bank.slot_generation[md["slot"]]) == int(g.slot_generation[md["slot"]])
