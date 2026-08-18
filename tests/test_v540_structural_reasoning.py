import torch

from lgae_v3 import make_graph_buffers, LGAEConfig, LGAEEngine
from lgae_v3.benchmark.tasks import StructuralAction
from lgae_v3.reasoning import (
    ConcreteAction, CandidateGenerator, StructuralReasoningExecutive,
    CounterfactualFactory, certify_ranked_candidates,
)


def _graph():
    return make_graph_buffers(
        6,
        [(0,1,1.0),(1,2,1.0),(3,4,1.0),(4,5,1.0)],
        capacity=16,
    )


def test_candidate_generator_is_bounded_and_noop_present():
    g = _graph(); z = torch.randn(6, 8)
    gen = CandidateGenerator(max_candidates=12, seed=1)
    out = gen.generate(g, z)
    assert len(out) <= 12
    assert out[0].action == StructuralAction.NO_OP
    assert len({c.key() for c in out}) == len(out)


def test_reasoner_scores_concrete_actions():
    g = _graph(); z = torch.randn(6, 8)
    r = StructuralReasoningExecutive(d_max=16, hidden_dim=32, max_candidates=10, seed=2)
    cs = r.generate_candidates(g, z)
    vals = r.predict(g, z, cs)
    assert len(vals) == len(cs)
    assert all(torch.isfinite(torch.tensor(v.score)) for v in vals)
    assert all(v.std_delta_utility > 0 for v in vals)
    plan = r.plan(g, z)
    assert plan.selected in plan.ranked


def test_graph_encoder_is_permutation_equivariant_at_node_level():
    g = _graph(); z = torch.randn(6, 8)
    r = StructuralReasoningExecutive(d_max=8, hidden_dim=24, max_candidates=8, seed=3)
    with torch.no_grad():
        h, gg = r.encoder(g, z)
    perm = torch.tensor([2,0,1,5,3,4])
    inv = torch.empty_like(perm); inv[perm] = torch.arange(6)
    edges=[]
    for i in torch.where(g.valid)[0].tolist():
        edges.append((int(inv[int(g.src[i])]), int(inv[int(g.dst[i])]), float(g.weight[i]), float(g.length[i])))
    gp = make_graph_buffers(6, edges, capacity=16)
    zp = z[perm]
    with torch.no_grad():
        hp, ggp = r.encoder(gp, zp)
    assert torch.allclose(hp, h[perm], atol=1e-5, rtol=1e-5)
    assert torch.allclose(ggp, gg, atol=1e-5, rtol=1e-5)


def test_counterfactual_factory_governor_grounded():
    cfg = LGAEConfig(); cfg.fiber.d_base=4; cfg.fiber.d_max=8
    cfg.audit.persistent_homology_enabled=False
    g=_graph(); engine=LGAEEngine(g,cfg)
    z=engine.fibers().detach().clone()
    candidates=[
        ConcreteAction(StructuralAction.NO_OP, channel='baseline'),
        ConcreteAction(StructuralAction.ADD_EDGE, {'u':2,'v':3,'weight':1.0,'length':1.0}, 'test'),
    ]
    factory=CounterfactualFactory(horizon=0)
    utility=lambda graph,z: float(graph.edge_count)
    out=factory.evaluate(g,z,candidates,governor=engine.governor,utility_fn=utility)
    assert len(out)==2
    assert out[0].delta_utility==0.0
    assert out[0].accepted_by_governor
    assert out[1].decision in {'accept','reject','quarantine'}


def test_replay_training_updates_q_model():
    g=_graph(); z=torch.randn(6,8)
    r=StructuralReasoningExecutive(d_max=8, hidden_dim=24, max_candidates=8, seed=4)
    c1=ConcreteAction(StructuralAction.NO_OP, channel='baseline')
    c2=ConcreteAction(StructuralAction.ADD_EDGE, {'u':2,'v':3,'weight':1.0,'length':1.0}, 'test')
    from lgae_v3.reasoning import CounterfactualOutcome
    for i in range(20):
        r.record(g,z,CounterfactualOutcome(c1,0.0,0,0,True,'accept',g.state_hash(),g.state_hash()))
        r.record(g,z,CounterfactualOutcome(c2,1.0,0,1,True,'accept',g.state_hash(),g.state_hash()))
    metrics=r.train_step(batch_size=16)
    assert metrics['samples']==16
    assert torch.isfinite(torch.tensor(metrics['loss']))
