import math
import torch

from lgae_v3 import make_graph_buffers
from lgae_v3.benchmark.tasks import StructuralAction
from lgae_v3.reasoning import ConcreteAction
from lgae_v3.structural_intelligence import (
    StateGroupedReplayBuffer,
    EnsembleStructuralQ,
    StructuralIntelligenceExecutive,
    ProceduralGraphFactory,
    effective_resistance_matrix,
    effective_resistance_candidates,
    exact_candidate_deltas,
    candidate_regret,
    uncertainty_calibration,
)


def _g():
    return make_graph_buffers(6, [(0,1,1.0),(1,2,1.0),(2,3,1.0),(3,4,1.0),(4,5,1.0)], capacity=16)


def _cands(z):
    return [
        ConcreteAction(StructuralAction.NO_OP, channel="baseline"),
        ConcreteAction(StructuralAction.ADD_EDGE, {"u":0,"v":5,"weight":1.0,"length":float(torch.linalg.vector_norm(z[0]-z[5]))}, channel="test"),
        ConcreteAction(StructuralAction.ADD_EDGE, {"u":1,"v":4,"weight":1.0,"length":float(torch.linalg.vector_norm(z[1]-z[4]))}, channel="test"),
    ]


def test_state_grouped_replay_keeps_competing_actions_together():
    g=_g(); z=torch.randn(6,8); cs=_cands(z)
    b=StateGroupedReplayBuffer(capacity_states=4)
    b.add_group(g,z,[(cs[0],0.0,0.0,0.0),(cs[1],1.0,0.0,0.1),(cs[2],0.3,0.0,0.0)])
    groups=b.sample_groups(1)
    assert len(groups)==1 and len(groups[0])==3
    assert b.action_count==3
    assert {x.candidate.key() for x in groups[0]}=={c.key() for c in cs}


def test_ensemble_exposes_epistemic_and_aleatoric_uncertainty():
    g=_g(); z=torch.randn(6,8); cs=_cands(z)
    m=EnsembleStructuralQ(d_max=8, hidden_dim=24, members=3, seed=11)
    with torch.no_grad():
        p=m(g,z,cs)
    assert p['member_means'].shape==(3,3)
    assert torch.all(p['epistemic_std']>=0)
    assert torch.all(p['aleatoric_std']>0)
    assert torch.allclose(p['total_std'].square(), p['epistemic_std'].square()+p['aleatoric_std'].square(), atol=1e-5)


def test_grouped_training_uses_all_pairwise_preferences():
    g=_g(); z=torch.randn(6,8); cs=_cands(z)
    r=StructuralIntelligenceExecutive(d_max=8,hidden_dim=24,members=3,seed=7)
    class O:
        def __init__(self,c,d):
            self.candidate=c; self.delta_utility=d; self.accepted_by_governor=True; self.metadata={}
    r.add_counterfactual_group(g,z,[O(cs[0],0.0),O(cs[1],1.0),O(cs[2],0.25)])
    metrics=r.train_step(batch_states=1)
    assert metrics['states']==1 and metrics['samples']==3
    assert math.isfinite(metrics['loss']) and metrics['ranking_loss']>0


def test_effective_resistance_finds_long_path_endpoints():
    g=_g(); z=torch.randn(6,8)
    R=effective_resistance_matrix(g)
    assert R.shape==(6,6)
    assert torch.allclose(R,R.T,atol=1e-5)
    assert float(R[0,5]) > float(R[0,2])
    cs=effective_resistance_candidates(g,z,top_k=2)
    assert cs and cs[0].channel=='effective_resistance'
    assert {cs[0].target['u'],cs[0].target['v']}=={0,5}


def test_procedural_factory_generates_disjoint_heldout_families():
    f=ProceduralGraphFactory(latent_dim=6)
    train=f.sample(['erdos_renyi','block','tree'],6,n_range=(10,12),seed=3)
    held=f.sample(['small_world','scale_free','geometric'],6,n_range=(10,12),seed=4)
    assert {c.family for c in train}.isdisjoint({c.family for c in held})
    assert all(c.graph.num_nodes>=10 and c.z.shape[1]==6 for c in train+held)


def test_exact_regret_and_calibration_metrics_are_well_formed():
    g=_g(); z=torch.randn(6,8); cs=_cands(z)
    exact=exact_candidate_deltas(g,z,cs)
    # Deliberately choose the exact ordering to produce zero regret.
    rr=candidate_regret(exact,exact)
    assert rr.regret==0.0 and rr.oracle_delta==rr.chosen_delta
    cal=uncertainty_calibration([0.1,0.3,0.2],[0.2,0.2,0.5])
    assert 0<=cal['coverage_1sigma']<=1
    assert 0<=cal['coverage_2sigma']<=1
