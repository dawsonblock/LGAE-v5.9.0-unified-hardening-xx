import torch
from lgae_v3.types import make_graph_buffers
from lgae_v3.structural_intelligence import (
    RandomizedPriorEnsembleQ, SpectralStratifiedReplayBuffer, structural_regime_features,
    fosr_candidates, forman_flow_candidates, effective_resistance_candidates,
    merge_candidate_channels, ConservativeStructuralExecutive, ContrastiveCandidateRetriever,
    exact_candidate_deltas,
)
from lgae_v3.reasoning import ConcreteAction
from lgae_v3.benchmark.tasks import StructuralAction


def g(): return make_graph_buffers(7,[(0,1,1.),(1,2,1.),(2,3,1.),(3,4,1.),(4,5,1.),(5,6,1.)],capacity=20)

def test_randomized_prior_ensemble_preserves_member_diversity():
    G=g(); z=torch.randn(7,8); cs=effective_resistance_candidates(G,z,3)
    m=RandomizedPriorEnsembleQ(d_max=8,hidden_dim=24,members=4,prior_scale=.75,seed=81)
    with torch.no_grad(): p=m(G,z,cs)
    assert p['member_means'].shape==(4,len(cs)); assert float(p['epistemic_std'].mean())>0
    assert all(not x.requires_grad for pr in m.priors for x in pr.parameters())


def test_structural_regime_replay_samples_multiple_bins():
    b=SpectralStratifiedReplayBuffer(capacity_states=20)
    z=torch.randn(7,8); c=ConcreteAction(StructuralAction.NO_OP)
    g1=g(); g2=make_graph_buffers(7,[(i,j,1.) for i in range(7) for j in range(i+1,7)],capacity=30)
    b.add_group(g1,z,[(c,0.,0.,0.)]); b.add_group(g2,z,[(c,0.,0.,0.)])
    assert len(b._bins)>=2; assert len(b.sample_groups(2))==2
    assert structural_regime_features(g1).density < structural_regime_features(g2).density


def test_reference_candidate_channels_and_merge():
    G=g(); z=torch.randn(7,8)
    fo=fosr_candidates(G,z,2); ff=forman_flow_candidates(G,z,2); er=effective_resistance_candidates(G,z,2)
    assert fo and ff and er
    merged=merge_candidate_channels(fo,ff,er,max_candidates=7)
    assert merged[0].action==StructuralAction.NO_OP
    assert len({x.key() for x in merged})==len(merged)


def test_contrastive_retriever_loss_and_candidates_are_finite():
    G=g(); z=torch.randn(7,8); base=ConservativeStructuralExecutive(d_max=8,hidden_dim=24,members=3,seed=3)
    nh,_=base.model.encoder(G,z); r=ContrastiveCandidateRetriever(24,12)
    mask=torch.ones(7,7,dtype=torch.bool); mask.fill_diagonal_(False)
    for i in torch.where(G.valid)[0].tolist():
        u,v=int(G.src[i]),int(G.dst[i]); mask[u,v]=mask[v,u]=False
    adv=torch.randn(7,7); loss=r.loss(nh,adv,mask)
    assert torch.isfinite(loss); assert r.candidates(G,z,nh,3)


def test_conservative_executive_trains_with_anchored_heads():
    G=g(); z=torch.randn(7,8); cs=merge_candidate_channels(effective_resistance_candidates(G,z,3),fosr_candidates(G,z,2),max_candidates=6)
    exact=exact_candidate_deltas(G,z,cs)
    ex=ConservativeStructuralExecutive(d_max=8,hidden_dim=24,members=3,seed=9)
    class O:
        def __init__(self,c,d): self.candidate=c; self.delta_utility=d; self.accepted_by_governor=True; self.metadata={}
    ex.add_counterfactual_group(G,z,[O(c,d) for c,d in zip(cs,exact)])
    m=ex.train_step(batch_states=1)
    assert m['samples']==len(cs) and torch.isfinite(torch.tensor(m['loss']))
    vals=ex.predict(G,z,cs); assert len(vals)==len(cs)
