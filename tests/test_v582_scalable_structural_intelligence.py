import torch
from lgae_v3.types import make_graph_buffers
from lgae_v3.structural_intelligence import (ANNCandidateRetriever, approximate_fosr_candidates, EpistemicScaleCalibrator,
    WLDeduplicatedSpectralReplayBuffer, wl_graph_hash, contextual_lcb_beta, ScalableStructuralExecutive, exact_candidate_deltas)
from lgae_v3.reasoning import ConcreteAction
from lgae_v3.benchmark.tasks import StructuralAction

def graph(n=40):
    edges=[(i,i+1,1.) for i in range(n-1)]+[(i,(i+7)%n,1.) for i in range(0,n,5) if i!=(i+7)%n]
    return make_graph_buffers(n,edges,capacity=max(128,len(edges)+64))

def test_ann_retriever_does_not_call_dense_logits(monkeypatch):
    g=graph(); z=torch.randn(g.num_nodes,8); nh=torch.randn(g.num_nodes,24); r=ANNCandidateRetriever(24,12,ann_backend='numpy')
    monkeypatch.setattr(r,'logits',lambda *_: (_ for _ in ()).throw(AssertionError('dense path used')))
    cs=r.candidates(g,z,nh,top_k=12,neighbors_per_node=6)
    assert cs and len(cs)<=12 and all(c.channel=='learned_ann' for c in cs)

def test_approx_fosr_returns_valid_nonedges():
    g=graph(20); z=torch.randn(20,8); cs=approximate_fosr_candidates(g,z,8)
    existing={tuple(sorted((int(g.src[i]),int(g.dst[i])))) for i in torch.where(g.valid)[0].tolist()}
    assert cs and all(tuple(sorted((c.target['u'],c.target['v']))) not in existing for c in cs)

def test_calibrator_improves_synthetic_nll():
    raw=torch.full((128,),.15); err=torch.full((128,),.60); c=EpistemicScaleCalibrator(); res=c.fit(raw,err,steps=80)
    assert res.nll_after < res.nll_before and res.scale>1

def test_wl_hash_is_label_permutation_invariant_for_path():
    a=make_graph_buffers(5,[(0,1,1.),(1,2,1.),(2,3,1.),(3,4,1.)],capacity=8)
    b=make_graph_buffers(5,[(4,2,1.),(2,0,1.),(0,3,1.),(3,1,1.)],capacity=8)
    assert wl_graph_hash(a)==wl_graph_hash(b)

def test_contextual_beta_is_more_conservative_for_risky_irreversible_ood():
    safe=contextual_lcb_beta(1.0,epistemic_std=.1,risk=.1,reversibility=1,governor_margin=1,ood_score=0)
    risky=contextual_lcb_beta(1.0,epistemic_std=.8,risk=.9,reversibility=.1,governor_margin=.2,ood_score=.8)
    assert risky>safe

def test_scalable_executive_predicts_and_replay_accepts():
    g=graph(12); z=torch.randn(12,8); cs=[ConcreteAction(StructuralAction.NO_OP)]+approximate_fosr_candidates(g,z,4)
    ex=ScalableStructuralExecutive(d_max=8,hidden_dim=24,members=3,seed=2); vals=ex.predict(g,z,cs); assert len(vals)==len(cs)
    class O:
        def __init__(self,c,d): self.candidate=c; self.delta_utility=d; self.accepted_by_governor=True; self.metadata={}
    ds=exact_candidate_deltas(g,z,cs); ex.add_counterfactual_group(g,z,[O(c,d) for c,d in zip(cs,ds)]); assert len(ex.replay)==1
