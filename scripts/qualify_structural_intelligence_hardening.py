#!/usr/bin/env python3
from __future__ import annotations
import json, random
from pathlib import Path
import torch
from lgae_v3.structural_intelligence import (
    ProceduralGraphFactory, ConservativeStructuralExecutive,
    effective_resistance_candidates, fosr_candidates, forman_flow_candidates,
    merge_candidate_channels, exact_candidate_deltas, candidate_regret,
    uncertainty_calibration,
)
from lgae_v3.reasoning import ConcreteAction
from lgae_v3.benchmark.tasks import StructuralAction

SEED=581
random.seed(SEED); torch.manual_seed(SEED)
factory=ProceduralGraphFactory(latent_dim=8)
train_families=['erdos_renyi','block','tree']
heldout_families=['small_world','scale_free','geometric']
train=factory.sample(train_families,12,n_range=(8,10),seed=SEED)
held=factory.sample(heldout_families,6,n_range=(8,10),seed=SEED+1)
model=ConservativeStructuralExecutive(d_max=8,hidden_dim=32,members=4,prior_scale=.5,beta_lcb=1.5,seed=SEED)

class O:
    def __init__(self,c,d): self.candidate=c; self.delta_utility=float(d); self.accepted_by_governor=True; self.metadata={}

def channels(case,k=4):
    return merge_candidate_channels(
        effective_resistance_candidates(case.graph,case.z,k),
        fosr_candidates(case.graph,case.z,k),
        forman_flow_candidates(case.graph,case.z,k),
        max_candidates=1+3*k,
    )

def all_nonedges(case):
    g,z=case.graph,case.z
    existing={tuple(sorted((int(g.src[i]),int(g.dst[i])))) for i in torch.where(g.valid)[0].tolist()}
    out=[ConcreteAction(StructuralAction.NO_OP,channel='oracle_noop')]
    for u in range(g.num_nodes):
        for v in range(u+1,g.num_nodes):
            if (u,v) in existing: continue
            d=float(torch.linalg.vector_norm(z[u]-z[v]))
            out.append(ConcreteAction(StructuralAction.ADD_EDGE,{'u':u,'v':v,'weight':1.0,'length':max(d,1e-3)},channel='oracle_pool'))
    return out

for case in train:
    cs=channels(case); exact=exact_candidate_deltas(case.graph,case.z,cs)
    model.add_counterfactual_group(case.graph,case.z,[O(c,d) for c,d in zip(cs,exact)])
training=[model.train_step(batch_states=6,ranking_weight=.75) for _ in range(16)]

learned_regrets=[]; er_regrets=[]; fosr_regrets=[]; ff_regrets=[]; errors=[]; stds=[]; recall=[]; epi_held=[]
for case in held:
    cs=channels(case); exact=exact_candidate_deltas(case.graph,case.z,cs)
    vals=model.predict(case.graph,case.z,cs); score={v.candidate.key():v.score for v in vals}
    learned_regrets.append(candidate_regret([score[c.key()] for c in cs],exact).regret)
    for channel,bag in [('effective_resistance',er_regrets),('fosr_reference',fosr_regrets),('forman_flow_reference',ff_regrets)]:
        hs=[-1e9]*len(cs); hs[0]=0.0
        for i,c in enumerate(cs):
            if c.channel==channel: hs[i]=c.prior_score
        bag.append(candidate_regret(hs,exact).regret)
    bykey={v.candidate.key():v for v in vals}
    for c,d in zip(cs,exact):
        if d==float('-inf'): continue
        v=bykey[c.key()]; errors.append(v.mean_delta_utility-d); stds.append(v.std_delta_utility); epi_held.append(float(c.metadata.get('epistemic_std',0.0)))
    oracle_pool=all_nonedges(case); od=exact_candidate_deltas(case.graph,case.z,oracle_pool); oi=max(range(len(od)),key=lambda i:od[i]); recall.append(oracle_pool[oi].key() in {c.key() for c in cs})

# ID uncertainty reference on training states after the same fitted model.
epi_id=[]
for case in train[:6]:
    cs=channels(case); vals=model.predict(case.graph,case.z,cs)
    epi_id.extend(float(v.candidate.metadata.get('epistemic_std',0.0)) for v in vals)

mean=lambda xs: sum(xs)/max(len(xs),1)
cal=uncertainty_calibration(errors,stds)
checks={
    'train_heldout_families_disjoint':set(train_families).isdisjoint(heldout_families),
    'anchored_ensemble_members':model.model.members==4 and len(model.model.priors)==4,
    'frozen_randomized_priors':all(not p.requires_grad for pr in model.model.priors for p in pr.parameters()),
    'stratified_replay_populated':len(model.replay)>0 and len(model.replay._bins)>1,
    'reference_channels_present':all(channels(c) for c in held),
    'candidate_recall_measured':len(recall)==len(held),
    'regret_finite':all(torch.isfinite(torch.tensor(learned_regrets))),
    'calibration_finite':all(torch.isfinite(torch.tensor(list(cal.values())))),
}
best_baseline=min(mean(er_regrets),mean(fosr_regrets),mean(ff_regrets))
learned=mean(learned_regrets)
payload={
    'schema':'LGAE_STRUCTURAL_INTELLIGENCE_HARDENING_QUALIFICATION_V5_8_1',
    'status':'PASS' if all(checks.values()) else 'FAIL',
    'scientific_generalization_status':'PASS' if learned < best_baseline else 'NOT_YET_QUALIFIED',
    'checks':checks,
    'training':{'families':train_families,'states':len(train),'actions':model.replay.action_count,'last_step':training[-1]},
    'heldout':{
        'families':heldout_families,'states':len(held),
        'mean_learned_lcb_regret':learned,
        'mean_effective_resistance_regret':mean(er_regrets),
        'mean_fosr_reference_regret':mean(fosr_regrets),
        'mean_forman_flow_reference_regret':mean(ff_regrets),
        'oracle_candidate_recall_at_union':mean([1.0 if x else 0.0 for x in recall]),
        'beats_best_reference_baseline':learned < best_baseline,
    },
    'uncertainty':{
        'mean_id_epistemic_std':mean(epi_id),
        'mean_heldout_epistemic_std':mean(epi_held),
        'ood_greater_than_id':mean(epi_held)>mean(epi_id),
        'calibration':cal,
    },
    'baseline_note':'fosr_reference is an exact spectral-gap-gain proposal reference on bounded graphs. forman_flow_reference is a BORF-style AF3 deficit proposal proxy, not a paper-exact BORF implementation.',
    'claim_boundary':'PASS certifies v5.8.1 infrastructure and measurement integrity only. Learned generalization is claimed only when scientific_generalization_status is PASS.'
}
Path('structural_intelligence_hardening_qualification_report.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
raise SystemExit(0 if payload['status']=='PASS' else 1)
