#!/usr/bin/env python3
from __future__ import annotations
import json, random
from pathlib import Path
import torch

from lgae_v3.structural_intelligence import (
    ProceduralGraphFactory, StructuralIntelligenceExecutive,
    effective_resistance_candidates, exact_candidate_deltas, candidate_regret,
    uncertainty_calibration,
)
from lgae_v3.reasoning import ConcreteAction
from lgae_v3.benchmark.tasks import StructuralAction

SEED=580
random.seed(SEED); torch.manual_seed(SEED)
factory=ProceduralGraphFactory(latent_dim=8)
train_families=['erdos_renyi','block','tree']
heldout_families=['small_world','scale_free','geometric']
train=factory.sample(train_families, count=18, n_range=(8,12), seed=SEED)
held=factory.sample(heldout_families, count=9, n_range=(8,12), seed=SEED+1)
model=StructuralIntelligenceExecutive(d_max=8, hidden_dim=32, members=3, seed=SEED)

class Outcome:
    def __init__(self,c,d):
        self.candidate=c; self.delta_utility=float(d); self.accepted_by_governor=True; self.metadata={}

def candidate_set(case, k=6):
    cs=[ConcreteAction(StructuralAction.NO_OP, channel='baseline')]
    cs += effective_resistance_candidates(case.graph,case.z,top_k=k)
    # bounded random nonedge diversity so the oracle isn't identical to ER ranking
    existing={tuple(sorted((int(case.graph.src[i]),int(case.graph.dst[i])))) for i in torch.where(case.graph.valid)[0].tolist()}
    rng=random.Random(case.seed+99)
    seen={c.key() for c in cs}; attempts=0
    while len(cs)<1+k+3 and attempts<200:
        attempts+=1
        u,v=rng.sample(range(case.graph.num_nodes),2); u,v=sorted((u,v))
        if (u,v) in existing: continue
        d=float(torch.linalg.vector_norm(case.z[u]-case.z[v]))
        c=ConcreteAction(StructuralAction.ADD_EDGE,{'u':u,'v':v,'weight':1.0,'length':max(d,1e-3)},channel='exploration')
        if c.key() not in seen:
            seen.add(c.key()); cs.append(c)
    return cs

# Exact procedural counterfactual supervision.
for case in train:
    cs=candidate_set(case)
    exact=exact_candidate_deltas(case.graph,case.z,cs)
    model.add_counterfactual_group(case.graph,case.z,[Outcome(c,d) for c,d in zip(cs,exact)])

training=[]
for _ in range(24):
    training.append(model.train_step(batch_states=6, ranking_weight=0.75))

learned_regrets=[]; er_regrets=[]; errors=[]; stds=[]; family_rows=[]
for case in held:
    cs=candidate_set(case)
    exact=exact_candidate_deltas(case.graph,case.z,cs)
    vals=model.predict(case.graph,case.z,cs)
    score_by_key={v.candidate.key():v.score for v in vals}
    predicted=[score_by_key[c.key()] for c in cs]
    lr=candidate_regret(predicted,exact)
    # ER baseline: NOOP vs first ER proposal according to ER prior, evaluated exactly.
    er_scores=[-1e9]*len(cs); er_scores[0]=0.0
    for i,c in enumerate(cs):
        if c.channel=='effective_resistance': er_scores[i]=c.prior_score
    er=candidate_regret(er_scores,exact)
    learned_regrets.append(lr.regret); er_regrets.append(er.regret)
    # calibration of ensemble predicted mean against exact delta
    pred_by_key={v.candidate.key():v for v in vals}
    for c,d in zip(cs,exact):
        if d == float('-inf'): continue
        v=pred_by_key[c.key()]
        errors.append(v.mean_delta_utility-float(d)); stds.append(v.std_delta_utility)
    family_rows.append({'family':case.family,'learned_regret':lr.regret,'effective_resistance_regret':er.regret,'oracle_delta':lr.oracle_delta})

mean_lr=sum(learned_regrets)/len(learned_regrets)
mean_er=sum(er_regrets)/len(er_regrets)
cal=uncertainty_calibration(errors,stds)
checks={
    'train_heldout_families_disjoint': set(train_families).isdisjoint(heldout_families),
    'state_grouped_replay_populated': len(model.replay)==len(train) and model.replay.action_count>len(train),
    'ensemble_members': model.model.members==3,
    'finite_heldout_regret': all(torch.isfinite(torch.tensor(learned_regrets))),
    'epistemic_metrics_finite': all(torch.isfinite(torch.tensor(list(cal.values())))),
    'heldout_cases_present': len(held)==9,
}
scientific_generalization_pass = mean_lr < mean_er
payload={
    'schema':'LGAE_STRUCTURAL_INTELLIGENCE_QUALIFICATION_V5_8_0',
    'status':'PASS' if all(checks.values()) else 'FAIL',
    'scientific_generalization_status':'PASS' if scientific_generalization_pass else 'NOT_YET_QUALIFIED',
    'checks':checks,
    'training':{'families':train_families,'states':len(train),'actions':model.replay.action_count,'last_step':training[-1]},
    'heldout':{'families':heldout_families,'states':len(held),'mean_learned_regret':mean_lr,'mean_effective_resistance_regret':mean_er,'beats_effective_resistance':scientific_generalization_pass,'cases':family_rows},
    'uncertainty_calibration':cal,
    'note':'PASS certifies the v5.8 qualification infrastructure and deterministic held-out evaluation path. Learned structural generalization is claimed only when scientific_generalization_status is PASS.'
}
Path('structural_intelligence_qualification_report.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
raise SystemExit(0 if payload['status']=='PASS' else 1)
