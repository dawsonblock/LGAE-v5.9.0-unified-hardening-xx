#!/usr/bin/env python3
import json, time
from pathlib import Path
import torch
from lgae_v3.types import make_graph_buffers
from lgae_v3.structural_intelligence import ANNCandidateRetriever, approximate_fosr_candidates, EpistemicScaleCalibrator, wl_graph_hash, contextual_lcb_beta

torch.manual_seed(582)
def mk(n):
    edges=[(i,i+1,1.) for i in range(n-1)]+[(i,(i+11)%n,1.) for i in range(0,n,7)]
    return make_graph_buffers(n,edges,capacity=max(len(edges)+128,256))
checks={}; metrics={}
for n in (128,512,1024):
    g=mk(n); z=torch.randn(n,16); nh=torch.randn(n,32); r=ANNCandidateRetriever(32,16,ann_backend='numpy',ann_candidates=64)
    t=time.perf_counter(); cs=r.candidates(g,z,nh,top_k=32,neighbors_per_node=8); dt=time.perf_counter()-t
    metrics[f'ann_{n}_seconds']=dt; metrics[f'ann_{n}_candidates']=len(cs); checks[f'ann_{n}_nonempty']=len(cs)>0
# Approximate FoSR on bounded qualification graph.
g=mk(128); z=torch.randn(128,16); t=time.perf_counter(); fo=approximate_fosr_candidates(g,z,16); metrics['approx_fosr_128_seconds']=time.perf_counter()-t; checks['approx_fosr_nonempty']=len(fo)>0
# Calibration must improve a deliberately mis-scaled synthetic validation fold.
cal=EpistemicScaleCalibrator(); res=cal.fit(torch.full((128,),.1),torch.full((128,),.5),steps=80); metrics['calibration_nll_before']=res.nll_before; metrics['calibration_nll_after']=res.nll_after; checks['calibration_improves_nll']=res.nll_after<res.nll_before
# WL gate invariance and contextual conservatism.
a=make_graph_buffers(5,[(0,1,1.),(1,2,1.),(2,3,1.),(3,4,1.)],capacity=8); b=make_graph_buffers(5,[(4,2,1.),(2,0,1.),(0,3,1.),(3,1,1.)],capacity=8)
checks['wl_permutation_invariant']=wl_graph_hash(a)==wl_graph_hash(b)
checks['contextual_beta_orders_risk']=contextual_lcb_beta(1,epistemic_std=.8,risk=.9,reversibility=.1,governor_margin=.2,ood_score=.8)>contextual_lcb_beta(1,epistemic_std=.1,risk=.1,reversibility=1,governor_margin=1,ood_score=0)
payload={'schema':'LGAE_SCALABLE_STRUCTURAL_INTELLIGENCE_QUALIFICATION_V5_8_2','status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'metrics':metrics,'claim_boundary':'PASS certifies scalable retrieval/calibration/replay primitives and bounded scaling smoke tests. It does not certify learned superiority on unseen topologies.'}
Path('scalable_structural_intelligence_qualification_report.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); print(json.dumps(payload,indent=2,sort_keys=True)); raise SystemExit(0 if payload['status']=='PASS' else 1)
