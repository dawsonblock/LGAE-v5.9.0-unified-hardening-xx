#!/usr/bin/env python3
"""Fast v5.5 evidence-grounded structural-reasoning qualification smoke test."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import torch
from lgae_v3 import make_graph_buffers
from lgae_v3.reasoning import StructuralReasoningExecutive, ConcreteAction, CounterfactualOutcome
from lgae_v3.benchmark.tasks import StructuralAction
from lgae_v3.memory import StructuralExperienceMemory
from lgae_v3.evidence import EvidenceLedger, EvidenceRecord
from lgae_v3.reasoning_graph import ReasoningGraph, ReasoningNode, ReasoningEvidence
from lgae_v3.version import VERSION


def main():
    torch.manual_seed(550)
    g=make_graph_buffers(8,[(0,1),(1,2),(2,3),(4,5),(5,6),(6,7)],capacity=32)
    z=torch.randn(8,16)
    memory=StructuralExperienceMemory()
    r=StructuralReasoningExecutive(d_max=16,hidden_dim=64,max_candidates=32,seed=550)
    r.attach_memory(memory)
    p=r.plan(g,z)

    rg=ReasoningGraph([
        ReasoningNode('spectral_proxy', lambda c: ReasoningEvidence('spectral_proxy','diagnosis',{'edge_count':c['graph'].edge_count},1.0,c['graph'].state_hash())),
        ReasoningNode('memory_probe', lambda c: ReasoningEvidence('memory_probe','memory',{'matches':len(c['memory'].retrieve(c['graph'],c['z'],k=2))},1.0,c['graph'].state_hash())),
    ])
    run=rg.run({'graph':g,'z':z,'memory':memory})

    with tempfile.TemporaryDirectory() as td:
        ledger=EvidenceLedger(Path(td)/'evidence.jsonl')
        envelope=ledger.append(EvidenceRecord('qualification',g.state_hash(),{'run_id':run.run_id}))
        ledger_ok,_=ledger.verify()

    payload={
        'schema':'LGAE_REASONING_QUALIFICATION_V5_6_3',
        'version':VERSION,
        'status':'PASS',
        'candidates':p.candidates_considered,
        'selected_action':p.selected.candidate.action.value,
        'selected_channel':p.selected.candidate.channel,
        'finite_scores':all(torch.isfinite(torch.tensor(v.score)).item() for v in p.ranked),
        'no_op_present':any(v.candidate.action.value=='no_op' for v in p.ranked),
        'typed_reasoning_graph':sorted(run.outputs)==['memory_probe','spectral_proxy'],
        'parallel_layer_width':len(run.execution_layers[0]),
        'evidence_ledger_verified':ledger_ok,
        'evidence_hash_present':bool(envelope.get('sha256')),
        'memory_attached':r.memory is memory,
    }
    if not all([payload['finite_scores'],payload['no_op_present'],payload['typed_reasoning_graph'],payload['evidence_ledger_verified'],payload['evidence_hash_present'],payload['memory_attached']]):
        payload['status']='FAIL'
    with open('reasoning_qualification_report.json','w') as f: json.dump(payload,f,indent=2,sort_keys=True)
    print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=='__main__': main()
