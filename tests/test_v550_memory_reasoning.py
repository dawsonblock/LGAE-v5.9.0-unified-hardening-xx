import json
from pathlib import Path

import torch

from lgae_v3 import (
    EvidenceLedger, EvidenceRecord, StructuralExperienceMemory, MemoryKind,
    ReasoningGraph, ReasoningNode, ReasoningEvidence,
    CandidateGenerator, ConcreteAction, StructuralAction,
    CounterfactualOutcome, make_graph_buffers,
)


def graph_and_z(n=6):
    edges=[(0,1,1.0),(1,2,1.0),(2,3,1.0),(3,4,1.0),(4,5,1.0)]
    g=make_graph_buffers(n, edges, capacity=16)
    z=torch.arange(n*4,dtype=torch.float32).reshape(n,4)/10
    return g,z


def test_evidence_ledger_append_and_tamper_detection(tmp_path):
    g,z=graph_and_z()
    path=tmp_path/'evidence.jsonl'
    ledger=EvidenceLedger(path)
    a=ledger.append(EvidenceRecord('plan',g.state_hash(),{'x':1}))
    b=ledger.append(EvidenceRecord('outcome',g.state_hash(),{'ok':True}))
    assert b['previous_hash']==a['sha256']
    ok,errors=ledger.verify(); assert ok and not errors
    lines=path.read_text().splitlines(); obj=json.loads(lines[0]); obj['record']['payload']['x']=2; lines[0]=json.dumps(obj); path.write_text('\n'.join(lines)+'\n')
    ok,errors=ledger.verify(); assert not ok and any('sha256 mismatch' in e for e in errors)


def test_structural_memory_records_and_retrieves_experience():
    g,z=graph_and_z()
    mem=StructuralExperienceMemory()
    c=ConcreteAction(StructuralAction.ADD_EDGE,{'u':0,'v':5,'weight':1.0,'length':1.0},'test')
    out=CounterfactualOutcome(c,0.4,1.0,1.4,True,'accept',g.state_hash(),g.state_hash(),{})
    mem.record_outcome(g,z,out,evidence_hash='abc',prediction={'mean':0.3,'std':0.1})
    matches=mem.retrieve(g,z,k=3)
    assert matches and matches[0].similarity > 0.99
    prior,count=mem.action_prior(g,z,c)
    assert count >= 1 and prior > 0


def test_reasoning_graph_fanout_reduce_is_typed_and_deterministic():
    def mk(node_id,value):
        return lambda ctx: ReasoningEvidence(node_id,'diagnosis',{'value':value})
    def reduce(ctx):
        deps=ctx['dependencies']
        total=sum(v.payload['value'] for v in deps.values())
        return ReasoningEvidence('reduce','synthesis',{'total':total})
    rg=ReasoningGraph([
        ReasoningNode('a',mk('a',2)), ReasoningNode('b',mk('b',3)),
        ReasoningNode('reduce',reduce,('a','b')),
    ])
    r1=rg.run({'input':1}); r2=rg.run({'input':1})
    assert r1.execution_layers == [['a','b'],['reduce']]
    assert r1.outputs['reduce'].payload['total']==5
    assert r1.run_id==r2.run_id


def test_reasoning_graph_rejects_cycles_and_bad_contract():
    import pytest
    with pytest.raises(ValueError):
        ReasoningGraph([ReasoningNode('a',lambda c:None,('b',)),ReasoningNode('b',lambda c:None,('a',))])
    rg=ReasoningGraph([ReasoningNode('x',lambda c:{'bad':True})])
    with pytest.raises(TypeError): rg.run({})


def test_candidate_exploration_is_bounded_without_pair_materialization():
    n=2000
    edges=[(i,i+1,1.0) for i in range(100)]
    g=make_graph_buffers(n,edges,capacity=256)
    z=torch.randn(n,8)
    gen=CandidateGenerator(max_candidates=32,seed=4)
    out=gen.generate(g,z)
    assert 1 <= len(out) <= 32
    assert out[0].action == StructuralAction.NO_OP
