from __future__ import annotations

import argparse, json, math
import networkx as nx
import torch

from .config import load_config
from .evolution import LGAEEngine
from .types import make_graph_buffers
from .curvature import crosscheck_lly
from .version import VERSION


def _demo_graph(n: int, capacity_factor: float = 2.0):
    g=nx.barbell_graph(max(2,n//3),max(0,n-2*max(2,n//3))) if n>=6 else nx.path_graph(n)
    edges=[(int(u),int(v),1.0) for u,v in g.edges()]
    return g, make_graph_buffers(g.number_of_nodes(),edges,capacity=max(len(edges)+8,int(len(edges)*capacity_factor)))


def _json_safe(x):
    if isinstance(x,dict): return {str(k):_json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [_json_safe(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):
        return "Infinity" if x > 0 else ("-Infinity" if x < 0 else "NaN")
    return x

def _snapshot_dict(s):
    return _json_safe({
        "lambda2":s.lambda2,"operator_discrepancy":s.operator_discrepancy,
        "integral_lly_deficit":s.integral_lly_deficit,"weak_entropic_min":s.weak_entropic_min,
        "bakry_min":s.bakry_min,"cde_residual":s.cde_residual,
        "topology_signature":s.topology_signature,"details":s.details,
    })


# ---------------------------------------------------------------------------
# v5.10 canonical runtime CLI commands
# ---------------------------------------------------------------------------

def _runtime_from_args(args):
    from . import ResearchConfig, LGAERuntime, RuntimeConfig
    cfg = ResearchConfig()
    # Apply minimal safe defaults for CLI usage.
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 0
    cfg.audit.persistent_homology_enabled = False
    cfg.audit.entropic_nodes = 0
    cfg.audit.bakry_nodes = 0
    cfg.audit.cde_nodes = 0
    cfg.audit.exact_lly_top_k = 0
    cfg.audit.orc_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]
    cfg.mutation.curvature_ema_enabled = False
    n = int(getattr(args, "nodes", 8))
    edges = [(i, i + 1) for i in range(n - 1)]
    graph = make_graph_buffers(n, edges, capacity=max(n * 2, 16))
    rt_cfg = RuntimeConfig()
    return LGAERuntime(graph, cfg, runtime_config=rt_cfg)


def _cmd_inspect(args) -> int:
    rt = _runtime_from_args(args)
    snap = rt.snapshot()
    out = {
        "version": VERSION,
        "phase": "v5.10-canonical-runtime",
        "graph": {
            "num_nodes": int(rt.engine.graph.num_nodes),
            "num_edges": int(rt.engine.graph.valid.sum().item()),
            "version": int(rt.engine.graph.version),
            "state_hash": rt.engine.graph.state_hash(),
        },
        "authority_hash": rt.authority_hash,
        "generation": int(rt.generation),
        "snapshot": snap.to_summary(),
        "boundary": rt.boundary.to_summary(),
        "governance": DEFAULT_AUTHORITY_POLICY_SUMMARY,
    }
    print(json.dumps(_json_safe(out), indent=2, default=str))
    return 0


def _cmd_diagnose(args) -> int:
    from .runtime import DiagnosticCascade, DiagnosticEscalationPolicy, DiagnosticLevel
    from .mutations import MutationAuthorityLevel
    rt = _runtime_from_args(args)
    policy = DiagnosticEscalationPolicy()
    level = policy.level_for(
        risk=float(getattr(args, "risk", 0.0)),
        uncertainty=float(getattr(args, "uncertainty", 0.0)),
        disagreement=float(getattr(args, "disagreement", 0.0)),
        authority=MutationAuthorityLevel(getattr(args, "authority", "reversible")),
    )
    out = {
        "version": VERSION,
        "selected_level": int(level),
        "level_name": level.name,
        "authority": getattr(args, "authority", "reversible"),
        "policy": {
            "risk_l1": policy.risk_l1, "risk_l2": policy.risk_l2, "risk_l3": policy.risk_l3,
        },
        "graph_state_hash": rt.engine.graph.state_hash(),
    }
    print(json.dumps(_json_safe(out), indent=2, default=str))
    return 0


def _cmd_propose(args) -> int:
    from .runtime import build_candidate_union
    from .structural_intelligence import fosr_candidates, effective_resistance_candidates, forman_flow_candidates
    rt = _runtime_from_args(args)
    z = rt.engine.fibers.latent
    state_id = rt.engine.graph.state_hash()
    union = build_candidate_union(
        state_id,
        channels={
            "fosr": fosr_candidates(rt.engine.graph, z, top_k=int(getattr(args, "top_k", 8))),
            "er": effective_resistance_candidates(rt.engine.graph, z, top_k=int(getattr(args, "top_k", 8))),
            "forman": forman_flow_candidates(rt.engine.graph, z, top_k=int(getattr(args, "top_k", 8))),
        },
    )
    out = {
        "version": VERSION,
        "state_id": state_id,
        "candidate_count": union.size,
        "channel_counts": union.channel_counts(),
        "candidates": [
            {"id": c.id[:12], "action": c.action.value, "channel": c.channel, "target": c.target}
            for c in union.candidates()[: int(getattr(args, "top_k", 8)) + 1]
        ],
    }
    print(json.dumps(_json_safe(out), indent=2, default=str))
    return 0


def _cmd_step(args) -> int:
    rt = _runtime_from_args(args)
    res = rt.step()
    out = {
        "version": VERSION,
        "step": int(res.step),
        "committed": bool(res.committed),
        "authority_hash_after": res.snapshot_after.authority_hash,
        "phases": res.phases,
        "result": res.to_log(),
    }
    print(json.dumps(_json_safe(out), indent=2, default=str))
    return 0


def _cmd_run(args) -> int:
    rt = _runtime_from_args(args)
    steps = int(getattr(args, "steps", 5))
    metrics_path = getattr(args, "metrics", None)
    sink = None
    if metrics_path:
        from .runtime import MetricsSink
        sink = MetricsSink(metrics_path)
    results = []
    for _ in range(steps):
        res = rt.step()
        if sink is not None:
            for ev in rt.events():
                sink.record_event(ev)
        results.append({
            "step": int(res.step), "committed": bool(res.committed),
            "authority_hash_after": res.snapshot_after.authority_hash,
        })
    out = {
        "version": VERSION,
        "steps": steps,
        "results": results,
        "final_authority_hash": rt.authority_hash,
        "final_generation": int(rt.generation),
    }
    if sink is not None:
        out["metrics"] = sink.snapshot()
        sink.close()
    print(json.dumps(_json_safe(out), indent=2, default=str))
    return 0


def _cmd_qualify(args) -> int:
    from .governance import DEFAULT_REGISTRY
    rt = _runtime_from_args(args)
    # Run invariants against a state-like wrapper.
    class _State:
        graph = rt.engine.graph
        engine = rt.engine
        receipt_path = getattr(rt, "_receipt_path", None)
    results = DEFAULT_REGISTRY.check_all(_State())
    out = {
        "version": VERSION,
        "invariants": [r.to_log() for r in results],
        "all_blocking_passed": all(r.passed for r in results if r.severity.value == "blocking"),
        "authority_hash": rt.authority_hash,
    }
    print(json.dumps(_json_safe(out), indent=2, default=str))
    return 0


# Lazy import to avoid circular dependency at module load.
def _get_authority_summary():
    try:
        from .governance import DEFAULT_AUTHORITY_POLICY
        return DEFAULT_AUTHORITY_POLICY.to_summary()
    except Exception:
        return {}


DEFAULT_AUTHORITY_POLICY_SUMMARY = {}


def main(argv=None):
    global DEFAULT_AUTHORITY_POLICY_SUMMARY
    DEFAULT_AUTHORITY_POLICY_SUMMARY = _get_authority_summary()
    p=argparse.ArgumentParser(prog="lgae-v3")
    p.add_argument("--version", action="version", version=VERSION)
    p.add_argument("--config",default=None)
    sub=p.add_subparsers(dest="cmd",required=True)

    # Legacy commands
    d=sub.add_parser("demo"); d.add_argument("--nodes",type=int,default=10); d.add_argument("--steps",type=int,default=4)
    q=sub.add_parser("qualify-lly"); q.add_argument("--graph",choices=["path","cycle","complete"],default="cycle"); q.add_argument("--nodes",type=int,default=4)

    # v5.10 runtime commands
    common = {"--nodes": {"type": int, "default": 8}}

    insp = sub.add_parser("inspect")
    insp.add_argument("--nodes", type=int, default=8)

    diag = sub.add_parser("diagnose")
    diag.add_argument("--nodes", type=int, default=8)
    diag.add_argument("--risk", type=float, default=0.0)
    diag.add_argument("--uncertainty", type=float, default=0.0)
    diag.add_argument("--disagreement", type=float, default=0.0)
    diag.add_argument("--authority", choices=["reversible", "structural", "high_impact", "irreversible"], default="reversible")

    prop = sub.add_parser("propose")
    prop.add_argument("--nodes", type=int, default=8)
    prop.add_argument("--top_k", type=int, default=8)

    stp = sub.add_parser("step")
    stp.add_argument("--nodes", type=int, default=8)

    run = sub.add_parser("run")
    run.add_argument("--nodes", type=int, default=8)
    run.add_argument("--steps", type=int, default=5)
    run.add_argument("--metrics", type=str, default=None)

    qual = sub.add_parser("qualify")
    qual.add_argument("--nodes", type=int, default=8)

    args=p.parse_args(argv); cfg=load_config(args.config)

    if args.cmd=="qualify-lly":
        g={"path":nx.path_graph,"cycle":nx.cycle_graph,"complete":nx.complete_graph}[args.graph](args.nodes)
        print(json.dumps(crosscheck_lly(g),indent=2,default=str)); return 0

    if args.cmd=="demo":
        g,buffers=_demo_graph(args.nodes)
        eng=LGAEEngine(buffers,cfg)
        for _ in range(args.steps): eng.diffuse_(eta=0.02)
        fiber=eng.fiber_tick()
        mutation=eng.propose_midpoint_edge()
        result=eng.evaluate_and_maybe_commit(mutation) if mutation else None
        out={"version":VERSION,"nodes":buffers.num_nodes,"edges_before":len(g.edges()),"capacity_mean":float(fiber["capacity"].float().mean()),"audit":_snapshot_dict(eng.audit())}
        if result:
            out["mutation"]={"decision":result.decision.value,"reasons":result.reasons,"metadata":result.metadata}
        print(json.dumps(out,indent=2,default=str)); return 0

    # v5.10 runtime commands
    if args.cmd == "inspect": return _cmd_inspect(args)
    if args.cmd == "diagnose": return _cmd_diagnose(args)
    if args.cmd == "propose": return _cmd_propose(args)
    if args.cmd == "step": return _cmd_step(args)
    if args.cmd == "run": return _cmd_run(args)
    if args.cmd == "qualify": return _cmd_qualify(args)

    p.error(f"unknown command: {args.cmd}")
    return 2

if __name__=="__main__": raise SystemExit(main())
