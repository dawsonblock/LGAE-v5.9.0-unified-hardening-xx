import networkx as nx
import torch

from lgae_v3.types import make_graph_buffers
from lgae_v3.sheaf_diffusion import sheaf_laplacian_diffusion
from lgae_v3.transactions import journaled_graph_transaction
from lgae_v3.curvature.ollivier import OllivierNeighborhoodCache, multiscale_ollivier_edge
from lgae_v3.config import LGAEConfig
from lgae_v3.governor import GeometryGovernor
from lgae_v3.mutations import PruneEdge
from lgae_v3.memory import StructuralExperienceMemory, MemoryKind


def test_external_sheaf_maps_are_projected_by_default():
    z = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    src = torch.tensor([0]); dst = torch.tensor([1]); w = torch.tensor([1.0])
    # grossly expansive external map; projection should remove scale 100
    U = torch.tensor([[[100.0, 0.0], [0.0, 0.01]]])
    out = sheaf_laplacian_diffusion(z, src, dst, U, w, num_steps=1, eta=1.0)
    assert torch.isfinite(out).all()
    assert float(torch.linalg.vector_norm(out[1])) <= 1.0001


def test_delta_transaction_rolls_back_only_touched_slot():
    g = make_graph_buffers(4, [(0,1),(1,2),(2,3)], capacity=16)
    before = g.state_hash()
    with journaled_graph_transaction(g) as tx:
        tx.set_slot(1, weight=3.5, length=1/3.5)
    assert g.state_hash() == before
    with journaled_graph_transaction(g) as tx:
        tx.set_slot(1, weight=2.0, length=0.5)
        tx.commit()
    assert g.state_hash() != before
    assert g.version == 1


def test_ollivier_ephemeral_cache_reuses_supports():
    g = nx.cycle_graph(8)
    c = OllivierNeighborhoodCache(g)
    a = multiscale_ollivier_edge(g, 0, 1, radius=2, backend='sinkhorn_log', cache=c)
    nballs = len(c.balls); nhops = len(c.hops)
    b = multiscale_ollivier_edge(g, 0, 1, radius=2, backend='sinkhorn_log', cache=c)
    assert abs(a-b) < 1e-10
    assert len(c.balls) == nballs and len(c.hops) == nhops


def test_prune_can_preserve_two_edge_connectivity():
    # K4 has edge-connectivity 3; deleting one edge drops it to 2 and is allowed at floor 2.
    g = make_graph_buffers(4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)], capacity=12)
    cfg = LGAEConfig(); cfg.audit.min_edge_connectivity_after_prune = 3
    gov = GeometryGovernor(cfg)
    ok, reason = gov._local_mutation_gate(g, PruneEdge(0,1))
    assert not ok and reason == 'edge_connectivity_below_prune_floor'


def test_memory_authority_prefers_evidence_grounded_node():
    m = StructuralExperienceMemory()
    f = (1.0,)*9
    # Force identical id by identical payload/evidence; second lower confidence cannot replace authority.
    n1 = m.add_node(MemoryKind.OUTCOME, f, {'accepted': True, 'x': 1}, evidence_hash='abc', confidence=1.0)
    n2 = m.add_node(MemoryKind.OUTCOME, f, {'accepted': True, 'x': 1}, evidence_hash='abc', confidence=0.1)
    assert n2 is n1
    assert n1.authority > 0.9
