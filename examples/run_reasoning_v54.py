"""Minimal v5.4 concrete structural-reasoning example."""
import torch
from lgae_v3 import make_graph_buffers
from lgae_v3.reasoning import StructuralReasoningExecutive


torch.manual_seed(54)
graph = make_graph_buffers(
    8,
    [(0,1),(1,2),(2,3),(4,5),(5,6),(6,7)],
    capacity=32,
)
z = torch.randn(8, 16)
reasoner = StructuralReasoningExecutive(d_max=16, hidden_dim=64, max_candidates=32, seed=54)
plan = reasoner.plan(graph, z)

print("selected:", plan.selected.candidate.action.value, plan.selected.candidate.target)
for value in plan.ranked[:5]:
    print(value.candidate.action.value, value.candidate.channel, round(value.score, 4), round(value.std_delta_utility, 4))
