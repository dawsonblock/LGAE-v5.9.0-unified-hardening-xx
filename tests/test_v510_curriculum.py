"""v5.10 Phase 24: curriculum graph generator tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3.runtime import (
    GraphFamily, CurriculumEntry, CurriculumGenerator, generate_graph,
)


def test_graph_family_enum_has_diverse_families():
    families = list(GraphFamily)
    assert len(families) >= 10
    assert GraphFamily.PATH in families
    assert GraphFamily.CYCLE in families
    assert GraphFamily.STAR in families
    assert GraphFamily.GRID in families
    assert GraphFamily.BARBELL in families
    assert GraphFamily.RANDOM_ER in families
    assert GraphFamily.RANDOM_BA in families
    assert GraphFamily.RANDOM_WS in families


def test_generate_path_graph():
    entry = CurriculumEntry(family=GraphFamily.PATH, n_nodes=10, seed=0)
    g = generate_graph(entry)
    assert g.num_nodes == 10
    assert int(g.valid.sum()) == 9  # n-1 edges


def test_generate_cycle_graph():
    entry = CurriculumEntry(family=GraphFamily.CYCLE, n_nodes=10, seed=0)
    g = generate_graph(entry)
    assert g.num_nodes == 10
    assert int(g.valid.sum()) == 10  # n edges


def test_generate_star_graph():
    entry = CurriculumEntry(family=GraphFamily.STAR, n_nodes=10, seed=0)
    g = generate_graph(entry)
    assert g.num_nodes == 10
    assert int(g.valid.sum()) == 9  # n-1 edges


def test_generate_complete_graph():
    entry = CurriculumEntry(family=GraphFamily.COMPLETE, n_nodes=5, seed=0)
    g = generate_graph(entry)
    assert g.num_nodes == 5
    assert int(g.valid.sum()) == 10  # n*(n-1)/2


def test_generate_tree_graph():
    entry = CurriculumEntry(family=GraphFamily.TREE, n_nodes=7, seed=0)
    g = generate_graph(entry)
    assert g.num_nodes == 7
    assert int(g.valid.sum()) == 6  # n-1 edges


def test_generate_random_er_is_deterministic():
    e1 = CurriculumEntry(family=GraphFamily.RANDOM_ER, n_nodes=20, seed=42, params={"p": 0.2})
    e2 = CurriculumEntry(family=GraphFamily.RANDOM_ER, n_nodes=20, seed=42, params={"p": 0.2})
    g1 = generate_graph(e1)
    g2 = generate_graph(e2)
    assert int(g1.valid.sum()) == int(g2.valid.sum())
    assert torch.equal(g1.src, g2.src)


def test_generate_random_ba():
    entry = CurriculumEntry(family=GraphFamily.RANDOM_BA, n_nodes=20, seed=0, params={"m": 2})
    g = generate_graph(entry)
    assert g.num_nodes == 20
    assert int(g.valid.sum()) > 0


def test_generate_random_ws():
    entry = CurriculumEntry(family=GraphFamily.RANDOM_WS, n_nodes=20, seed=0, params={"k": 4, "p": 0.1})
    g = generate_graph(entry)
    assert g.num_nodes == 20
    assert int(g.valid.sum()) > 0


def test_generate_bipartite():
    entry = CurriculumEntry(family=GraphFamily.BIPARTITE, n_nodes=10, seed=0, params={"p": 0.5})
    g = generate_graph(entry)
    assert g.num_nodes == 10
    assert int(g.valid.sum()) > 0


def test_curriculum_generator_generates_entries():
    gen = CurriculumGenerator(seed=42)
    entries = gen.generate_curriculum(n_nodes=15, n_seeds=2)
    assert len(entries) == len(list(GraphFamily)) * 2
    families_in_entries = set(e.family for e in entries)
    assert len(families_in_entries) == len(list(GraphFamily))


def test_curriculum_generator_split_has_disjoint_families():
    gen = CurriculumGenerator(seed=42)
    split = gen.generate_split(n_nodes=15, n_seeds=2)
    train_families = set(e.family for e in split["train"])
    held_out_families = set(e.family for e in split["held_out"])
    assert train_families.isdisjoint(held_out_families)
    assert len(split["train"]) > 0
    assert len(split["held_out"]) > 0


def test_curriculum_entry_family_id_is_unique():
    gen = CurriculumGenerator(seed=42)
    entries = gen.generate_curriculum(n_nodes=10, n_seeds=2)
    ids = set(e.family_id for e in entries)
    # Each (family, seed) pair has a unique family_id.
    assert len(ids) == len(entries)


def test_iter_graphs_yields_valid_buffers():
    gen = CurriculumGenerator(seed=42)
    entries = gen.generate_curriculum(n_nodes=8, families=[GraphFamily.PATH, GraphFamily.CYCLE], n_seeds=1)
    for entry, graph in gen.iter_graphs(entries):
        assert graph.num_nodes == 8
        assert int(graph.valid.sum()) > 0


def test_curriculum_entry_to_log():
    e = CurriculumEntry(family=GraphFamily.PATH, n_nodes=10, seed=0)
    log = e.to_log()
    assert log["family"] == "path"
    assert log["n_nodes"] == 10
    assert log["seed"] == 0
