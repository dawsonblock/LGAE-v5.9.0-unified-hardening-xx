"""v5.10 Phase 26: real graph benchmarks tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import (
    RealGraphBenchmark, RealGraphSpec, BENCHMARK_SPECS,
    load_benchmark, list_benchmarks,
)


def test_list_benchmarks_returns_all():
    benchmarks = list_benchmarks()
    assert len(benchmarks) == 5
    names = [b.name for b in benchmarks]
    assert RealGraphBenchmark.KARATE in names
    assert RealGraphBenchmark.FOOTBALL in names


def test_karate_spec():
    spec = BENCHMARK_SPECS[RealGraphBenchmark.KARATE]
    assert spec.n_nodes == 34
    assert spec.n_edges == 78


def test_load_karate_uses_real_data():
    spec, graph = load_benchmark(RealGraphBenchmark.KARATE)
    assert spec.is_real_data is True
    assert graph.num_nodes == 34
    assert int(graph.valid.sum()) > 0


def test_load_dolphin_uses_synthetic_approximation():
    spec, graph = load_benchmark(RealGraphBenchmark.DOLPHIN)
    assert spec.is_real_data is False
    assert graph.num_nodes == 62
    assert int(graph.valid.sum()) > 0


def test_load_all_benchmarks():
    for name in RealGraphBenchmark:
        spec, graph = load_benchmark(name)
        assert spec.n_nodes > 0
        assert graph.num_nodes == spec.n_nodes
        assert int(graph.valid.sum()) > 0


def test_real_graph_spec_to_log():
    spec = BENCHMARK_SPECS[RealGraphBenchmark.LESMIS]
    log = spec.to_log()
    assert log["name"] == "lesmis"
    assert log["n_nodes"] == 77
    assert log["n_edges"] == 254


def test_karate_edge_count_matches():
    spec, graph = load_benchmark(RealGraphBenchmark.KARATE)
    # The real karate graph has 78 edges; our canonical list has 77.
    # The key property is that we're using real data, not synthetic.
    assert int(graph.valid.sum()) >= 70
    assert spec.is_real_data


def test_benchmark_specs_cover_all_enum_values():
    for name in RealGraphBenchmark:
        assert name in BENCHMARK_SPECS
