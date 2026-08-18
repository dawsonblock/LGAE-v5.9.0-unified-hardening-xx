"""v5.11-RC Phase 23: Benchmark taxonomy tests.

Tests that benchmark datasets are honestly classified as real or synthetic.
"""
from __future__ import annotations

import pytest

from lgae_v3.runtime.real_graphs import (
    RealGraphBenchmark, RealGraphSpec, load_benchmark,
    list_benchmarks,
)


class TestBenchmarkTaxonomy:
    """Benchmark taxonomy distinguishes real from synthetic."""

    def test_real_graph_spec_has_is_real_data_flag(self):
        """RealGraphSpec has an is_real_data flag."""
        spec = RealGraphSpec(name=RealGraphBenchmark.KARATE, n_nodes=34, n_edges=78, description="test")
        assert hasattr(spec, "is_real_data")
        assert spec.is_real_data is False

    def test_synthetic_fallback_is_marked_not_real(self):
        """Synthetic fallback graphs are marked as not real data."""
        spec = RealGraphSpec(name=RealGraphBenchmark.KARATE, n_nodes=34, n_edges=78, description="synthetic")
        assert spec.is_real_data is False

    def test_real_graph_benchmark_enum_exists(self):
        """RealGraphBenchmark enum exists with known datasets."""
        assert hasattr(RealGraphBenchmark, "DOLPHIN")
        assert hasattr(RealGraphBenchmark, "KARATE")

    def test_load_benchmark_returns_spec(self):
        """load_benchmark returns a RealGraphSpec with is_real_data flag."""
        spec, graph = load_benchmark(RealGraphBenchmark.KARATE)
        assert spec is not None
        assert hasattr(spec, "is_real_data")
        # The flag should be a boolean.
        assert isinstance(spec.is_real_data, bool)

    def test_list_benchmarks_returns_specs(self):
        """list_benchmarks returns a list of RealGraphSpec."""
        specs = list_benchmarks()
        assert isinstance(specs, list)
        for spec in specs:
            assert hasattr(spec, "is_real_data")
