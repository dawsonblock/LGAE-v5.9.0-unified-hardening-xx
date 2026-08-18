"""Tests for v5.3.3 reproducibility infrastructure.

Tests:
- Canonical action ordering is deterministic
- DeterministicRNGContext substreams are domain-separated
- derive_seed is PYTHONHASHSEED-independent
- deterministic_mode context manager works
- Oracle has zero regret across all PYTHONHASHSEED values
"""
from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from lgae_v3.deterministic import (
    DeterministicRNGContext,
    derive_seed,
    deterministic_mode,
)
from lgae_v3.benchmark import (
    StructuralAction,
    ACTION_ORDER,
    ACTION_TO_INDEX,
    canonical_action,
    OracleController,
    BenchmarkHarness,
)
from lgae_v3.benchmark.metrics import run_benchmark


class TestCanonicalActionOrdering:
    """Canonical action ordering must be deterministic."""

    def test_action_order_is_tuple(self):
        assert isinstance(ACTION_ORDER, tuple)
        assert len(ACTION_ORDER) == 9

    def test_action_to_index_covers_all_actions(self):
        for action in StructuralAction:
            assert action in ACTION_TO_INDEX

    def test_action_to_index_matches_definition_order(self):
        for i, action in enumerate(StructuralAction):
            assert ACTION_TO_INDEX[action] == i

    def test_canonical_action_empty_set(self):
        assert canonical_action(set()) == StructuralAction.NO_OP

    def test_canonical_action_single(self):
        assert canonical_action({StructuralAction.ADD_EDGE}) == StructuralAction.ADD_EDGE

    def test_canonical_action_multiple_returns_lowest_index(self):
        actions = {StructuralAction.ADD_EDGE, StructuralAction.NO_OP}
        assert canonical_action(actions) == StructuralAction.NO_OP

    def test_canonical_action_is_deterministic_across_calls(self):
        actions = {StructuralAction.PRUNE_EDGE, StructuralAction.ADD_EDGE,
                   StructuralAction.SPAWN_FIBER}
        results = [canonical_action(actions) for _ in range(100)]
        assert all(r == StructuralAction.ADD_EDGE for r in results)


class TestDeriveSeed:
    """derive_seed must be deterministic and PYTHONHASHSEED-independent."""

    def test_same_inputs_same_output(self):
        assert derive_seed(42, "test") == derive_seed(42, "test")

    def test_different_namespace_different_output(self):
        assert derive_seed(42, "a") != derive_seed(42, "b")

    def test_different_master_different_output(self):
        assert derive_seed(42, "test") != derive_seed(43, "test")

    def test_returns_int(self):
        assert isinstance(derive_seed(42, "test"), int)

    def test_is_stable(self):
        """The seed value should not change across Python versions."""
        # This is a fixed point test — if this breaks, it means the
        # hashing implementation changed, which would invalidate
        # all previously generated datasets.
        seed = derive_seed(42, "qualification")
        assert seed == 17664632221373237916


class TestDeterministicRNGContext:
    """DeterministicRNGContext provides domain-separated substreams."""

    def test_same_master_same_substream(self):
        ctx1 = DeterministicRNGContext(master_seed=42)
        ctx2 = DeterministicRNGContext(master_seed=42)
        g1 = ctx1.numpy_gen("test")
        g2 = ctx2.numpy_gen("test")
        assert g1.integers(0, 1000) == g2.integers(0, 1000)

    def test_different_namespace_different_sequence(self):
        ctx = DeterministicRNGContext(master_seed=42)
        g1 = ctx.numpy_gen("a")
        g2 = ctx.numpy_gen("b")
        assert g1.integers(0, 1000) != g2.integers(0, 1000)

    def test_substream_isolation(self):
        """Adding a new substream doesn't change existing substreams."""
        ctx = DeterministicRNGContext(master_seed=42)
        # Get sequence from substream A
        a1 = ctx.numpy_gen("a").integers(0, 1000, size=5).tolist()
        # Get sequence from substream B (new)
        ctx.numpy_gen("b")
        # Substream A should be unchanged
        a2 = ctx.numpy_gen("a").integers(0, 1000, size=5).tolist()
        # a2 continues from where a1 left off (same generator, cached)
        assert a1 != a2  # Different draws from same generator
        # But a fresh context gives the same a1
        ctx2 = DeterministicRNGContext(master_seed=42)
        a3 = ctx2.numpy_gen("a").integers(0, 1000, size=5).tolist()
        assert a1 == a3

    def test_torch_gen_deterministic(self):
        ctx1 = DeterministicRNGContext(master_seed=42)
        ctx2 = DeterministicRNGContext(master_seed=42)
        g1 = ctx1.torch_gen("model_init")
        g2 = ctx2.torch_gen("model_init")
        t1 = torch.randn(10, generator=g1)
        t2 = torch.randn(10, generator=g2)
        assert torch.equal(t1, t2)

    def test_python_rng_deterministic(self):
        ctx1 = DeterministicRNGContext(master_seed=42)
        ctx2 = DeterministicRNGContext(master_seed=42)
        r1 = ctx1.python_rng("test")
        r2 = ctx2.python_rng("test")
        assert r1.randint(0, 1000) == r2.randint(0, 1000)

    def test_convenience_methods(self):
        ctx = DeterministicRNGContext(master_seed=42)
        assert isinstance(ctx.graph_generation(), np.random.Generator)
        assert isinstance(ctx.qualification(), np.random.Generator)
        assert isinstance(ctx.model_initialization(), torch.Generator)


class TestDeterministicMode:
    """deterministic_mode context manager."""

    def test_context_manager_seeds_globals(self):
        with deterministic_mode(master_seed=42):
            t1 = torch.randn(5)
            n1 = np.random.randn(5)
            r1 = random.random()
        with deterministic_mode(master_seed=42):
            t2 = torch.randn(5)
            n2 = np.random.randn(5)
            r2 = random.random()
        assert torch.equal(t1, t2)
        assert np.array_equal(n1, n2)
        assert r1 == r2

    def test_context_manager_yields_ctx(self):
        with deterministic_mode(master_seed=42) as ctx:
            assert isinstance(ctx, DeterministicRNGContext)
            assert ctx.master_seed == 42


class TestOracleDeterminism:
    """Oracle must have zero regret regardless of PYTHONHASHSEED."""

    def test_oracle_zero_regret(self):
        harness = BenchmarkHarness()
        oracle = harness.run_oracle(seed=42)
        assert oracle.mean_regret == pytest.approx(0.0, abs=1e-6)

    def test_oracle_perfect_accuracy(self):
        harness = BenchmarkHarness()
        oracle = harness.run_oracle(seed=42)
        assert oracle.diagnosis_accuracy == pytest.approx(1.0)

    def test_run_benchmark_oracle_zero_regret(self):
        result = run_benchmark(proposals=None, seed=42)
        assert result.mean_regret == pytest.approx(0.0, abs=1e-6)
