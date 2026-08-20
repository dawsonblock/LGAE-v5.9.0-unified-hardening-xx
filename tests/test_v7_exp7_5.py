"""Tests for v7.0-exp5: Real LLM Routing Validation."""
import os
import pytest
import numpy as np

from lgae_v3.experimental.exp7_5 import (
    BackendConfig, MOCK_CONFIG, make_openai_config, BudgetGuard,
    load_prompt, load_all_prompts, get_prompt_hashes, format_prompt,
    make_split, DataSplit,
    run_smoke_test, run_topology_sensitivity_check, run_node_ablation,
    run_targeted_node_ablation,
    run_exp7_5, create_backend_from_config,
    ResponseCache, CachedBackend,
)
from lgae_v3.experimental.exp7_2 import (
    MockModelBackend, ObjectiveWeights, generate_benchmark,
)


class TestBackendConfig:
    """Test backend configuration."""

    def test_mock_config(self):
        assert MOCK_CONFIG.provider == "mock"
        assert MOCK_CONFIG.model_id == "mock-v1"
        assert MOCK_CONFIG.config_hash  # non-empty

    def test_config_hash_stable(self):
        c1 = BackendConfig(provider="openai", model_id="gpt-4o-mini")
        c2 = BackendConfig(provider="openai", model_id="gpt-4o-mini")
        assert c1.config_hash == c2.config_hash

    def test_config_hash_differs(self):
        c1 = BackendConfig(provider="openai", model_id="gpt-4o-mini")
        c2 = BackendConfig(provider="openai", model_id="gpt-4o")
        assert c1.config_hash != c2.config_hash

    def test_compute_dollar_cost(self):
        config = BackendConfig(
            provider="openai", model_id="test",
            input_price_per_mtok=1.0, output_price_per_mtok=2.0,
        )
        cost = config.compute_dollar_cost(input_tokens=1_000_000, output_tokens=500_000)
        assert cost == 2.0  # 1.0 + 1.0

    def test_compute_dollar_cost_with_cache(self):
        config = BackendConfig(
            provider="openai", model_id="test",
            input_price_per_mtok=1.0, output_price_per_mtok=2.0,
            cached_input_price_per_mtok=0.5,
        )
        cost = config.compute_dollar_cost(
            input_tokens=1_000_000, output_tokens=500_000, cached_tokens=400_000,
        )
        # non_cached=600K at 1.0 + cached=400K at 0.5 + output=500K at 2.0
        assert abs(cost - (0.6 + 0.2 + 1.0)) < 0.01

    def test_to_dict(self):
        d = MOCK_CONFIG.to_dict()
        assert "provider" in d
        assert "config_hash" in d


class TestBudgetGuard:
    """Test budget guard."""

    def test_budget_not_exceeded(self):
        guard = BudgetGuard(max_api_calls=100, max_tokens=10000, max_dollar_cost=10.0)
        assert not guard.check()

    def test_budget_exceeded_calls(self):
        guard = BudgetGuard(max_api_calls=5, max_tokens=10000, max_dollar_cost=10.0)
        for _ in range(5):
            guard.record(100, 0.1)
        assert guard.check()

    def test_budget_exceeded_cost(self):
        guard = BudgetGuard(max_api_calls=100, max_tokens=10000, max_dollar_cost=1.0)
        guard.record(100, 2.0)
        assert guard.check()

    def test_budget_summary(self):
        guard = BudgetGuard(max_api_calls=100)
        guard.record(500, 0.5)
        s = guard.summary()
        assert s["n_calls"] == 1
        assert s["n_tokens"] == 500
        assert s["dollar_cost"] == 0.5


class TestPrompts:
    """Test prompt management."""

    def test_load_prompt(self):
        record = load_prompt("planner")
        assert record.role == "planner"
        assert "TASK_INPUT" in record.content
        assert len(record.sha256) == 64

    def test_load_all_prompts(self):
        prompts = load_all_prompts()
        assert len(prompts) == 6
        for role in ["planner", "worker", "researcher", "critic", "verifier", "memory"]:
            assert role in prompts

    def test_prompt_hashes_stable(self):
        h1 = get_prompt_hashes()
        h2 = get_prompt_hashes()
        assert h1 == h2

    def test_format_prompt(self):
        formatted = format_prompt(
            "Task: {TASK_INPUT}\nContext: {UPSTREAM_CONTEXT}",
            "test task",
            "some context",
        )
        assert "test task" in formatted
        assert "some context" in formatted


class TestDataSplit:
    """Test train/calibration/test split."""

    def test_make_split(self):
        split = make_split(n_per_class=50, seed=42)
        assert len(split.train) > 0
        assert len(split.calibration) > 0
        assert len(split.test) > 0
        assert split.total == 300

    def test_split_disjoint(self):
        split = make_split(n_per_class=10, seed=42)
        train_ids = set(t.task_id for t in split.train)
        cal_ids = set(t.task_id for t in split.calibration)
        test_ids = set(t.task_id for t in split.test)
        # No overlap.
        assert train_ids & cal_ids == set()
        assert train_ids & test_ids == set()
        assert cal_ids & test_ids == set()

    def test_split_to_dict(self):
        split = make_split(n_per_class=10, seed=42)
        d = split.to_dict()
        assert d["total"] == 60
        assert "train_size" in d


class TestValidation:
    """Test pre-experiment validation."""

    def test_smoke_test_mock(self):
        backend = MockModelBackend(seed=42)
        result = run_smoke_test(backend)
        assert result.n_roles_tested == 6
        assert result.passed

    def test_topology_sensitivity_mock(self):
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        tasks = generate_benchmark(n_per_class=5, seed=42)
        result = run_topology_sensitivity_check(backend, tasks, weights, n_tasks=10)
        assert result.n_tasks == 10
        assert len(result.quality_diff) == 10

    def test_node_ablation_mock(self):
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        tasks = generate_benchmark(n_per_class=5, seed=42)
        results = run_node_ablation(backend, tasks, weights, n_tasks=10)
        assert len(results) == 4  # researcher, critic, verifier, memory
        for r in results:
            assert r.node in ["researcher", "critic", "verifier", "memory"]


class TestExperimentRunner:
    """Test the full experiment runner."""

    def test_run_exp7_5_mock_smoke(self):
        """Smoke test with mock backend, minimal tasks."""
        result = run_exp7_5(
            backend_config=MOCK_CONFIG,
            n_tasks_per_class=5,
            run_smoke=True,
            run_sensitivity=True,
            run_ablation=True,
            run_main_experiment=True,
        )
        assert result is not None
        assert result.smoke_test.get("passed", False)
        assert len(result.condition_results) == 3
        assert len(result.gates) == 15

    def test_run_exp7_5_no_main(self):
        """Test just validation phases without main experiment."""
        result = run_exp7_5(
            backend_config=MOCK_CONFIG,
            n_tasks_per_class=5,
            run_smoke=True,
            run_sensitivity=True,
            run_ablation=True,
            run_main_experiment=False,
        )
        assert result.smoke_test.get("passed", False)
        assert len(result.node_ablation) == 4


class TestNoSecretLeakage:
    """Verify no secrets leak into artifacts."""

    def test_no_api_key_in_config_dict(self):
        d = MOCK_CONFIG.to_dict()
        for key, val in d.items():
            if isinstance(val, str):
                assert "sk-" not in val
                assert "API_KEY" not in val.upper() or val == "OPENAI_API_KEY_PRESENT"

    def test_no_api_key_in_budget_summary(self):
        guard = BudgetGuard()
        s = guard.summary()
        for key, val in s.items():
            if isinstance(val, str):
                assert "sk-" not in val


class TestResponseCache:
    """Test response caching."""

    def test_cache_hit(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        from lgae_v3.experimental.exp7_2.model_backend import Message, ModelResponse
        msgs = [Message(role="user", content="test")]
        # Put a response.
        resp = ModelResponse(text="hello", tokens_in=10, tokens_out=5, status="SUCCESS")
        cache.put("model", "system", msgs, 100, 0.0, "worker", resp)
        # Get should return cached.
        cached = cache.get("model", "system", msgs, 100, 0.0, "worker")
        assert cached is not None
        assert cached.text == "hello"
        assert cache.n_hits == 1

    def test_cache_miss(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        from lgae_v3.experimental.exp7_2.model_backend import Message
        msgs = [Message(role="user", content="test")]
        cached = cache.get("model", "system", msgs, 100, 0.0, "worker")
        assert cached is None
        assert cache.n_misses == 1

    def test_cached_backend(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        mock = MockModelBackend(seed=42)
        cached = CachedBackend(mock, cache)
        from lgae_v3.experimental.exp7_2.model_backend import Message
        msgs = [Message(role="user", content="What is 2+2?")]
        r1 = cached.generate(role="worker", system_prompt="test", messages=msgs, max_tokens=100, temperature=0.0)
        r2 = cached.generate(role="worker", system_prompt="test", messages=msgs, max_tokens=100, temperature=0.0)
        assert r1.text == r2.text
        assert cache.n_hits == 1

    def test_cache_different_context_different_key(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        from lgae_v3.experimental.exp7_2.model_backend import Message
        msgs1 = [Message(role="user", content="task A")]
        msgs2 = [Message(role="user", content="task B")]
        from lgae_v3.experimental.exp7_2.model_backend import ModelResponse
        cache.put("model", "sys", msgs1, 100, 0.0, "worker", ModelResponse(text="A", status="SUCCESS"))
        # Different context should miss.
        cached = cache.get("model", "sys", msgs2, 100, 0.0, "worker")
        assert cached is None


class TestTargetedAblation:
    """Test targeted node ablation."""

    def test_targeted_ablation_mock(self):
        backend = MockModelBackend(seed=42)
        weights = ObjectiveWeights()
        tasks = generate_benchmark(n_per_class=5, seed=42)
        results = run_targeted_node_ablation(backend, tasks, weights, n_per_family=3)
        assert len(results) == 4  # all 4 nodes
        for r in results:
            assert r.n_tasks > 0
