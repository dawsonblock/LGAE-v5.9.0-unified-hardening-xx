"""v5.10 Phase 43: typed configuration presets tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import RuntimeConfig, RuntimeMode
from lgae_v3.runtime.runtime_config import (
    research_runtime_config, production_runtime_config, benchmark_runtime_config,
    PRESETS, load_runtime_config,
)


def test_research_preset_defaults():
    cfg = research_runtime_config()
    assert cfg.mode == RuntimeMode.RESEARCH
    assert not cfg.require_signed_receipts
    assert cfg.deterministic_ordering


def test_production_preset_requires_signing_key():
    cfg = production_runtime_config(
        evidence_path="/tmp/evidence.jsonl",
        receipt_path="/tmp/receipts.jsonl",
        signing_key="test-key",
        wal_path="/tmp/wal.jsonl",
    )
    assert cfg.mode == RuntimeMode.PRODUCTION
    assert cfg.require_signed_receipts
    assert cfg.signing_key == "test-key"
    assert cfg.evidence_path == "/tmp/evidence.jsonl"


def test_production_preset_fails_without_signing_key():
    # Production with require_signed_receipts but no signing_key should fail.
    with pytest.raises(ValueError):
        RuntimeConfig(
            mode=RuntimeMode.PRODUCTION,
            evidence_path="/tmp/evidence.jsonl",
            receipt_path="/tmp/receipts.jsonl",
            require_signed_receipts=True,
            signing_key=None,
        )


def test_benchmark_preset_has_larger_candidate_set():
    cfg = benchmark_runtime_config()
    assert cfg.max_candidates == 16
    assert cfg.ensemble_size == 3


def test_load_runtime_config_by_preset():
    cfg = load_runtime_config("research")
    assert cfg.mode == RuntimeMode.RESEARCH


def test_load_runtime_config_with_overrides():
    cfg = load_runtime_config("research", max_candidates=10)
    assert cfg.max_candidates == 10


def test_load_runtime_config_unknown_preset_raises():
    with pytest.raises(ValueError):
        load_runtime_config("bogus")


def test_load_runtime_config_none_preset_returns_default():
    cfg = load_runtime_config(None)
    assert cfg.mode == RuntimeMode.RESEARCH


def test_presets_dict_contains_all_three():
    assert set(PRESETS.keys()) == {"research", "production", "benchmark"}


def test_config_to_summary_roundtrips_mode():
    cfg = research_runtime_config()
    s = cfg.to_summary()
    assert s["mode"] == "research"
    assert "mpc_horizon" in s
    assert "deterministic_ordering" in s
