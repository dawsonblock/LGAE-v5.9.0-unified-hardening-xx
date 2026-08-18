"""v5.11 Phase 11: verify production mode is truly fail-closed.

After Phase 11, production mode requires:
- require_signed_receipts=True
- signing_key
- evidence_path (persistent evidence store)
- receipt_path (persistent receipt store)
- deterministic_ordering=True

This test replaces the v5.10 regression test that documented the defect.
"""
from __future__ import annotations

import pytest

from lgae_v3.runtime import RuntimeConfig, RuntimeMode
from lgae_v3.config import ProductionConfig


def test_production_fails_without_evidence_store():
    """Production mode must fail without evidence_path."""
    with pytest.raises(ValueError, match="evidence_path"):
        RuntimeConfig(
            mode=RuntimeMode.PRODUCTION,
            require_signed_receipts=True,
            signing_key="test_key",
            receipt_path="/tmp/receipts.jsonl",
        )


def test_production_fails_without_receipt_store():
    """Production mode must fail without receipt_path."""
    with pytest.raises(ValueError, match="receipt_path"):
        RuntimeConfig(
            mode=RuntimeMode.PRODUCTION,
            require_signed_receipts=True,
            signing_key="test_key",
            evidence_path="/tmp/evidence.jsonl",
        )


def test_production_fails_without_signing_key():
    """Production mode must fail without signing_key."""
    with pytest.raises(ValueError, match="signing_key"):
        RuntimeConfig(
            mode=RuntimeMode.PRODUCTION,
            require_signed_receipts=True,
            evidence_path="/tmp/evidence.jsonl",
            receipt_path="/tmp/receipts.jsonl",
        )


def test_production_fails_without_signed_receipts():
    """Production mode must fail without require_signed_receipts."""
    with pytest.raises(ValueError, match="require_signed_receipts"):
        RuntimeConfig(
            mode=RuntimeMode.PRODUCTION,
            signing_key="test_key",
            evidence_path="/tmp/evidence.jsonl",
            receipt_path="/tmp/receipts.jsonl",
        )


def test_production_succeeds_with_all_requirements():
    """Production mode succeeds when all requirements are met."""
    config = RuntimeConfig(
        mode=RuntimeMode.PRODUCTION,
        require_signed_receipts=True,
        signing_key="test_key",
        evidence_path="/tmp/evidence.jsonl",
        receipt_path="/tmp/receipts.jsonl",
        wal_path="/tmp/wal.jsonl",
    )
    assert config.is_production


def test_production_fails_without_wal():
    """Production mode must fail without wal_path."""
    with pytest.raises(ValueError, match="wal_path"):
        RuntimeConfig(
            mode=RuntimeMode.PRODUCTION,
            require_signed_receipts=True,
            signing_key="test_key",
            evidence_path="/tmp/evidence.jsonl",
            receipt_path="/tmp/receipts.jsonl",
        )
