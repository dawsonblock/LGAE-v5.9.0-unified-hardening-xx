"""v5.11 Phases 20-24: packaging, manifest, API, and CLI tests.

These tests verify:
- Version and schema constants are consistent
- The manifest schema matches the version
- The public API exports are stable
- The CLI is functional
- `python -m lgae_v3` works
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


class TestVersionAndSchema:
    """Version and schema constants are consistent."""

    def test_version_is_v511_dev(self):
        from lgae_v3.version import VERSION
        assert VERSION == "5.11.0"

    def test_schema_version_is_v511(self):
        from lgae_v3.version import SCHEMA_VERSION
        assert "V5_11_0" in SCHEMA_VERSION

    def test_manifest_schema_is_v511(self):
        from lgae_v3.version import MANIFEST_SCHEMA
        assert "V5_11_0" in MANIFEST_SCHEMA

    def test_all_schema_constants_are_nonempty(self):
        from lgae_v3.version import (
            SCHEMA_VERSION, QUALIFICATION_SCHEMA, CHECKPOINT_SCHEMA,
            SAFE_CHECKPOINT_SCHEMA, RECEIPT_SCHEMA, GRAPH_STATE_SCHEMA,
            MANIFEST_SCHEMA,
        )
        for s in [SCHEMA_VERSION, QUALIFICATION_SCHEMA, CHECKPOINT_SCHEMA,
                  SAFE_CHECKPOINT_SCHEMA, RECEIPT_SCHEMA, GRAPH_STATE_SCHEMA,
                  MANIFEST_SCHEMA]:
            assert s and len(s) > 0

    def test_pyproject_version_matches(self):
        import tomllib
        root = Path(__file__).resolve().parents[2]
        with open(root / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["project"]["version"] == "5.11.0"


class TestPublicAPI:
    """The public API exports are stable and importable."""

    def test_main_package_imports(self):
        import lgae_v3
        assert hasattr(lgae_v3, "ResearchConfig")
        assert hasattr(lgae_v3, "make_graph_buffers")
        assert hasattr(lgae_v3, "MutationDecision")

    def test_runtime_imports(self):
        from lgae_v3.runtime import (
            LGAERuntime, RuntimeConfig, CommitChannel,
            StructuralTransaction, make_graph_transaction,
            WriteAheadLog, replay_committed_transactions,
        )
        assert LGAERuntime is not None
        assert CommitChannel is not None

    def test_contract_imports(self):
        from lgae_v3.runtime.contracts import (
            CANONICAL_PHASE_ORDER,
            ObservationSnapshot, AuthorizationResult, AuthorizationStatus,
            CommitResult, LearningResult,
        )
        assert len(CANONICAL_PHASE_ORDER) == 8

    def test_promotion_imports(self):
        from lgae_v3.runtime.promotion import (
            PromotionLevel, evaluate_promotion, assert_promotion,
        )
        assert PromotionLevel.PRODUCTION.value == 3

    def test_transaction_imports(self):
        from lgae_v3.runtime.transaction import (
            StructuralTransaction, GraphDelta, FiberDelta, GaugeDelta,
            TransactionValidationError, StaleTransactionError,
            AuthorizationBindingError, make_graph_transaction,
        )
        assert StructuralTransaction is not None


class TestCLI:
    """The CLI is functional."""

    def test_cli_help(self):
        from lgae_v3.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_cli_version(self):
        from lgae_v3.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_python_m_lgae_v3_works(self):
        """`python -m lgae_v3 --version` works."""
        result = subprocess.run(
            [sys.executable, "-m", "lgae_v3", "--version"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "5.11.0" in result.stdout

    def test_python_m_lgae_v3_help(self):
        """`python -m lgae_v3 --help` works."""
        result = subprocess.run(
            [sys.executable, "-m", "lgae_v3", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()


class TestManifestIntegrity:
    """The manifest is clean and consistent."""

    def test_manifest_schema_constant_exists(self):
        from lgae_v3.version import MANIFEST_SCHEMA
        assert MANIFEST_SCHEMA == "LGAE_MANIFEST_V5_11_0"

    def test_no_v510_references_in_version(self):
        """No v5.10 references remain in version.py."""
        version_file = Path(__file__).resolve().parents[2] / "src" / "lgae_v3" / "version.py"
        content = version_file.read_text()
        # v5.10 should not appear in the current version constants.
        assert "V5_10_0" not in content, (
            "v5.10 schema references remain in version.py — "
            "the manifest has not been cleaned for v5.11"
        )

    def test_no_v590_references_in_version(self):
        """No v5.9.0 references remain in version.py."""
        version_file = Path(__file__).resolve().parents[2] / "src" / "lgae_v3" / "version.py"
        content = version_file.read_text()
        assert "V5_9_0" not in content, (
            "v5.9.0 schema references remain in version.py — "
            "the manifest has not been cleaned for v5.11"
        )
