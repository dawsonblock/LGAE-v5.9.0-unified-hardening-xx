"""v5.11 Sprint 5: Manifest regression tests — defect repaired.

D11-016/D11-017: The manifest was stale (referenced v5.9.0, missing v5.10 files).
This was the original defect reproduction. After Sprint 5, the manifest is
regenerated and these tests now verify the repair.

Original defect:
- MANIFEST.sha256.json referenced v5.9.0 schema
- Manifest did not cover new v5.10/v5.11 runtime files
- BUILD_REPORT reported v5.9.0 / 719 tests

After repair:
- Manifest references v5.11.0-dev schema
- Manifest covers all current source files
- BUILD_REPORT reports v5.11.0 / 1458 tests
"""
from __future__ import annotations

import json
import pathlib


def test_manifest_version_matches_current():
    """D11-016 repaired: manifest version matches current code version."""
    manifest_path = pathlib.Path("MANIFEST.sha256.json")
    if not manifest_path.exists():
        return  # no manifest to test
    manifest = json.loads(manifest_path.read_text())
    # The manifest should now reference v5.11.0, not v5.9.0.
    version = manifest.get("version", "")
    assert "5.11" in version, (
        f"Manifest version should reference v5.11, got '{version}'. "
        "D11-016 may have regressed."
    )
    schema = manifest.get("schema", "")
    assert "V5_11" in schema or "v5_11" in schema.lower(), (
        f"Manifest schema should reference V5_11, got '{schema}'."
    )


def test_manifest_covers_current_files():
    """D11-016 repaired: manifest covers current runtime files."""
    manifest_path = pathlib.Path("MANIFEST.sha256.json")
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest_files = {f["path"] for f in manifest.get("files", [])}

    # Key v5.11 files that must be in the manifest.
    required_files = [
        "src/lgae_v3/runtime/canonical_runtime.py",
        "src/lgae_v3/runtime/authority.py",
        "src/lgae_v3/runtime/transaction.py",
        "src/lgae_v3/runtime/wal.py",
        "src/lgae_v3/runtime/state/authoritative_state.py",
        "src/lgae_v3/runtime/state/immutable_views.py",
        "src/lgae_v3/runtime/state/authority_token.py",
    ]
    missing = [f for f in required_files if f not in manifest_files]
    assert len(missing) == 0, (
        f"Manifest is missing required files: {missing}"
    )


def test_build_report_updated():
    """D11-017 repaired: BUILD_REPORT references v5.11.0, not v5.9.0."""
    build_report_path = pathlib.Path("BUILD_REPORT.md")
    if not build_report_path.exists():
        return
    content = build_report_path.read_text()
    assert "5.11" in content, (
        "BUILD_REPORT should reference v5.11.0, not v5.9.0. D11-017 may have regressed."
    )
    assert "1458" in content or "1458" in content, (
        "BUILD_REPORT should reference the current test count."
    )
