#!/usr/bin/env python3
"""Verify that release qualification metadata meets promotion requirements.

Fails if:
- qualification_mode != "release"
- tests_failed > 0
- manifest is invalid

Usage:
    python scripts/verify_release_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def verify_release_gate() -> int:
    """Check release promotion gate. Returns 0 if pass, 1 if fail."""
    errors: list[str] = []

    # 1. Check qualification_summary.json exists and has release mode.
    qual_path = ROOT / "qualification_summary.json"
    if not qual_path.exists():
        errors.append("qualification_summary.json not found")
    else:
        with open(qual_path) as f:
            qual = json.load(f)

        mode = qual.get("qualification_mode", "unknown")
        if mode != "release":
            errors.append(
                f"qualification_mode is '{mode}', must be 'release' for promotion. "
                f"Fast-mode metadata cannot be used for release qualification."
            )

        test_results = qual.get("test_results", {})
        failed = test_results.get("failed", -1)
        errors_count = test_results.get("errors", -1)
        if failed != 0:
            errors.append(f"tests_failed={failed}, must be 0")
        if errors_count != 0:
            errors.append(f"tests_errors={errors_count}, must be 0")

        status = qual.get("status", "unknown")
        if status != "QUALIFIED":
            errors.append(f"status is '{status}', must be 'QUALIFIED'")

    # 2. Check manifest is valid.
    manifest_path = ROOT / "MANIFEST.sha256.json"
    if not manifest_path.exists():
        errors.append("MANIFEST.sha256.json not found")
    else:
        # Verify manifest by re-running the check.
        import subprocess
        res = subprocess.run(
            [sys.executable, "scripts/generate_manifest.py", "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            errors.append(f"Manifest verification failed: {res.stdout.strip()}")
        elif "OK" not in res.stdout:
            errors.append(f"Manifest verification unclear: {res.stdout.strip()}")

    # 3. Check release_verification.json exists.
    release_path = ROOT / "release_verification.json"
    if not release_path.exists():
        errors.append("release_verification.json not found")
    else:
        with open(release_path) as f:
            release = json.load(f)
        if release.get("status") != "PASS":
            errors.append(f"release_verification status is '{release.get('status')}', must be 'PASS'")

    # Report.
    if errors:
        print("RELEASE GATE: FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("RELEASE GATE: PASS")
        print(f"  qualification_mode: {qual.get('qualification_mode')}")
        print(f"  tests: {test_results.get('passed')} passed, {test_results.get('failed')} failed")
        print(f"  status: {qual.get('status')}")
        print(f"  manifest: valid")
        return 0


if __name__ == "__main__":
    sys.exit(verify_release_gate())
