"""v5.11 Phase 4: No Python hash() in deterministic runtime paths.

This test scans the runtime source tree for Python hash() usage and
fails if any is found in deterministic paths.

Python hash() is nondeterministic across PYTHONHASHSEED values and
must never be used in:
- Transaction identity
- State hashing
- Authorization binding
- Deterministic replay
- Golden scenario digests

Allow hash() only in:
- Test files
- Non-authoritative utility code
- Hash maps/dicts (Python internal usage)
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


# Files where hash() is allowed (non-deterministic paths or tests).
_HASH_ALLOWED_DIRS = {
    "tests",
    "qualification",
}

# Files where hash() is allowed even in src/ (non-deterministic paths).
_HASH_ALLOWED_FILES = {
    # These files may use hash() for non-identity purposes.
}


def _find_python_files(root: Path) -> list[Path]:
    """Find all Python files in src/lgae_v3/."""
    files = []
    for p in root.rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        files.append(p)
    return files


def _uses_python_hash(source: str) -> list[tuple[int, str]]:
    """Check if source code uses Python hash().

    Returns a list of (line_number, line_content) for each usage.
    Excludes:
    - hash() in string literals
    - hash() in comments
    - .hash() method calls (not Python hash())
    - __hash__ method definitions
    """
    violations = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return violations

    lines = source.split("\n")
    for node in ast.walk(tree):
        # Look for calls to hash() — the builtin function.
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "hash":
                line = lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) else ""
                violations.append((node.lineno, line))
    return violations


class TestNoPythonHashInDeterministicPaths:
    """Ensure no Python hash() is used in deterministic runtime paths."""

    def test_no_hash_in_transaction_module(self):
        """transaction.py must not use Python hash()."""
        root = Path(__file__).resolve().parents[2] / "src" / "lgae_v3" / "runtime"
        source = (root / "transaction.py").read_text()
        violations = _uses_python_hash(source)
        assert violations == [], (
            f"Python hash() found in transaction.py (deterministic path):\n"
            + "\n".join(f"  line {n}: {line}" for n, line in violations)
        )

    def test_no_hash_in_wal_module(self):
        """wal.py must not use Python hash()."""
        root = Path(__file__).resolve().parents[2] / "src" / "lgae_v3" / "runtime"
        source = (root / "wal.py").read_text()
        violations = _uses_python_hash(source)
        assert violations == [], (
            f"Python hash() found in wal.py (deterministic path):\n"
            + "\n".join(f"  line {n}: {line}" for n, line in violations)
        )

    def test_no_hash_in_authority_module(self):
        """authority.py must not use Python hash()."""
        root = Path(__file__).resolve().parents[2] / "src" / "lgae_v3" / "runtime"
        source = (root / "authority.py").read_text()
        violations = _uses_python_hash(source)
        assert violations == [], (
            f"Python hash() found in authority.py (deterministic path):\n"
            + "\n".join(f"  line {n}: {line}" for n, line in violations)
        )

    def test_no_hash_in_determinism_module(self):
        """determinism.py must not use Python hash()."""
        root = Path(__file__).resolve().parents[2] / "src" / "lgae_v3" / "runtime"
        source = (root / "determinism.py").read_text()
        violations = _uses_python_hash(source)
        assert violations == [], (
            f"Python hash() found in determinism.py:\n"
            + "\n".join(f"  line {n}: {line}" for n, line in violations)
        )

    def test_no_hash_in_canonical_runtime(self):
        """canonical_runtime.py must not use Python hash()."""
        root = Path(__file__).resolve().parents[2] / "src" / "lgae_v3" / "runtime"
        source = (root / "canonical_runtime.py").read_text()
        violations = _uses_python_hash(source)
        assert violations == [], (
            f"Python hash() found in canonical_runtime.py:\n"
            + "\n".join(f"  line {n}: {line}" for n, line in violations)
        )

    def test_no_hash_in_state_modules(self):
        """state/ modules must not use Python hash()."""
        root = Path(__file__).resolve().parents[2] / "src" / "lgae_v3" / "runtime" / "state"
        for pyfile in root.glob("*.py"):
            if pyfile.name == "__init__.py":
                continue
            source = pyfile.read_text()
            violations = _uses_python_hash(source)
            assert violations == [], (
                f"Python hash() found in {pyfile.name}:\n"
                + "\n".join(f"  line {n}: {line}" for n, line in violations)
            )

    def test_no_hash_in_curriculum_module(self):
        """curriculum.py must not use Python hash()."""
        root = Path(__file__).resolve().parents[2] / "src" / "lgae_v3" / "runtime"
        source = (root / "curriculum.py").read_text()
        violations = _uses_python_hash(source)
        assert violations == [], (
            f"Python hash() found in curriculum.py:\n"
            + "\n".join(f"  line {n}: {line}" for n, line in violations)
        )

    def test_fiber_delta_raises_on_missing_state_hash(self):
        """FiberDelta.to_hash() raises DeterminismError if snapshot has no state_hash."""
        from lgae_v3.runtime.transaction import FiberDelta
        from lgae_v3.runtime.state.state_errors import DeterminismError

        class SnapshotWithoutHash:
            pass

        delta = FiberDelta(shadow_fiber_snapshot=SnapshotWithoutHash(), action="test")
        with pytest.raises(DeterminismError):
            delta.to_hash()

    def test_fiber_delta_works_with_state_hash(self):
        """FiberDelta.to_hash() works when snapshot has state_hash()."""
        from lgae_v3.runtime.transaction import FiberDelta

        class SnapshotWithHash:
            def state_hash(self) -> str:
                return "abc123"

        delta = FiberDelta(shadow_fiber_snapshot=SnapshotWithHash(), action="test")
        h = delta.to_hash()
        assert h and len(h) > 0
        # Same input → same hash (deterministic).
        delta2 = FiberDelta(shadow_fiber_snapshot=SnapshotWithHash(), action="test")
        assert delta.to_hash() == delta2.to_hash()
