"""v5.11 Phase 14: verify hash() is not used for determinism.

After Phase 14, curriculum.py uses SHA-256 instead of hash() for
deterministic seed derivation.

This test replaces the v5.10 regression test that documented the defect.
"""
from __future__ import annotations

import inspect

from lgae_v3.runtime.curriculum import CurriculumGenerator


def test_curriculum_does_not_use_hash():
    """The curriculum generator must not use hash() for seed derivation."""
    source = inspect.getsource(CurriculumGenerator)
    # hash() should NOT be used for deterministic seed derivation.
    # We check that the SHA-256 approach is used instead.
    assert "hashlib" in source or "sha256" in source, (
        "Expected curriculum.py to use SHA-256 for seed derivation. "
        "hash() must not be used for determinism-critical code."
    )
    # Verify hash( is not used for seed derivation (it may appear in
    # comments or other contexts, but not for seed calculation).
    # The specific line that used hash() should be gone.
    assert "hash(family.value)" not in source, (
        "curriculum.py still uses hash(family.value) for seed derivation. "
        "This is non-deterministic across PYTHONHASHSEED values."
    )


def test_curriculum_seed_is_deterministic():
    """The same family and seed must produce the same curriculum entry."""
    gen = CurriculumGenerator(seed=42)
    from lgae_v3.runtime.curriculum import GraphFamily
    entries1 = list(gen.generate_curriculum(
        families=[GraphFamily.RANDOM_BA, GraphFamily.RANDOM_WS],
        n_seeds=2,
    ))
    entries2 = list(gen.generate_curriculum(
        families=[GraphFamily.RANDOM_BA, GraphFamily.RANDOM_WS],
        n_seeds=2,
    ))
    # Seeds must match (deterministic).
    for e1, e2 in zip(entries1, entries2):
        assert e1.seed == e2.seed, (
            f"Seed mismatch for {e1.family}: {e1.seed} != {e2.seed}"
        )
