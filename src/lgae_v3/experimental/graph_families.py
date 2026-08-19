"""Frozen graph family splits for v6 experimental evaluation.

These splits are FROZEN at v6.0-exp1 creation time. They must never change
for any v6.x experiment. This ensures:

- Train families are used for any learned model training.
- Validation families are used for model selection / hyperparameter tuning.
- Held-out families are NEVER seen during training and are used only for
  the final scientific qualification gate.

The families extend the v5.11 ``curriculum.GraphFamily`` enum with explicit
split assignments. The split is deterministic and reproducible.

Scientific gate requirement:
    LGAE_v6 > LGAE_v5.11 > Strong Baselines?
must be answered on HELD-OUT families, not train families.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..runtime.curriculum import GraphFamily, CurriculumEntry, CurriculumGenerator


@dataclass(frozen=True, slots=True)
class GraphFamilySplit:
    """A frozen train/validation/held-out split of graph families."""
    train: tuple[GraphFamily, ...]
    validation: tuple[GraphFamily, ...]
    held_out: tuple[GraphFamily, ...]
    n_nodes: int
    n_seeds: int
    base_seed: int

    def __post_init__(self) -> None:
        all_families = set(self.train) | set(self.validation) | set(self.held_out)
        train_set = set(self.train)
        val_set = set(self.validation)
        held_out_set = set(self.held_out)
        # No overlap between splits.
        assert not (train_set & val_set), f"train/validation overlap: {train_set & val_set}"
        assert not (train_set & held_out_set), f"train/held_out overlap: {train_set & held_out_set}"
        assert not (val_set & held_out_set), f"validation/held_out overlap: {val_set & held_out_set}"

    @property
    def all_families(self) -> tuple[GraphFamily, ...]:
        return self.train + self.validation + self.held_out

    def to_log(self) -> dict[str, Any]:
        return {
            "train": [f.value for f in self.train],
            "validation": [f.value for f in self.validation],
            "held_out": [f.value for f in self.held_out],
            "n_nodes": int(self.n_nodes),
            "n_seeds": int(self.n_seeds),
            "base_seed": int(self.base_seed),
        }


# ---------------------------------------------------------------------------
# Frozen split definitions.
#
# Train (7 families): path, cycle, star, grid, barbell, random_er, random_ba
# Validation (2 families): random_ws, bipartite
# Held-out (3 families): complete, tree, + one more from the curriculum
#
# These are chosen so:
# - Train covers the main structural regimes (chains, rings, hubs, lattices,
#   barbell bridges, ER random, BA scale-free).
# - Validation covers small-world and bipartite (related but distinct).
# - Held-out covers fully-connected, tree, and one more.
# ---------------------------------------------------------------------------

FROZEN_TRAIN_FAMILIES: tuple[GraphFamily, ...] = (
    GraphFamily.PATH,
    GraphFamily.CYCLE,
    GraphFamily.STAR,
    GraphFamily.GRID,
    GraphFamily.BARBELL,
    GraphFamily.RANDOM_ER,
    GraphFamily.RANDOM_BA,
)

FROZEN_VALIDATION_FAMILIES: tuple[GraphFamily, ...] = (
    GraphFamily.RANDOM_WS,
    GraphFamily.BIPARTITE,
)

FROZEN_HELD_OUT_FAMILIES: tuple[GraphFamily, ...] = (
    GraphFamily.COMPLETE,
    GraphFamily.TREE,
)

# v6.0-exp5.1: TEST-B — untouched external split.
# These families were NEVER used during exp4.2 or exp5 development.
# They are only opened once for the final exp5.1 qualification.
# The old held-out (COMPLETE, TREE) is now TEST-A / development data
# for exp5, since exp5 used it for model selection (linear vs MLP).
FROZEN_TEST_B_FAMILIES: tuple[GraphFamily, ...] = (
    GraphFamily.WHEEL,
    GraphFamily.LADDER,
    GraphFamily.CIRCULAR_LADDER,
    GraphFamily.HYPERCUBE,
)

# The frozen split instance (original exp4.2 split).
FROZEN_SPLIT = GraphFamilySplit(
    train=FROZEN_TRAIN_FAMILIES,
    validation=FROZEN_VALIDATION_FAMILIES,
    held_out=FROZEN_HELD_OUT_FAMILIES,
    n_nodes=20,
    n_seeds=3,
    base_seed=42,
)

# v6.0-exp5.1: Extended split with TEST-B.
# The old held_out is now "test_a" (development data for exp5).
# test_b is the untouched final external split.
FROZEN_SPLIT_V5_1 = GraphFamilySplit(
    train=FROZEN_TRAIN_FAMILIES,
    validation=FROZEN_VALIDATION_FAMILIES,
    held_out=FROZEN_TEST_B_FAMILIES,  # TEST-B replaces old held-out
    n_nodes=20,
    n_seeds=3,
    base_seed=42,
)


class FrozenGraphFamilyRegistry:
    """Registry that generates curriculum entries from the frozen split.

    This is the single source of truth for graph family generation in v6
    experiments. All experiments MUST use this registry (or a split derived
    from it) to ensure reproducibility and comparability.
    """

    def __init__(self, split: GraphFamilySplit | None = None) -> None:
        self.split = split or FROZEN_SPLIT
        self._generator = CurriculumGenerator(seed=self.split.base_seed)

    def test_b_entries(self) -> list[CurriculumEntry]:
        """Curriculum entries for the TEST-B split (v6.0-exp5.1).

        These families were never used during exp4.2 or exp5.
        They are only opened once for final exp5.1 qualification.
        """
        return self._generator.generate_curriculum(
            n_nodes=self.split.n_nodes,
            families=list(FROZEN_TEST_B_FAMILIES),
            n_seeds=self.split.n_seeds,
        )

    def train_entries(self) -> list[CurriculumEntry]:
        """Curriculum entries for the train split."""
        return self._generator.generate_curriculum(
            n_nodes=self.split.n_nodes,
            families=list(self.split.train),
            n_seeds=self.split.n_seeds,
        )

    def validation_entries(self) -> list[CurriculumEntry]:
        """Curriculum entries for the validation split."""
        return self._generator.generate_curriculum(
            n_nodes=self.split.n_nodes,
            families=list(self.split.validation),
            n_seeds=self.split.n_seeds,
        )

    def held_out_entries(self) -> list[CurriculumEntry]:
        """Curriculum entries for the held-out split."""
        return self._generator.generate_curriculum(
            n_nodes=self.split.n_nodes,
            families=list(self.split.held_out),
            n_seeds=self.split.n_seeds,
        )

    def all_entries(self) -> dict[str, list[CurriculumEntry]]:
        """All splits as a dictionary."""
        return {
            "train": self.train_entries(),
            "validation": self.validation_entries(),
            "held_out": self.held_out_entries(),
        }

    def to_log(self) -> dict[str, Any]:
        return {
            "split": self.split.to_log(),
            "n_train": len(self.train_entries()),
            "n_validation": len(self.validation_entries()),
            "n_held_out": len(self.held_out_entries()),
        }


def get_frozen_registry() -> FrozenGraphFamilyRegistry:
    """Get the default frozen graph family registry."""
    return FrozenGraphFamilyRegistry()
