"""Data split for exp7.5.

TRAIN → collect node marginal-value observations
CALIBRATION → choose thresholds / routing operating point
TEST → frozen final comparison

No retuning after TEST results are visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ..exp7_2.benchmark import BenchmarkTask, generate_benchmark, TASK_CLASSES


@dataclass
class DataSplit:
    """Train/Calibration/Test split of benchmark tasks."""
    train: list[BenchmarkTask] = field(default_factory=list)
    calibration: list[BenchmarkTask] = field(default_factory=list)
    test: list[BenchmarkTask] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.train) + len(self.calibration) + len(self.test)

    def to_dict(self) -> dict:
        return {
            "train_size": len(self.train),
            "calibration_size": len(self.calibration),
            "test_size": len(self.test),
            "total": self.total,
            "train_classes": list(set(t.task_class for t in self.train)),
            "test_classes": list(set(t.task_class for t in self.test)),
        }


def make_split(
    n_per_class: int = 50,
    *,
    train_frac: float = 0.4,
    calibration_frac: float = 0.2,
    test_frac: float = 0.4,
    seed: int = 42,
) -> DataSplit:
    """Create a train/calibration/test split.

    Default: 40% train, 20% calibration, 40% test.
    For 50/class → 20 train, 10 calibration, 20 test per class.
    """
    total = n_per_class
    n_train = int(total * train_frac)
    n_cal = int(total * calibration_frac)
    n_test = total - n_train - n_cal

    # Generate all tasks with the frozen seed.
    all_tasks = generate_benchmark(n_per_class=n_per_class, seed=seed)

    # Split per class to maintain distribution.
    train, calibration, test = [], [], []
    by_class: dict[str, list[BenchmarkTask]] = {}
    for t in all_tasks:
        by_class.setdefault(t.task_class, []).append(t)

    rng = np.random.RandomState(seed + 1)
    for cls, tasks in by_class.items():
        indices = list(range(len(tasks)))
        rng.shuffle(indices)

        train_idx = indices[:n_train]
        cal_idx = indices[n_train:n_train + n_cal]
        test_idx = indices[n_train + n_cal:]

        train.extend(tasks[i] for i in train_idx)
        calibration.extend(tasks[i] for i in cal_idx)
        test.extend(tasks[i] for i in test_idx)

    return DataSplit(train=train, calibration=calibration, test=test)
