"""Reproducibility controls for v6 experiments.

Ensures every experiment is fully reproducible by:
- Seeding all random number generators (Python, NumPy, PyTorch).
- Hashing the configuration to produce a run fingerprint.
- Recording the exact environment (Python version, dependency versions).
- Providing a ``RunFingerprint`` that uniquely identifies a configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json
import random
import sys

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig:
    """Configuration for reproducible experiments."""
    seed: int = 42
    python_hash_seed: int = 0
    torch_deterministic: bool = True
    torch_benchmark: bool = False
    cudnn_deterministic: bool = True
    use_deterministic_algorithms: bool = True

    def to_log(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "python_hash_seed": int(self.python_hash_seed),
            "torch_deterministic": bool(self.torch_deterministic),
            "torch_benchmark": bool(self.torch_benchmark),
            "cudnn_deterministic": bool(self.cudnn_deterministic),
            "use_deterministic_algorithms": bool(self.use_deterministic_algorithms),
        }


def seed_all(seed: int) -> None:
    """Seed all random number generators.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU and CUDA).
    Does NOT set PYTHONHASHSEED (that must be set before interpreter start).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def config_hash(config: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of a configuration dictionary.

    The hash is independent of key insertion order.
    """
    content = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class RunFingerprint:
    """A unique fingerprint for an experimental run.

    This captures everything needed to reproduce a run:
    - The reproducibility config (seeds, deterministic flags).
    - The experiment config (hyperparameters, graph families, etc.).
    - The environment (Python version, key library versions).
    - A combined hash that uniquely identifies the configuration.
    """
    reproducibility: ReproducibilityConfig
    experiment_config: dict[str, Any]
    python_version: str
    torch_version: str
    numpy_version: str
    combined_hash: str

    def to_log(self) -> dict[str, Any]:
        return {
            "reproducibility": self.reproducibility.to_log(),
            "experiment_config": self.experiment_config,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "numpy_version": self.numpy_version,
            "combined_hash": self.combined_hash,
        }

    @classmethod
    def create(
        cls,
        experiment_config: dict[str, Any],
        reproducibility: ReproducibilityConfig | None = None,
    ) -> "RunFingerprint":
        """Create a run fingerprint from an experiment config."""
        repro = reproducibility or ReproducibilityConfig()
        combined = {
            "reproducibility": repro.to_log(),
            "experiment_config": experiment_config,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }
        chash = config_hash(combined)
        return cls(
            reproducibility=repro,
            experiment_config=dict(experiment_config),
            python_version=sys.version,
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            combined_hash=chash,
        )
