"""Reproducibility metadata for qualification reports (v5.3.3).

Every qualification report must contain reproducibility metadata so that
two runs can be compared byte-for-byte.  This module provides a single
function that collects all reproducibility-relevant environment info.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch


def _git_commit(repo: Path | None = None) -> str:
    """Get the current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo or Path.cwd(),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _source_tree_hash(repo: Path | None = None) -> str:
    """Hash all Python source files for a deterministic source tree hash."""
    root = repo or Path.cwd()
    h = hashlib.sha256()
    for pattern in ["src/**/*.py", "tests/**/*.py", "scripts/**/*.py"]:
        for f in sorted(root.glob(pattern)):
            if f.is_file():
                h.update(f.relative_to(root).as_posix().encode())
                h.update(b"\0")
                h.update(f.read_bytes())
                h.update(b"\0")
    return h.hexdigest()


def _config_hash(config: Any) -> str:
    """Hash a configuration object deterministically."""
    if config is None:
        return "none"
    if hasattr(config, "__dict__"):
        try:
            data = json.dumps(asdict(config), sort_keys=True, default=str)
        except Exception:
            data = str(config)
    elif isinstance(config, dict):
        data = json.dumps(config, sort_keys=True, default=str)
    else:
        data = str(config)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class ReproducibilityInfo:
    """Reproducibility metadata for a qualification run."""

    seed: int
    python_hash_seed: str
    torch_deterministic: bool
    source_commit: str
    source_tree_sha256: str
    python_version: str
    torch_version: str
    cuda_version: str
    device: str
    config_hash: str
    platform: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def collect(
        cls,
        seed: int = 42,
        config: Any = None,
        repo: Path | None = None,
    ) -> "ReproducibilityInfo":
        """Collect reproducibility metadata from the environment."""
        return cls(
            seed=seed,
            python_hash_seed=os.environ.get("PYTHONHASHSEED", "random"),
            torch_deterministic=torch.are_deterministic_algorithms_enabled(),
            source_commit=_git_commit(repo),
            source_tree_sha256=_source_tree_hash(repo),
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            cuda_version=str(torch.version.cuda) if torch.cuda.is_available() else "none",
            device="cuda" if torch.cuda.is_available() else "cpu",
            config_hash=_config_hash(config),
            platform=platform.platform(),
        )


def qualification_id(info: ReproducibilityInfo) -> str:
    """Compute a deterministic qualification run identifier.

    QID = SHA256(source_tree ∥ config ∥ seed ∥ python_version ∥ torch_version)
    """
    payload = (
        f"{info.source_tree_sha256}"
        f":{info.config_hash}"
        f":{info.seed}"
        f":{info.python_version}"
        f":{info.torch_version}"
    )
    return "lgae-q-" + hashlib.sha256(payload.encode()).hexdigest()[:12]
