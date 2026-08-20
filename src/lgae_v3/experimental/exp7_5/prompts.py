"""Prompt management for exp7.5.

Loads versioned role prompts and records their hashes for provenance.
Prompts are frozen — never altered between conditions.
"""
from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import Optional

PROMPTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "prompts"
)

ROLE_PROMPTS = {
    "planner": "planner_v1.txt",
    "worker": "worker_v1.txt",
    "researcher": "researcher_v1.txt",
    "critic": "critic_v1.txt",
    "verifier": "verifier_v1.txt",
    "memory": "memory_v1.txt",
}


@dataclass(frozen=True)
class PromptRecord:
    role: str
    filename: str
    content: str
    sha256: str

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "filename": self.filename,
            "sha256": self.sha256,
            "length": len(self.content),
        }


def load_prompt(role: str) -> PromptRecord:
    """Load a versioned role prompt."""
    filename = ROLE_PROMPTS.get(role)
    if not filename:
        raise ValueError(f"Unknown role: {role}")

    path = os.path.join(PROMPTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(path, "r") as f:
        content = f.read()

    sha256 = hashlib.sha256(content.encode()).hexdigest()

    return PromptRecord(
        role=role,
        filename=filename,
        content=content,
        sha256=sha256,
    )


def load_all_prompts() -> dict[str, PromptRecord]:
    """Load all role prompts."""
    return {role: load_prompt(role) for role in ROLE_PROMPTS}


def get_prompt_hashes() -> dict[str, str]:
    """Get SHA256 hashes of all prompts for provenance."""
    return {role: load_prompt(role).sha256 for role in ROLE_PROMPTS}


def format_prompt(template: str, task_input: str, upstream_context: str = "") -> str:
    """Format a prompt template with task input and upstream context."""
    return template.replace("{TASK_INPUT}", task_input).replace(
        "{UPSTREAM_CONTEXT}", upstream_context or "(none)"
    )
