"""Task representation features for exp7.3.

LGAE does NOT receive task labels. It receives structural features
derived from the task input text — token count, question structure,
complexity indicators. These features let LGAE learn task-specific
routing without cheating.

The features are intentionally NOT the task class label. They are
measurements any system could compute from the input text.
"""
from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskFeatures:
    """Structural features of a task input — no task labels."""

    # Text structure
    n_tokens: int = 0           # word count
    n_chars: int = 0            # character count
    avg_word_length: float = 0.0
    n_sentences: int = 0

    # Question structure
    has_question_mark: bool = False
    has_code_keywords: bool = False
    has_debug_keywords: bool = False
    has_research_keywords: bool = False
    has_verify_keywords: bool = False
    has_memory_keywords: bool = False
    has_reasoning_keywords: bool = False

    # Complexity indicators
    complexity_score: float = 0.0   # 0=simple, 1=complex
    estimated_difficulty: float = 0.0

    # Routing hints (derived, not labels)
    suggests_research: bool = False
    suggests_verification: bool = False
    suggests_memory: bool = False
    suggests_planning: bool = False
    suggests_critic: bool = False

    def to_vector(self) -> list[float]:
        """Convert to a feature vector for learning."""
        return [
            self.n_tokens / 50.0,           # normalized token count
            self.n_chars / 200.0,           # normalized char count
            self.avg_word_length / 10.0,    # normalized
            self.n_sentences / 5.0,         # normalized
            1.0 if self.has_question_mark else 0.0,
            1.0 if self.has_code_keywords else 0.0,
            1.0 if self.has_debug_keywords else 0.0,
            1.0 if self.has_research_keywords else 0.0,
            1.0 if self.has_verify_keywords else 0.0,
            1.0 if self.has_memory_keywords else 0.0,
            1.0 if self.has_reasoning_keywords else 0.0,
            self.complexity_score,
            self.estimated_difficulty,
            1.0 if self.suggests_research else 0.0,
            1.0 if self.suggests_verification else 0.0,
            1.0 if self.suggests_memory else 0.0,
            1.0 if self.suggests_planning else 0.0,
            1.0 if self.suggests_critic else 0.0,
        ]

    def to_dict(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "n_chars": self.n_chars,
            "avg_word_length": round(self.avg_word_length, 2),
            "n_sentences": self.n_sentences,
            "has_question_mark": self.has_question_mark,
            "has_code_keywords": self.has_code_keywords,
            "has_debug_keywords": self.has_debug_keywords,
            "has_research_keywords": self.has_research_keywords,
            "has_verify_keywords": self.has_verify_keywords,
            "has_memory_keywords": self.has_memory_keywords,
            "has_reasoning_keywords": self.has_reasoning_keywords,
            "complexity_score": round(self.complexity_score, 3),
            "estimated_difficulty": round(self.estimated_difficulty, 3),
            "suggests_research": self.suggests_research,
            "suggests_verification": self.suggests_verification,
            "suggests_memory": self.suggests_memory,
            "suggests_planning": self.suggests_planning,
            "suggests_critic": self.suggests_critic,
        }


# Keyword sets for feature extraction (NOT task labels — text analysis)
CODE_KEYWORDS = {"code", "function", "bug", "debug", "snippet", "program", "script", "compile", "syntax"}
DEBUG_KEYWORDS = {"debug", "bug", "fix", "error", "trace", "stack", "crash", "fault"}
RESEARCH_KEYWORDS = {"research", "synthesize", "sources", "information", "gather", "analyze", "summary", "comprehensive"}
VERIFY_KEYWORDS = {"verify", "check", "validate", "confirm", "assumption", "evidence", "proof"}
MEMORY_KEYWORDS = {"recall", "memory", "context", "previous", "stored", "remember", "history"}
REASONING_KEYWORDS = {"solve", "reason", "logical", "steps", "multi-step", "derive", "infer", "deduce"}


def extract_features(task_input: str) -> TaskFeatures:
    """Extract structural features from a task input string.

    These features are derived from text analysis, NOT from task labels.
    Any system could compute these from the input text.
    """
    text = task_input.lower()
    words = text.split()
    n_tokens = len(words)
    n_chars = len(task_input)

    # Sentence count (rough).
    n_sentences = max(1, text.count(".") + text.count("?") + text.count("!"))

    avg_word_length = sum(len(w) for w in words) / max(n_tokens, 1)

    # Keyword detection.
    word_set = set(words)
    has_code = bool(word_set & CODE_KEYWORDS)
    has_debug = bool(word_set & DEBUG_KEYWORDS)
    has_research = bool(word_set & RESEARCH_KEYWORDS)
    has_verify = bool(word_set & VERIFY_KEYWORDS)
    has_memory = bool(word_set & MEMORY_KEYWORDS)
    has_reasoning = bool(word_set & REASONING_KEYWORDS)

    has_question = "?" in task_input

    # Complexity score: based on text length and keyword density.
    complexity = min(1.0, n_tokens / 30.0)
    keyword_density = sum([
        has_code, has_debug, has_research, has_verify,
        has_memory, has_reasoning,
    ]) / 6.0
    complexity = (complexity * 0.5 + keyword_density * 0.5)

    # Estimated difficulty: more keywords and longer text = harder.
    difficulty = min(1.0, (n_tokens / 25.0) * 0.4 + keyword_density * 0.6)

    # Routing hints — derived from keywords, NOT from task labels.
    suggests_research = has_research or (complexity > 0.6 and not has_code)
    suggests_verification = has_verify or has_debug or has_code
    suggests_memory = has_memory
    suggests_planning = has_reasoning or complexity > 0.7
    suggests_critic = has_debug or has_code or complexity > 0.6

    return TaskFeatures(
        n_tokens=n_tokens,
        n_chars=n_chars,
        avg_word_length=avg_word_length,
        n_sentences=n_sentences,
        has_question_mark=has_question,
        has_code_keywords=has_code,
        has_debug_keywords=has_debug,
        has_research_keywords=has_research,
        has_verify_keywords=has_verify,
        has_memory_keywords=has_memory,
        has_reasoning_keywords=has_reasoning,
        complexity_score=complexity,
        estimated_difficulty=difficulty,
        suggests_research=suggests_research,
        suggests_verification=suggests_verification,
        suggests_memory=suggests_memory,
        suggests_planning=suggests_planning,
        suggests_critic=suggests_critic,
    )


def features_to_topology_hints(features: TaskFeatures) -> dict:
    """Convert task features into topology weight hints.

    These are SOFT hints — LGAE can use them to bias its topology
    decisions, but they are not hard rules. The hints are derived
    from text analysis, not task labels.
    """
    hints = {
        "research_weight": 1.0 + (0.5 if features.suggests_research else -0.3),
        "critic_weight": 1.0 + (0.5 if features.suggests_critic else -0.2),
        "verifier_weight": 1.0 + (0.3 if features.suggests_verification else 0.0),
        "memory_weight": 1.0 + (0.5 if features.suggests_memory else -0.3),
        "planner_weight": 1.0 + (0.3 if features.suggests_planning else 0.0),
    }
    # Clamp to reasonable range.
    for key in hints:
        hints[key] = max(0.1, min(2.0, hints[key]))
    return hints
