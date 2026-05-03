"""Poison / injection detection for NeuralMem.

Analyzes memory content for adversarial patterns including prompt injection,
source-dominance attacks, and imperative command abuse.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralmem.core.types import Memory


class RiskLevel(str, Enum):
    """Severity level for flagged memories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class FlaggedMemory:
    """A single flagged memory with its risk assessment."""
    memory_id: str
    risk_level: RiskLevel
    reasons: list[str]
    content_preview: str = ""


@dataclass
class PoisonReport:
    """Aggregate report from a poison detection scan."""
    flagged: list[FlaggedMemory] = field(default_factory=list)
    total_scanned: int = 0
    summary: dict[str, int] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        """True if no memories were flagged."""
        return len(self.flagged) == 0


# ---------------------------------------------------------------------------
# Compiled regex patterns for injection detection
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel]] = [
    (
        "ignore_previous",
        re.compile(
            r"\bignore\s+(all\s+)?previous\s+(instructions?|prompts?|rules?)\b",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
    ),
    (
        "system_prompt",
        re.compile(
            r"\b(system\s+prompt|you\s+are\s+(now|a)\s+|"
            r"new\s+instructions?:|override\s+instructions?)\b",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
    ),
    (
        "you_are_now",
        re.compile(
            r"\byou\s+are\s+now\s+\w+",
            re.IGNORECASE,
        ),
        RiskLevel.MEDIUM,
    ),
    (
        "forget_instructions",
        re.compile(
            r"\bforget\s+(all\s+)?(your\s+)?(instructions?|rules?|training)\b",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
    ),
    (
        "role_hijack",
        re.compile(
            r"\b(pretend|act\s+as\s+if|behave\s+as\s+if|"
            r"your\s+new\s+role)\b",
            re.IGNORECASE,
        ),
        RiskLevel.MEDIUM,
    ),
    (
        "exfiltration",
        re.compile(
            r"\b(send|post|upload|exfiltrate)\s+(all\s+)?"
            r"(memories?|data|information)\s+to\b",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
    ),
]

_IMPERATIVE_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel]] = [
    (
        "delete_all",
        re.compile(
            r"\b(delete|erase|remove|wipe)\s+(all\s+)?(memories?|data|everything)\b",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
    ),
    (
        "forget_everything",
        re.compile(
            r"\bforget\s+(everything|all)\b",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
    ),
    (
        "reset_memory",
        re.compile(
            r"\b(reset|clear|purge)\s+(the\s+)?(memory|memories|database|store)\b",
            re.IGNORECASE,
        ),
        RiskLevel.MEDIUM,
    ),
    (
        "disable_security",
        re.compile(
            r"\b(disable|bypass|circumvent)\s+"
            r"(security|safety|filters?|guardrails?|protections?)\b",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
    ),
]


class PoisonDetector:
    """Detects adversarial memory content via regex patterns.

    Parameters
    ----------
    sensitivity:
        Detection sensitivity level.  ``"low"`` flags only HIGH-risk
        patterns; ``"medium"`` flags MEDIUM+; ``"high"`` flags everything.
    dominance_threshold:
        Fraction of memories from a single source that triggers
        a dominance warning (default 0.6 = 60%).
    """

    VALID_SENSITIVITIES = {"low", "medium", "high"}

    def __init__(
        self,
        sensitivity: str = "medium",
        dominance_threshold: float = 0.6,
    ) -> None:
        if sensitivity not in self.VALID_SENSITIVITIES:
            raise ValueError(
                f"Invalid sensitivity '{sensitivity}'; "
                f"choose from {self.VALID_SENSITIVITIES}"
            )
        self.sensitivity = sensitivity
        self.dominance_threshold = dominance_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, memories: list[Memory]) -> PoisonReport:
        """Scan a list of memories for poison / injection content.

        Parameters
        ----------
        memories:
            List of ``Memory`` objects to analyse.

        Returns
        -------
        PoisonReport
            Contains flagged memories with risk levels and reasons.
        """
        flagged: list[FlaggedMemory] = []
        risk_counts: Counter[str] = Counter()

        # Content-level analysis
        for mem in memories:
            reasons: list[str] = []
            worst_level: RiskLevel | None = None

            # Injection patterns
            for label, pattern, level in _INJECTION_PATTERNS:
                if not self._should_flag(level):
                    continue
                if pattern.search(mem.content):
                    reasons.append(f"injection:{label}")
                    worst_level = self._max_risk(worst_level, level)

            # Imperative command patterns
            for label, pattern, level in _IMPERATIVE_PATTERNS:
                if not self._should_flag(level):
                    continue
                if pattern.search(mem.content):
                    reasons.append(f"imperative:{label}")
                    worst_level = self._max_risk(worst_level, level)

            if reasons and worst_level is not None:
                flagged.append(FlaggedMemory(
                    memory_id=mem.id,
                    risk_level=worst_level,
                    reasons=reasons,
                    content_preview=mem.content[:80],
                ))
                risk_counts[worst_level.value] += 1

        # Dominance analysis (across all memories)
        dominance_flags = self._check_dominance(memories)
        flagged.extend(dominance_flags)
        for f in dominance_flags:
            risk_counts[f.risk_level.value] += 1

        return PoisonReport(
            flagged=flagged,
            total_scanned=len(memories),
            summary=dict(risk_counts),
        )

    def scan_content(self, content: str) -> list[str]:
        """Scan a single text string and return matching pattern labels.

        Useful for pre-screening content before it enters the memory store.
        """
        matches: list[str] = []
        for label, pattern, level in _INJECTION_PATTERNS:
            if not self._should_flag(level):
                continue
            if pattern.search(content):
                matches.append(f"injection:{label}")
        for label, pattern, level in _IMPERATIVE_PATTERNS:
            if not self._should_flag(level):
                continue
            if pattern.search(content):
                matches.append(f"imperative:{label}")
        return matches

    # ------------------------------------------------------------------
    # Dominance detection
    # ------------------------------------------------------------------

    def _check_dominance(
        self, memories: list[Memory]
    ) -> list[FlaggedMemory]:
        """Flag if a single source dominates the memory pool."""
        if not memories:
            return []

        source_counts: Counter[str] = Counter()
        source_ids: dict[str, list[str]] = {}
        for mem in memories:
            src = mem.source or "unknown"
            source_counts[src] += 1
            source_ids.setdefault(src, []).append(mem.id)

        total = len(memories)
        flagged: list[FlaggedMemory] = []

        for src, count in source_counts.items():
            ratio = count / total
            if ratio > self.dominance_threshold and count > 1:
                level = RiskLevel.HIGH if ratio > 0.8 else RiskLevel.MEDIUM
                if not self._should_flag(level):
                    continue
                reason = (
                    f"dominance:source '{src}' has {count}/{total} "
                    f"memories ({ratio:.0%})"
                )
                for mid in source_ids[src][:3]:  # flag first 3
                    flagged.append(FlaggedMemory(
                        memory_id=mid,
                        risk_level=level,
                        reasons=[reason],
                    ))

        return flagged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_flag(self, level: RiskLevel) -> bool:
        """Return True if the given level meets the sensitivity threshold."""
        order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
        threshold = {"low": 2, "medium": 1, "high": 0}
        return order[level] >= threshold[self.sensitivity]

    @staticmethod
    def _max_risk(
        current: RiskLevel | None, candidate: RiskLevel
    ) -> RiskLevel:
        """Return the higher of two risk levels."""
        if current is None:
            return candidate
        order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
        if order[candidate] > order[current]:
            return candidate
        return current
