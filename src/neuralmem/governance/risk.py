"""Risk scanning for NeuralMem memory content.

Detects PII, API keys, sensitive words, and stale data in memory content.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

from neuralmem.core.types import Memory


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class RiskFinding:
    level: RiskLevel
    category: str
    description: str
    location: str
    suggestion: str


# --- Compiled regex patterns ---

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

_PHONE_RE = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# API key / secret patterns
_API_KEY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("GitHub Personal Access Token", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    ("OpenAI Secret Key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("AWS Access Key", re.compile(r"\bAKIA[A-Z0-9]{16}\b")),
    ("Generic API Key (long hex)", re.compile(r"\b[A-Fa-f0-9]{40,64}\b")),
    ("Slack Token", re.compile(r"\bxox[bpors]-[A-Za-z0-9-]{10,}\b")),
    ("JWT Token", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
]

# Default sensitive words (configurable)
_DEFAULT_SENSITIVE_WORDS: list[str] = [
    "password",
    "secret",
    "confidential",
    "proprietary",
    "classified",
    "private key",
    "credentials",
]

# Stale data threshold
_STALE_DAYS = 90


def scan_memory(
    memory: Memory,
    sensitive_words: Sequence[str] | None = None,
    stale_days: int = _STALE_DAYS,
) -> list[RiskFinding]:
    """Scan a single memory for risks and return findings."""
    findings: list[RiskFinding] = []
    content = memory.content
    loc = f"memory:{memory.id}"

    # --- PII detection ---
    for match in _EMAIL_RE.finditer(content):
        findings.append(RiskFinding(
            level=RiskLevel.HIGH,
            category="PII",
            description=f"Email address detected: {match.group()}",
            location=loc,
            suggestion="Remove or redact the email address.",
        ))

    for match in _PHONE_RE.finditer(content):
        findings.append(RiskFinding(
            level=RiskLevel.MEDIUM,
            category="PII",
            description=f"Phone number detected: {match.group()}",
            location=loc,
            suggestion="Remove or redact the phone number.",
        ))

    for match in _SSN_RE.finditer(content):
        findings.append(RiskFinding(
            level=RiskLevel.CRITICAL,
            category="PII",
            description=f"SSN detected: {match.group()}",
            location=loc,
            suggestion="Remove immediately. SSNs must never be stored in plain text.",
        ))

    # --- API key detection ---
    for name, pattern in _API_KEY_PATTERNS:
        for match in pattern.finditer(content):
            findings.append(RiskFinding(
                level=RiskLevel.CRITICAL,
                category="API_KEYS",
                description=f"{name} detected: {match.group()[:8]}...",
                location=loc,
                suggestion="Rotate the key immediately and remove from memory.",
            ))

    # --- Sensitive words ---
    words = sensitive_words if sensitive_words is not None else _DEFAULT_SENSITIVE_WORDS
    content_lower = content.lower()
    for word in words:
        if word.lower() in content_lower:
            findings.append(RiskFinding(
                level=RiskLevel.MEDIUM,
                category="SENSITIVE_WORDS",
                description=f"Sensitive word/phrase detected: '{word}'",
                location=loc,
                suggestion="Review whether this content should be stored.",
            ))

    # --- Stale data detection ---
    if memory.expires_at is not None:
        now = datetime.now(timezone.utc)
        if memory.expires_at < now:
            findings.append(RiskFinding(
                level=RiskLevel.HIGH,
                category="STALE_DATA",
                description=f"Memory has expired (expired at {memory.expires_at.isoformat()})",
                location=loc,
                suggestion="Archive or remove this expired memory.",
            ))
    elif memory.importance >= 0.7:
        now = datetime.now(timezone.utc)
        age = now - memory.created_at
        if age > timedelta(days=stale_days):
            findings.append(RiskFinding(
                level=RiskLevel.MEDIUM,
                category="STALE_DATA",
                description=(
                    f"High-importance memory is older than {stale_days} days"
                    f" (age: {age.days}d)"
                ),
                location=loc,
                suggestion="Review whether this memory is still accurate and relevant.",
            ))

    return findings


def scan_batch(
    memories: Sequence[Memory],
    sensitive_words: Sequence[str] | None = None,
    stale_days: int = _STALE_DAYS,
) -> list[RiskFinding]:
    """Scan a batch of memories and return all findings."""
    findings: list[RiskFinding] = []
    for mem in memories:
        findings.extend(scan_memory(mem, sensitive_words=sensitive_words, stale_days=stale_days))
    return findings
