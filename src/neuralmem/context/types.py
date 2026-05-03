"""Type definitions for the Context Composer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ContextSource(str, Enum):
    """Classification of context sources."""

    memory = "memory"
    web = "web"
    repo = "repo"
    tool_trace = "tool_trace"
    custom = "custom"


@dataclass
class ComposedContext:
    """Result of multi-source context composition."""

    query: str = ""
    sources: dict[ContextSource, list[str]] = field(default_factory=dict)
    composed: str = ""
    token_count: int = 0
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
