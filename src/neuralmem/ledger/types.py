"""Type definitions for the Evidence Ledger."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class FeedbackType(str, Enum):
    """Feedback classification for a retrieval record."""

    useful = "useful"
    wrong = "wrong"
    outdated = "outdated"
    sensitive = "sensitive"
    none = "none"


@dataclass
class RetrievalRecord:
    """A single retrieval audit record."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    query: str = ""
    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    composed_context: str = ""
    feedback: FeedbackType = FeedbackType.none
    feedback_note: str = ""
    latency_ms: float = 0.0
