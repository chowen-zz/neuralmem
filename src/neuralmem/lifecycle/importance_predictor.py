"""Importance predictor — estimate future value of a memory.

Uses lightweight heuristics (no LLM required) to predict how likely
a memory is to be useful in the future.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neuralmem.core.types import Memory

_logger = logging.getLogger(__name__)


def predict_importance(
    memory: Memory,
    access_history: list[datetime],
    query_keywords: list[str] | None = None,
) -> float:
    """Predict future importance of a memory on a 0.0–1.0 scale.

    Factors:

    1. **Recency** — newer memories score higher.
    2. **Access frequency** — frequently accessed memories are more valuable.
    3. **Content richness** — longer, entity-dense memories score higher.
    4. **Query alignment** — if query keywords are provided, memories
       containing them score higher.

    Parameters
    ----------
    memory:
        The memory to evaluate.
    access_history:
        Past access timestamps.
    query_keywords:
        Optional current query context keywords.

    Returns
    -------
    Predicted importance in ``[0.0, 1.0]``.
    """
    now = datetime.now(timezone.utc)
    scores: list[float] = []

    # 1. Recency — exponential dropoff with age
    age_days = max(0.0, (now - memory.created_at).total_seconds() / 86400.0)
    recency_score = max(0.0, 1.0 - age_days / 365.0)  # Linear over 1 year
    scores.append(recency_score * 0.25)

    # 2. Access frequency — saturating at 10 accesses
    access_count = len(access_history)
    access_score = min(access_count / 10.0, 1.0)
    scores.append(access_score * 0.30)

    # 3. Content richness — length + entity count
    content_len = len(memory.content)
    length_score = min(content_len / 1000.0, 1.0)
    entity_count = len(memory.entity_ids)
    entity_score = min(entity_count / 5.0, 1.0)
    richness_score = (length_score + entity_score) / 2.0
    scores.append(richness_score * 0.25)

    # 4. Query alignment
    if query_keywords:
        content_lower = memory.content.lower()
        matches = sum(1 for kw in query_keywords if kw.lower() in content_lower)
        alignment_score = min(matches / max(len(query_keywords), 1), 1.0)
        scores.append(alignment_score * 0.20)
    else:
        # No query context — give a neutral baseline
        scores.append(0.10)

    return min(max(sum(scores), 0.0), 1.0)


class ImportancePredictor:
    """Batch importance prediction for a collection of memories.

    Usage::

        predictor = ImportancePredictor()
        scores = predictor.predict_batch(memories, access_logs, query="AI safety")
    """

    def __init__(self) -> None:
        self._keyword_cache: dict[str, list[str]] = {}

    def predict(
        self,
        memory: Memory,
        access_history: list[datetime],
        query: str | None = None,
    ) -> float:
        """Predict importance for a single memory."""
        keywords = self._extract_keywords(query) if query else None
        return predict_importance(memory, access_history, keywords)

    def predict_batch(
        self,
        memories: Sequence[Memory],
        access_logs: dict[str, list[datetime]],
        query: str | None = None,
    ) -> dict[str, float]:
        """Predict importance for a batch of memories.

        Returns mapping ``memory_id -> importance_score``.
        """
        keywords = self._extract_keywords(query) if query else None
        result: dict[str, float] = {}
        for mem in memories:
            history = access_logs.get(mem.id, [])
            result[mem.id] = predict_importance(mem, history, keywords)
        return result

    @staticmethod
    def _extract_keywords(text: str | None) -> list[str] | None:
        """Extract simple keywords from a query string."""
        if not text:
            return None
        # Simple tokenization — split on whitespace, filter short words
        words = [w.strip(".,!?;:'\"()-[]{}<>") for w in text.split()]
        return [w for w in words if len(w) > 2]
