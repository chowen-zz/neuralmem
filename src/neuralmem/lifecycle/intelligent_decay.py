"""Intelligent decay — adaptive memory fading based on access patterns.

Unlike fixed half-life decay, this module considers:
- Access frequency (recent vs. historical)
- Semantic importance (predicted future value)
- User context (active projects, goals)
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neuralmem.core.types import Memory

_logger = logging.getLogger(__name__)


def compute_adaptive_decay(
    memory: Memory,
    access_history: list[datetime],
    importance_score: float = 0.5,
    *,
    half_life_days: float = 30.0,
    access_boost: float = 0.15,
    recency_window_days: float = 7.0,
) -> float:
    """Compute an adaptive decay factor for a single memory.

    The factor ranges from 0.0 (completely faded) to 1.0 (fully retained).
    It combines:

    1. **Temporal decay** — standard exponential half-life.
    2. **Access reinforcement** — each access within the recency window
       adds a small boost.
    3. **Importance weighting** — predicted future value scales the result.

    Parameters
    ----------
    memory:
        The memory to evaluate.
    access_history:
        List of timestamps when this memory was accessed (e.g. via ``recall``).
    importance_score:
        Predicted future importance, 0.0–1.0. Higher = retain longer.
    half_life_days:
        Base half-life for exponential decay.
    access_boost:
        Additional retention per access in the recency window.
    recency_window_days:
        Only accesses within this window contribute reinforcement.

    Returns
    -------
    Retention factor in ``[0.0, 1.0]``.
    """
    now = datetime.now(timezone.utc)
    age_days = max(0.0, (now - memory.created_at).total_seconds() / 86400.0)

    # 1. Base temporal decay
    base_retention = math.exp(-age_days / half_life_days * math.log(2))

    # 2. Access reinforcement
    recent_accesses = [
        ts for ts in access_history
        if (now - ts).total_seconds() / 86400.0 <= recency_window_days
    ]
    reinforcement = min(access_boost * len(recent_accesses), 0.5)

    # 3. Importance weighting — importance_score scales the *effective* half-life
    importance_multiplier = 0.5 + importance_score  # 0.5 → 1.5x half-life

    retention = (base_retention + reinforcement) * importance_multiplier
    return min(max(retention, 0.0), 1.0)


class IntelligentDecay:
    """Adaptive decay engine that operates on a collection of memories.

    Usage::

        decay = IntelligentDecay(half_life_days=30)
        to_forget = decay.select_for_forgetting(memories, access_logs)
    """

    def __init__(
        self,
        *,
        half_life_days: float = 30.0,
        access_boost: float = 0.15,
        recency_window_days: float = 7.0,
        forget_threshold: float = 0.15,
    ) -> None:
        self.half_life_days = half_life_days
        self.access_boost = access_boost
        self.recency_window_days = recency_window_days
        self.forget_threshold = forget_threshold

    def evaluate(
        self,
        memory: Memory,
        access_history: list[datetime],
        importance_score: float = 0.5,
    ) -> float:
        """Evaluate retention factor for a single memory."""
        return compute_adaptive_decay(
            memory,
            access_history,
            importance_score,
            half_life_days=self.half_life_days,
            access_boost=self.access_boost,
            recency_window_days=self.recency_window_days,
        )

    def select_for_forgetting(
        self,
        memories: Sequence[Memory],
        access_logs: dict[str, list[datetime]],
        importance_scores: dict[str, float] | None = None,
    ) -> list[Memory]:
        """Select memories whose retention has fallen below threshold.

        Parameters
        ----------
        memories:
            Candidate memories.
        access_logs:
            Mapping ``memory_id -> list of access timestamps``.
        importance_scores:
            Optional mapping ``memory_id -> predicted importance``.

        Returns
        -------
        Memories that should be forgotten.
        """
        scores = importance_scores or {}
        to_forget: list[Memory] = []
        for mem in memories:
            history = access_logs.get(mem.id, [])
            importance = scores.get(mem.id, 0.5)
            retention = self.evaluate(mem, history, importance)
            _logger.debug(
                "Memory %s retention=%.3f (age=%.1f days, accesses=%d, importance=%.2f)",
                mem.id[:8],
                retention,
                (datetime.now(timezone.utc) - mem.created_at).total_seconds() / 86400,
                len(history),
                importance,
            )
            if retention < self.forget_threshold:
                to_forget.append(mem)
        return to_forget

    def batch_decay_importance(
        self,
        memories: Sequence[Memory],
        access_logs: dict[str, list[datetime]],
        importance_scores: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Return updated importance values after applying decay.

        The new importance = old importance * retention factor.
        """
        scores = importance_scores or {}
        result: dict[str, float] = {}
        for mem in memories:
            history = access_logs.get(mem.id, [])
            importance = scores.get(mem.id, mem.importance)
            retention = self.evaluate(mem, history, importance)
            result[mem.id] = importance * retention
        return result
