"""NeuralMem community features — user feedback on memory quality.

V1.3: FeedbackLoop
  • Record thumbs-up / thumbs-down / rating / comment on a memory
  • Aggregate feedback into quality scores
  • Flag low-quality memories for review
  • In-memory implementation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neuralmem.core.types import Memory


@dataclass
class FeedbackEntry:
    """A single feedback record."""
    feedback_id: str
    memory_id: str
    user_id: str
    rating: int | None = None          # 1-5 scale
    helpful: bool | None = None          # True = thumbs-up, False = thumbs-down
    comment: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class FeedbackLoop:
    """In-memory feedback manager.

    Usage:
        fb = FeedbackLoop()
        fb.submit_feedback(mem.id, user_id="u1", helpful=True, comment="accurate")
        score = fb.aggregate_score(mem.id)
        flagged = fb.list_flagged(threshold=0.3)
    """

    def __init__(self, flag_threshold: float = 0.3) -> None:
        # memory_id -> list of FeedbackEntry
        self._feedback: dict[str, list[FeedbackEntry]] = {}
        self._flag_threshold = flag_threshold
        self._counter: int = 0

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _next_id(self) -> str:
        self._counter += 1
        return f"fbk-{self._counter:06d}"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def submit_feedback(
        self,
        memory_id: str,
        user_id: str,
        *,
        rating: int | None = None,
        helpful: bool | None = None,
        comment: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackEntry:
        """Submit feedback for a memory.

        Args:
            memory_id: target memory ID.
            user_id: who is giving feedback.
            rating: optional 1-5 rating.
            helpful: optional boolean thumbs-up/down.
            comment: free-text comment.
            metadata: arbitrary extra data.

        Returns:
            The created FeedbackEntry.
        """
        entry = FeedbackEntry(
            feedback_id=self._next_id(),
            memory_id=memory_id,
            user_id=user_id,
            rating=rating,
            helpful=helpful,
            comment=comment,
            metadata=metadata or {},
        )
        self._feedback.setdefault(memory_id, []).append(entry)
        return entry

    def get_feedback(self, memory_id: str) -> list[FeedbackEntry]:
        """Return all feedback entries for a memory."""
        return list(self._feedback.get(memory_id, []))

    def get_user_feedback(self, user_id: str) -> list[FeedbackEntry]:
        """Return all feedback given by a user."""
        results: list[FeedbackEntry] = []
        for entries in self._feedback.values():
            for e in entries:
                if e.user_id == user_id:
                    results.append(e)
        return results

    def aggregate_score(self, memory_id: str) -> float:
        """Compute a quality score in [0, 1] for a memory.

        Based on:
          • helpful votes (thumbs-up = +1, thumbs-down = -1)
          • average rating (1-5 mapped to 0-1)
        """
        entries = self._feedback.get(memory_id, [])
        if not entries:
            return 0.5  # neutral default

        helpful_score: float = 0.0
        rating_sum: float = 0.0
        rating_count: int = 0

        for e in entries:
            if e.helpful is True:
                helpful_score += 1.0
            elif e.helpful is False:
                helpful_score -= 1.0
            if e.rating is not None:
                rating_sum += (e.rating - 1) / 4.0  # map 1-5 to 0-1
                rating_count += 1

        # normalize helpful to [-1, 1] then [0, 1]
        total = len(entries)
        helpful_norm = (helpful_score / total + 1) / 2 if total > 0 else 0.5

        if rating_count > 0:
            rating_norm = rating_sum / rating_count
            # weighted blend: 60% helpful, 40% rating
            return round(0.6 * helpful_norm + 0.4 * rating_norm, 3)

        return round(helpful_norm, 3)

    def is_flagged(self, memory_id: str) -> bool:
        """True if the memory's aggregate score is below the flag threshold."""
        return self.aggregate_score(memory_id) < self._flag_threshold

    def list_flagged(self, threshold: float | None = None) -> list[str]:
        """Return memory IDs with scores below threshold."""
        thr = threshold if threshold is not None else self._flag_threshold
        return [mid for mid in self._feedback if self.aggregate_score(mid) < thr]

    def get_summary(self, memory_id: str) -> dict[str, Any]:
        """Return a human-readable feedback summary."""
        entries = self._feedback.get(memory_id, [])
        ups = sum(1 for e in entries if e.helpful is True)
        downs = sum(1 for e in entries if e.helpful is False)
        ratings = [e.rating for e in entries if e.rating is not None]
        return {
            "memory_id": memory_id,
            "total_feedback": len(entries),
            "thumbs_up": ups,
            "thumbs_down": downs,
            "average_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "quality_score": self.aggregate_score(memory_id),
            "flagged": self.is_flagged(memory_id),
        }

    def get_global_stats(self) -> dict[str, Any]:
        """Return global feedback statistics."""
        total_entries = sum(len(v) for v in self._feedback.values())
        total_memories = len(self._feedback)
        flagged = self.list_flagged()
        return {
            "total_feedback_entries": total_entries,
            "memories_with_feedback": total_memories,
            "flagged_memory_count": len(flagged),
            "flagged_memory_ids": flagged,
        }
