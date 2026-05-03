"""Built-in plugins for common memory management patterns."""
from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

from neuralmem.core.types import Memory, SearchResult
from neuralmem.plugins.base import Plugin

if TYPE_CHECKING:
    from neuralmem.core.types import SearchQuery

_logger = logging.getLogger(__name__)


class DedupPlugin(Plugin):
    """Checks for near-duplicate memories on remember.

    If the incoming memory is very similar to an existing one
    (above threshold), merges them by extending the existing
    content rather than storing a duplicate.
    """

    def __init__(
        self,
        storage: object | None = None,
        embedder: object | None = None,
        threshold: float = 0.90,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "dedup"

    @property
    def priority(self) -> int:
        return 10

    def on_before_remember(self, memory: Memory) -> Memory:
        """Check for duplicates and merge if above threshold."""
        if self._storage is None or self._embedder is None:
            return memory

        try:
            embedding = getattr(memory, "embedding", None)
            if embedding is None:
                return memory

            similar = self._storage.find_similar(
                embedding,
                user_id=memory.user_id,
                threshold=self._threshold,
            )
            if not similar:
                return memory

            # Take the most similar existing memory
            best = similar[0]
            if best.id == memory.id:
                return memory

            # Merge: combine content of old into new
            merged_content = f"{best.content}\n{memory.content}"
            merged_tags = tuple(
                dict.fromkeys(list(best.tags) + list(memory.tags))
            )
            merged_importance = max(best.importance, memory.importance)

            return memory.model_copy(
                update={
                    "content": merged_content,
                    "tags": merged_tags,
                    "importance": merged_importance,
                    "supersedes": best.id,
                }
            )
        except Exception as exc:
            _logger.debug("DedupPlugin check failed: %s", exc)
            return memory


class ImportancePlugin(Plugin):
    """Adjusts importance score based on content richness.

    After remembering, boosts importance for memories with
    longer content and more entity references.
    """

    def __init__(
        self,
        length_weight: float = 0.1,
        entity_weight: float = 0.05,
        max_boost: float = 0.3,
    ) -> None:
        self._length_weight = length_weight
        self._entity_weight = entity_weight
        self._max_boost = max_boost

    @property
    def name(self) -> str:
        return "importance"

    @property
    def priority(self) -> int:
        return 50

    def on_after_remember(self, memory: Memory) -> None:
        """Adjust importance based on content length and entity count."""
        try:
            content_len = len(memory.content)
            entity_count = len(memory.entity_ids)

            # Log-scaled length bonus
            length_bonus = self._length_weight * math.log1p(content_len / 100.0)
            entity_bonus = self._entity_weight * entity_count

            total_boost = min(self._max_boost, length_bonus + entity_bonus)
            new_importance = min(1.0, memory.importance + total_boost)

            if new_importance > memory.importance and hasattr(
                memory, "_storage"
            ):
                memory._storage.update_memory(
                    memory.id, importance=new_importance
                )
        except Exception as exc:
            _logger.debug("ImportancePlugin failed: %s", exc)


class RecencyBoostPlugin(Plugin):
    """Boosts recent memories in recall results.

    Before recall, adds a recency context marker. After recall,
    adjusts scores of results based on how recently they were
    created — newer memories get a score boost.
    """

    def __init__(self, decay_hours: float = 168.0) -> None:
        """decay_hours: half-life in hours for recency decay."""
        self._decay_hours = decay_hours

    @property
    def name(self) -> str:
        return "recency_boost"

    @property
    def priority(self) -> int:
        return 80

    def on_before_recall(self, query: SearchQuery) -> SearchQuery:
        """Mark query with recency context."""
        return query

    def on_after_recall(
        self, results: list[SearchResult]
    ) -> list[SearchResult]:
        """Boost scores of recent memories and re-sort."""
        if not results:
            return results

        now = time.time()
        boosted: list[SearchResult] = []

        for result in results:
            try:
                created = result.memory.created_at
                created_ts = created.timestamp()
                age_hours = (now - created_ts) / 3600.0

                # Exponential decay: boost = e^(-age / half_life * ln2)
                decay = math.exp(
                    -age_hours / self._decay_hours * math.log(2)
                )
                boost = 0.1 * decay  # Max 10% boost for brand-new memories
                new_score = min(1.0, result.score + boost)

                boosted.append(
                    SearchResult(
                        memory=result.memory,
                        score=new_score,
                        retrieval_method=result.retrieval_method,
                        explanation=result.explanation,
                    )
                )
            except Exception:
                boosted.append(result)

        boosted.sort(key=lambda r: r.score, reverse=True)
        return boosted
