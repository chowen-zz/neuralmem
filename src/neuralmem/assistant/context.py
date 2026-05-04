"""ContextInjector — inject relevant memories into writing prompts."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from neuralmem.core.types import Memory, MemoryType, SearchResult

_logger = logging.getLogger(__name__)


class MemoryRetriever(Protocol):
    """Protocol: anything that can search memories by vector or keyword."""

    def vector_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]: ...

    def keyword_search(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]: ...

    def get_memory(self, memory_id: str) -> Memory | None: ...


class Embedder(Protocol):
    """Protocol: anything that can encode text to a vector."""

    def encode_one(self, text: str) -> list[float]: ...


@dataclass
class ContextConfig:
    """Configuration for context retrieval."""

    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    min_score: float = 0.3
    max_memories: int = 5
    recency_boost: bool = True
    recency_hours: int = 24
    recency_multiplier: float = 1.2
    deduplicate: bool = True


class ContextInjector:
    """Retrieves and injects relevant NeuralMem memories into prompts.

    Uses hybrid vector + keyword search with optional recency boosting
    and deduplication.
    """

    def __init__(
        self,
        retriever: MemoryRetriever,
        embedder: Embedder | None = None,
        config: ContextConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._embedder = embedder
        self._config = config or ContextConfig()

    def retrieve(
        self,
        query: str,
        user_id: str | None = None,
        limit: int | None = None,
        memory_types: list[MemoryType] | None = None,
        tags: list[str] | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant memories for a writing query.

        Returns ranked SearchResult list, highest relevance first.
        """
        limit = limit or self._config.max_memories
        scored: dict[str, float] = {}

        # --- Vector search ---
        if self._embedder is not None:
            try:
                vector = self._embedder.encode_one(query)
                vec_results = self._retriever.vector_search(
                    vector,
                    user_id=user_id,
                    memory_types=memory_types,
                    limit=limit * 3,
                )
                for mem_id, score in vec_results:
                    scored[mem_id] = scored.get(mem_id, 0.0) + score * self._config.vector_weight
            except Exception as exc:
                _logger.warning("Vector search failed: %s", exc)

        # --- Keyword search ---
        try:
            kw_results = self._retriever.keyword_search(
                query,
                user_id=user_id,
                memory_types=memory_types,
                limit=limit * 3,
            )
            for mem_id, score in kw_results:
                scored[mem_id] = scored.get(mem_id, 0.0) + score * self._config.keyword_weight
        except Exception as exc:
            _logger.warning("Keyword search failed: %s", exc)

        # --- Fetch memory objects ---
        memories: dict[str, Memory] = {}
        for mem_id in list(scored.keys()):
            mem = self._retriever.get_memory(mem_id)
            if mem is not None and mem.is_active:
                memories[mem_id] = mem
            else:
                scored.pop(mem_id, None)

        # --- Filter by tags ---
        if tags:
            filtered: dict[str, float] = {}
            filtered_mems: dict[str, Memory] = {}
            for mem_id, mem in memories.items():
                if any(t in mem.tags for t in tags):
                    filtered[mem_id] = scored[mem_id]
                    filtered_mems[mem_id] = mem
            scored = filtered
            memories = filtered_mems

        # --- Recency boost ---
        if self._config.recency_boost:
            from datetime import datetime, timedelta, timezone
            cutoff = datetime.now(timezone.utc) - timedelta(hours=self._config.recency_hours)
            for mem_id, mem in memories.items():
                if mem.created_at > cutoff:
                    scored[mem_id] *= self._config.recency_multiplier

        # --- Deduplication by content similarity ---
        if self._config.deduplicate:
            scored, memories = self._deduplicate(scored, memories)

        # --- Sort and build SearchResult ---
        sorted_ids = sorted(scored.keys(), key=lambda k: scored[k], reverse=True)
        results: list[SearchResult] = []
        for mem_id in sorted_ids[:limit]:
            if scored[mem_id] < self._config.min_score:
                continue
            results.append(
                SearchResult(
                    memory=memories[mem_id],
                    score=min(round(scored[mem_id], 4), 1.0),
                    retrieval_method="hybrid",
                )
            )
        return results

    @staticmethod
    def _deduplicate(
        scored: dict[str, float],
        memories: dict[str, Memory],
        similarity_threshold: float = 0.85,
    ) -> tuple[dict[str, float], dict[str, Memory]]:
        """Remove near-duplicate memories by simple content overlap."""
        kept_ids: list[str] = []
        kept_contents: list[set[str]] = []
        for mem_id in sorted(scored.keys(), key=lambda k: scored[k], reverse=True):
            words = set(memories[mem_id].content.lower().split())
            is_dup = False
            for existing in kept_contents:
                if not words:
                    continue
                overlap = len(words & existing) / len(words)
                if overlap >= similarity_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept_ids.append(mem_id)
                kept_contents.append(words)
        new_scored = {k: scored[k] for k in kept_ids}
        new_memories = {k: memories[k] for k in kept_ids}
        return new_scored, new_memories

    def format_context(self, results: list[SearchResult]) -> str:
        """Format search results into a context block for prompts."""
        if not results:
            return ""
        lines = ["--- Relevant Memory Context ---"]
        for r in results:
            m = r.memory
            lines.append(f"  [{m.memory_type.value}] {m.content} (relevance: {r.score:.2f})")
        lines.append("---")
        return "\n".join(lines)
