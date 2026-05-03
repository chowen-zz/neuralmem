"""CacheManager — wraps RetrievalEngine.search() with transparent caching."""
from __future__ import annotations

import hashlib
import logging
from typing import Any

from neuralmem.cache.lru_cache import CacheStats, LRUCache

_logger = logging.getLogger(__name__)


def _cache_key(
    query: str,
    user_id: str | None,
    memory_types: tuple[str, ...] | None,
) -> str:
    """Build a deterministic cache key from search parameters."""
    parts = [query, user_id or "", str(memory_types or "")]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


class CacheManager:
    """Transparent result cache wrapping a retrieval engine.

    Intercepts ``search(query)`` calls: on cache hit returns cached
    ``SearchResult`` list; on miss delegates to the real engine and
    stores the result.

    Also provides an *embedding cache* for vector lookups.
    """

    def __init__(
        self,
        retrieval_engine: Any,
        *,
        max_size: int = 256,
        ttl_seconds: float = 300.0,
        embedding_cache_size: int = 512,
        embedding_ttl: float = 600.0,
    ) -> None:
        self._engine = retrieval_engine
        self._result_cache = LRUCache(
            max_size=max_size, ttl_seconds=ttl_seconds
        )
        self._embedding_cache = LRUCache(
            max_size=embedding_cache_size, ttl_seconds=embedding_ttl
        )

    # ----- search caching --------------------------------------------------

    def search(self, query: Any) -> list[Any]:
        """Proxy to retrieval engine with transparent caching.

        ``query`` must have attributes: query, user_id, memory_types.
        """
        key = _cache_key(
            query=query.query,
            user_id=getattr(query, "user_id", None),
            memory_types=getattr(query, "memory_types", None),
        )

        cached = self._result_cache.get(key)
        if cached is not None:
            _logger.debug("Cache hit for search key %s", key[:12])
            return cached

        _logger.debug("Cache miss for search key %s", key[:12])
        results = self._engine.search(query)
        self._result_cache.put(key, results)
        return results

    # ----- embedding caching -----------------------------------------------

    def get_embedding(self, text: str) -> list[float] | None:
        """Return cached embedding or None."""
        return self._embedding_cache.get(text)

    def put_embedding(self, text: str, embedding: list[float]) -> None:
        """Store an embedding in the cache."""
        self._embedding_cache.put(text, embedding)

    # ----- invalidation ----------------------------------------------------

    def invalidate_search(self, key: str) -> bool:
        """Invalidate a specific search cache entry."""
        return self._result_cache.invalidate(key)

    def invalidate_embedding(self, text: str) -> bool:
        """Invalidate a specific embedding cache entry."""
        return self._embedding_cache.invalidate(text)

    def clear_all(self) -> None:
        """Clear both caches."""
        self._result_cache.clear()
        self._embedding_cache.clear()

    # ----- stats -----------------------------------------------------------

    @property
    def search_stats(self) -> CacheStats:
        return self._result_cache.stats()

    @property
    def embedding_stats(self) -> CacheStats:
        return self._embedding_cache.stats()

    @property
    def search_cache(self) -> LRUCache:
        """Direct access to the result cache (for testing)."""
        return self._result_cache

    @property
    def embedding_cache(self) -> LRUCache:
        """Direct access to the embedding cache (for testing)."""
        return self._embedding_cache
