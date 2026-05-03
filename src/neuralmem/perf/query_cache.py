"""QueryCache: LRU query result cache for sub-100ms P99 latency.

Caches recall() results keyed by query fingerprint (hash of query + user_id +
memory_types + tags + limit + min_score). Uses time-based TTL and LRU eviction
to keep memory bounded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")

_logger = logging.getLogger(__name__)


@dataclass
class CacheEntry(Generic[T]):
    """A single cached result with metadata."""
    value: T
    created_at: float = field(default_factory=time.monotonic)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.monotonic)


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    max_size: int = 0
    hit_rate: float = 0.0
    avg_hit_latency_ms: float = 0.0
    avg_miss_latency_ms: float = 0.0
    p99_hit_latency_ms: float = 0.0


class QueryCache(Generic[T]):
    """Thread-safe LRU cache for query results with TTL support.

    Designed to sit in front of ``NeuralMem.recall()`` to serve hot queries
    under the sub-100ms P99 target.
    """

    def __init__(
        self,
        *,
        max_size: int = 10_000,
        ttl_seconds: float = 300.0,
        cleanup_interval: float = 60.0,
    ) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=max_size)
        self._hit_latencies: list[float] = []
        self._miss_latencies: list[float] = []
        self._last_cleanup = time.monotonic()

    # ---- public API ----

    def get(self, key: str) -> T | None:
        """Retrieve a cached value by key."""
        t0 = time.monotonic()
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats.misses += 1
                self._record_miss(time.monotonic() - t0)
                return None
            if self._is_expired(entry):
                self._cache.pop(key, None)
                self._stats.expirations += 1
                self._stats.misses += 1
                self._record_miss(time.monotonic() - t0)
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.access_count += 1
            entry.last_accessed = time.monotonic()
            self._stats.hits += 1
            self._record_hit(time.monotonic() - t0)
            return entry.value

    def set(self, key: str, value: T) -> None:
        """Store a value in the cache."""
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_lru()
            self._cache[key] = CacheEntry(value=value)
            self._cache.move_to_end(key)
            self._stats.size = len(self._cache)
            # Periodic cleanup
            now = time.monotonic()
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup_expired()
                self._last_cleanup = now

    def invalidate(self, key: str | None = None) -> int:
        """Invalidate a specific key or the entire cache.

        Returns number of entries removed.
        """
        with self._lock:
            if key is None:
                count = len(self._cache)
                self._cache.clear()
                self._stats.size = 0
                return count
            removed = self._cache.pop(key, None) is not None
            self._stats.size = len(self._cache)
            return int(removed)

    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate keys containing a substring."""
        with self._lock:
            to_remove = [k for k in self._cache if pattern in k]
            for k in to_remove:
                self._cache.pop(k, None)
            self._stats.size = len(self._cache)
            return len(to_remove)

    def stats(self) -> CacheStats:
        """Return current cache statistics."""
        with self._lock:
            total = self._stats.hits + self._stats.misses
            hit_rate = self._stats.hits / total if total > 0 else 0.0
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expirations=self._stats.expirations,
                size=len(self._cache),
                max_size=self._max_size,
                hit_rate=round(hit_rate, 4),
                avg_hit_latency_ms=self._avg(self._hit_latencies) * 1000,
                avg_miss_latency_ms=self._avg(self._miss_latencies) * 1000,
                p99_hit_latency_ms=self._p99(self._hit_latencies) * 1000,
            )

    def make_key(
        self,
        query: str,
        *,
        user_id: str | None = None,
        memory_types: tuple[Any, ...] | None = None,
        tags: tuple[str, ...] | None = None,
        time_range: tuple[Any, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.3,
    ) -> str:
        """Build a deterministic cache key from query parameters."""
        # Normalize query: lowercase, strip whitespace
        normalized = query.lower().strip()
        # Pack parameters into a stable dict
        params: dict[str, Any] = {
            "q": normalized,
            "u": user_id,
            "mt": memory_types,
            "t": tags,
            "tr": time_range,
            "l": limit,
            "ms": min_score,
        }
        # Use compact JSON for deterministic serialization
        payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ---- internal ----

    def _is_expired(self, entry: CacheEntry[T]) -> bool:
        return (time.monotonic() - entry.created_at) > self._ttl_seconds

    def _evict_lru(self) -> None:
        if self._cache:
            self._cache.popitem(last=False)
            self._stats.evictions += 1

    def _cleanup_expired(self) -> None:
        now = time.monotonic()
        expired = [
            k for k, e in self._cache.items()
            if (now - e.created_at) > self._ttl_seconds
        ]
        for k in expired:
            self._cache.pop(k, None)
        self._stats.expirations += len(expired)
        self._stats.size = len(self._cache)

    def _record_hit(self, latency: float) -> None:
        self._hit_latencies.append(latency)
        if len(self._hit_latencies) > 1000:
            self._hit_latencies = self._hit_latencies[-1000:]

    def _record_miss(self, latency: float) -> None:
        self._miss_latencies.append(latency)
        if len(self._miss_latencies) > 1000:
            self._miss_latencies = self._miss_latencies[-1000:]

    @staticmethod
    def _avg(data: list[float]) -> float:
        return sum(data) / len(data) if data else 0.0

    @staticmethod
    def _p99(data: list[float]) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = max(0, min(len(sorted_data) - 1, int(len(sorted_data) * 0.99) - 1))
        return sorted_data[k]
