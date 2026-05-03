"""Thread-safe LRU cache with per-entry TTL expiration."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheStats:
    """Cache hit/miss statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hits / self.total


class LRUCache:
    """Thread-safe LRU cache with per-entry TTL.

    Uses dict + OrderedDict for O(1) get/put with LRU eviction.
    Each entry stores (value, expiry_time). Expired entries are lazily
    evicted on access.
    """

    def __init__(
        self,
        max_size: int = 256,
        ttl_seconds: float = 300.0,
    ) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._stats = CacheStats()

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    def get(self, key: str) -> Any | None:
        """Return cached value or None on miss/expiry."""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            value, expiry = self._cache[key]
            if time.monotonic() > expiry:
                # Expired — remove lazily
                del self._cache[key]
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._stats.hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        """Insert or update a cache entry."""
        expiry = time.monotonic() + self._ttl_seconds
        with self._lock:
            if key in self._cache:
                # Update existing — move to end
                self._cache.move_to_end(key)
                self._cache[key] = (value, expiry)
            else:
                self._cache[key] = (value, expiry)
                if len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)
                    self._stats.evictions += 1

    def invalidate(self, key: str) -> bool:
        """Remove a specific key. Returns True if key existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.invalidations += 1
                return True
            return False

    def clear(self) -> int:
        """Remove all entries. Returns number of entries removed."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def stats(self) -> CacheStats:
        """Return a snapshot of cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                invalidations=self._stats.invalidations,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            _, expiry = self._cache[key]
            if time.monotonic() > expiry:
                del self._cache[key]
                return False
            return True
