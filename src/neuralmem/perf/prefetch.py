"""PrefetchEngine: predictive preloading for sub-100ms P99 latency.

Analyzes query patterns and proactively fetches / warms caches for likely
next queries. Integrates with QueryCache to store prefetched results.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class PrefetchStats:
    """Prefetch engine statistics."""
    predictions_made: int = 0
    predictions_hit: int = 0
    predictions_miss: int = 0
    items_prefetched: int = 0
    prefetch_time_ms: float = 0.0
    avg_prefetch_latency_ms: float = 0.0


@dataclass
class _QueryRecord:
    """Internal record of a query execution."""
    query: str
    user_id: str | None
    timestamp: float
    result_count: int = 0


class PrefetchEngine:
    """Predictive query prefetcher based on temporal and semantic patterns.

    Maintains a sliding window of recent queries, detects repeating sequences,
    and prefetches results for predicted next queries into the cache.
    """

    def __init__(
        self,
        *,
        cache: Any | None = None,
        history_window: int = 100,
        min_sequence_length: int = 2,
        max_sequence_length: int = 4,
        prefetch_timeout: float = 0.05,
        similarity_threshold: float = 0.85,
    ) -> None:
        self._cache = cache
        self._history_window = history_window
        self._min_sequence_length = min_sequence_length
        self._max_sequence_length = max_sequence_length
        self._prefetch_timeout = prefetch_timeout
        self._similarity_threshold = similarity_threshold
        self._lock = threading.Lock()
        self._history: deque[_QueryRecord] = deque(maxlen=history_window)
        self._sequence_index: dict[tuple[str, ...], list[str]] = {}
        self._stats = PrefetchStats()
        self._prefetch_latencies: list[float] = []

    # ---- public API ----

    def record_query(
        self,
        query: str,
        *,
        user_id: str | None = None,
        result_count: int = 0,
    ) -> None:
        """Record a query execution for pattern learning."""
        record = _QueryRecord(
            query=query.lower().strip(),
            user_id=user_id,
            timestamp=time.monotonic(),
            result_count=result_count,
        )
        with self._lock:
            self._history.append(record)
            self._update_sequence_index()

    def predict_next(self, recent_queries: list[str]) -> list[str]:
        """Predict likely next queries based on recent history.

        Returns a ranked list of predicted query strings.
        """
        normalized = [q.lower().strip() for q in recent_queries]
        predictions: list[str] = []

        with self._lock:
            # Try longest sequences first
            for length in range(
                min(self._max_sequence_length, len(normalized)),
                self._min_sequence_length - 1,
                -1,
            ):
                seq = tuple(normalized[-length:])
                candidates = self._sequence_index.get(seq, [])
                for cand in candidates:
                    if cand not in predictions:
                        predictions.append(cand)
                if predictions:
                    break

        return predictions[:3]  # Top 3 predictions

    def prefetch(
        self,
        recent_queries: list[str],
        fetch_fn: Callable[[str], Any],
    ) -> int:
        """Prefetch predicted queries using the provided fetch function.

        Args:
            recent_queries: Recent query strings.
            fetch_fn: Callable that executes a query and returns results.
                Signature: fetch_fn(query) -> results.

        Returns:
            Number of items successfully prefetched.
        """
        predictions = self.predict_next(recent_queries)
        if not predictions or self._cache is None:
            return 0

        count = 0
        start = time.monotonic()
        for query in predictions:
            # Skip if already cached
            key = self._make_key(query)
            if hasattr(self._cache, "get") and self._cache.get(key) is not None:
                continue

            t0 = time.monotonic()
            try:
                results = fetch_fn(query)
                elapsed = time.monotonic() - t0
                if elapsed > self._prefetch_timeout:
                    _logger.debug(
                        "Prefetch for %r took %.3fs, exceeding timeout %.3fs",
                        query, elapsed, self._prefetch_timeout,
                    )
                    continue

                if hasattr(self._cache, "set"):
                    self._cache.set(key, results)
                count += 1
                self._prefetch_latencies.append(elapsed)
            except Exception as exc:
                _logger.warning("Prefetch failed for %r: %s", query, exc)

        total_time = time.monotonic() - start
        with self._lock:
            self._stats.predictions_made += len(predictions)
            self._stats.items_prefetched += count
            self._stats.prefetch_time_ms += total_time * 1000
            if self._prefetch_latencies:
                self._stats.avg_prefetch_latency_ms = (
                    sum(self._prefetch_latencies)
                    / len(self._prefetch_latencies)
                    * 1000
                )

        return count

    def on_cache_hit(self, query: str) -> None:
        """Notify the engine that a prefetch prediction was correct."""
        with self._lock:
            self._stats.predictions_hit += 1

    def on_cache_miss(self, query: str) -> None:
        """Notify the engine that a prefetch prediction was wrong."""
        with self._lock:
            self._stats.predictions_miss += 1

    def stats(self) -> PrefetchStats:
        """Return prefetch engine statistics."""
        with self._lock:
            return PrefetchStats(
                predictions_made=self._stats.predictions_made,
                predictions_hit=self._stats.predictions_hit,
                predictions_miss=self._stats.predictions_miss,
                items_prefetched=self._stats.items_prefetched,
                prefetch_time_ms=self._stats.prefetch_time_ms,
                avg_prefetch_latency_ms=self._stats.avg_prefetch_latency_ms,
            )

    def reset(self) -> None:
        """Clear all history and statistics."""
        with self._lock:
            self._history.clear()
            self._sequence_index.clear()
            self._stats = PrefetchStats()
            self._prefetch_latencies.clear()

    # ---- internal ----

    def _update_sequence_index(self) -> None:
        """Rebuild sequence index from current history."""
        self._sequence_index.clear()
        queries = [r.query for r in self._history]
        for length in range(self._min_sequence_length, self._max_sequence_length + 1):
            for i in range(len(queries) - length):
                seq = tuple(queries[i : i + length])
                next_q = queries[i + length]
                if seq not in self._sequence_index:
                    self._sequence_index[seq] = []
                if next_q not in self._sequence_index[seq]:
                    self._sequence_index[seq].append(next_q)

    @staticmethod
    def _make_key(query: str) -> str:
        """Create a simple hash key for a query string."""
        return hashlib.sha256(query.lower().strip().encode("utf-8")).hexdigest()
