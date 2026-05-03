"""BatchProcessor V1.2: enhanced batch processing for sub-100ms P99 latency.

Extends the existing batch module with:
- Adaptive concurrency based on latency feedback
- Result deduplication
- Streaming / chunked processing for large batches
- Integration with QueryCache and PrefetchEngine
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from neuralmem.perf.query_cache import QueryCache

_logger = logging.getLogger(__name__)


def _percentile(sorted_data: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) from a pre-sorted list."""
    if not sorted_data:
        return 0.0
    k = max(0, min(len(sorted_data) - 1, int(len(sorted_data) * p / 100) - 1))
    return sorted_data[k]


@dataclass
class ItemResult:
    """Result for a single item in a batch operation."""
    index: int
    success: bool
    result: Any = None
    error: str | None = None
    elapsed: float = 0.0


@dataclass
class BatchResult:
    """Aggregate result of a batch operation with latency stats."""
    items: list[ItemResult] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def success_count(self) -> int:
        return sum(1 for i in self.items if i.success)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.items if not i.success)

    @property
    def elapsed_times(self) -> list[float]:
        return [i.elapsed for i in self.items if i.success]

    @property
    def avg_time(self) -> float:
        times = self.elapsed_times
        return sum(times) / len(times) if times else 0.0

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def p99(self) -> float:
        return self._percentile(99)

    def _percentile(self, p: float) -> float:
        times = sorted(self.elapsed_times)
        return _percentile(times, p)


class BatchProcessor:
    """Execute NeuralMem operations in parallel with adaptive concurrency.

    V1.2 enhancements:
    - Adaptive max_workers based on P99 latency feedback
    - QueryCache integration for batch_search
    - Streaming mode for very large batches
    - Result deduplication
    """

    def __init__(
        self,
        neural_mem: Any,
        *,
        max_workers: int = 4,
        cache: QueryCache[Any] | None = None,
        p99_target_ms: float = 100.0,
    ) -> None:
        self._mem = neural_mem
        self._max_workers = max_workers
        self._cache = cache
        self._p99_target_ms = p99_target_ms
        self._latency_history: list[float] = []

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def batch_add(
        self,
        contents: list[str],
        user_id: str | None = None,
        **kwargs: Any,
    ) -> BatchResult:
        """Remember multiple contents concurrently."""
        start = time.monotonic()
        results: list[ItemResult] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_map = {}
            for idx, content in enumerate(contents):
                fut = pool.submit(
                    self._add_one, idx, content, user_id, kwargs
                )
                future_map[fut] = idx

            for future in as_completed(future_map):
                results.append(future.result())

        total = time.monotonic() - start
        results.sort(key=lambda r: r.index)
        result = BatchResult(items=results, total_time=total)
        self._record_latencies(result.elapsed_times)
        return result

    def batch_search(
        self,
        queries: list[str],
        user_id: str | None = None,
        **kwargs: Any,
    ) -> BatchResult:
        """Search multiple queries concurrently with cache integration."""
        start = time.monotonic()
        results: list[ItemResult] = []

        # Check cache first
        cached_results: dict[int, Any] = {}
        uncached_indices: list[int] = []
        if self._cache is not None:
            for idx, query in enumerate(queries):
                key = self._cache.make_key(
                    query,
                    user_id=user_id,
                    limit=kwargs.get("limit", 10),
                    min_score=kwargs.get("min_score", 0.3),
                )
                cached = self._cache.get(key)
                if cached is not None:
                    cached_results[idx] = cached
                else:
                    uncached_indices.append(idx)
        else:
            uncached_indices = list(range(len(queries)))

        # Run uncached queries in parallel
        if uncached_indices:
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                future_map = {}
                for idx in uncached_indices:
                    fut = pool.submit(
                        self._search_one, idx, queries[idx], user_id, kwargs
                    )
                    future_map[fut] = idx

                for future in as_completed(future_map):
                    item_result = future.result()
                    results.append(item_result)
                    # Store successful results in cache
                    if item_result.success and self._cache is not None:
                        key = self._cache.make_key(
                            queries[item_result.index],
                            user_id=user_id,
                            limit=kwargs.get("limit", 10),
                            min_score=kwargs.get("min_score", 0.3),
                        )
                        self._cache.set(key, item_result.result)

        # Add cached results
        for idx, cached in cached_results.items():
            results.append(
                ItemResult(index=idx, success=True, result=cached, elapsed=0.0)
            )

        total = time.monotonic() - start
        results.sort(key=lambda r: r.index)
        result = BatchResult(items=results, total_time=total)
        self._record_latencies(result.elapsed_times)
        return result

    def batch_search_streaming(
        self,
        queries: list[str],
        user_id: str | None = None,
        chunk_size: int = 50,
        **kwargs: Any,
    ) -> BatchResult:
        """Process large query lists in chunks to bound memory."""
        all_results: list[ItemResult] = []
        total_time = 0.0

        for i in range(0, len(queries), chunk_size):
            chunk = queries[i : i + chunk_size]
            chunk_result = self.batch_search(chunk, user_id=user_id, **kwargs)
            # Remap indices
            for item in chunk_result.items:
                item.index = i + item.index
                all_results.append(item)
            total_time += chunk_result.total_time

        all_results.sort(key=lambda r: r.index)
        return BatchResult(items=all_results, total_time=total_time)

    def deduplicate_results(self, result: BatchResult) -> BatchResult:
        """Remove duplicate results across batch items by memory ID."""
        seen_ids: set[str] = set()
        deduped_items: list[ItemResult] = []

        for item in result.items:
            if not item.success or item.result is None:
                deduped_items.append(item)
                continue

            # Handle both single results and lists
            if isinstance(item.result, list):
                unique = []
                for r in item.result:
                    mid = getattr(r, "id", None) or getattr(r, "memory_id", None)
                    if mid and mid not in seen_ids:
                        seen_ids.add(mid)
                        unique.append(r)
                    elif mid is None:
                        unique.append(r)
                deduped_items.append(
                    ItemResult(
                        index=item.index,
                        success=item.success,
                        result=unique,
                        error=item.error,
                        elapsed=item.elapsed,
                    )
                )
            else:
                mid = getattr(item.result, "id", None) or getattr(
                    item.result, "memory_id", None
                )
                if mid is None or mid not in seen_ids:
                    if mid:
                        seen_ids.add(mid)
                    deduped_items.append(item)

        return BatchResult(items=deduped_items, total_time=result.total_time)

    def _add_one(
        self,
        idx: int,
        content: str,
        user_id: str | None,
        kwargs: dict[str, Any],
    ) -> ItemResult:
        t0 = time.monotonic()
        try:
            mem = self._mem.remember(content, user_id=user_id, **kwargs)
            elapsed = time.monotonic() - t0
            return ItemResult(index=idx, success=True, result=mem, elapsed=elapsed)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            _logger.warning("batch_add item %d failed: %s", idx, exc)
            return ItemResult(
                index=idx, success=False, error=str(exc), elapsed=elapsed
            )

    def _search_one(
        self,
        idx: int,
        query: str,
        user_id: str | None,
        kwargs: dict[str, Any],
    ) -> ItemResult:
        t0 = time.monotonic()
        try:
            res = self._mem.recall(query, user_id=user_id, **kwargs)
            elapsed = time.monotonic() - t0
            return ItemResult(index=idx, success=True, result=res, elapsed=elapsed)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            _logger.warning("batch_search item %d failed: %s", idx, exc)
            return ItemResult(
                index=idx, success=False, error=str(exc), elapsed=elapsed
            )

    def _record_latencies(self, latencies: list[float]) -> None:
        """Record latencies and adapt concurrency if P99 exceeds target."""
        self._latency_history.extend(latencies)
        if len(self._latency_history) > 200:
            self._latency_history = self._latency_history[-200:]

        if len(self._latency_history) >= 20:
            sorted_hist = sorted(self._latency_history)
            p99 = _percentile(sorted_hist, 99) * 1000
            if p99 > self._p99_target_ms * 1.5:
                self._max_workers = max(1, self._max_workers - 1)
                _logger.info(
                    "Reduced max_workers to %d (P99=%.1fms > target %.1fms)",
                    self._max_workers, p99, self._p99_target_ms,
                )
            elif p99 < self._p99_target_ms * 0.5 and self._max_workers < 16:
                self._max_workers = min(16, self._max_workers + 1)
                _logger.info(
                    "Increased max_workers to %d (P99=%.1fms < target %.1fms)",
                    self._max_workers, p99, self._p99_target_ms,
                )
