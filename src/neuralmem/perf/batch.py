"""Batch operations with thread-pool parallelism for NeuralMem."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


def _percentile(sorted_data: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) from a pre-sorted list.

    Uses nearest-rank method — no numpy required.
    """
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
    """Aggregate result of a batch operation."""
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
    """Execute NeuralMem operations in parallel via ThreadPoolExecutor.

    Provides ``batch_add`` and ``batch_search`` that run items concurrently
    while handling per-item errors gracefully.
    """

    def __init__(
        self,
        neural_mem: Any,
        *,
        max_workers: int = 4,
    ) -> None:
        self._mem = neural_mem
        self._max_workers = max_workers

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def batch_add(
        self,
        contents: list[str],
        user_id: str | None = None,
        **kwargs: Any,
    ) -> BatchResult:
        """Remember multiple contents concurrently.

        Args:
            contents: List of text strings to store.
            user_id: User identifier for scoping.
            **kwargs: Forwarded to ``NeuralMem.remember()``.

        Returns:
            BatchResult with per-item outcomes and aggregate timing stats.
        """
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
        return BatchResult(items=results, total_time=total)

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

    def batch_search(
        self,
        queries: list[str],
        user_id: str | None = None,
        **kwargs: Any,
    ) -> BatchResult:
        """Search multiple queries concurrently.

        Args:
            queries: List of query strings.
            user_id: User identifier for scoping.
            **kwargs: Forwarded to ``NeuralMem.recall()``.

        Returns:
            BatchResult with per-item outcomes and aggregate timing stats.
        """
        start = time.monotonic()
        results: list[ItemResult] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_map = {}
            for idx, query in enumerate(queries):
                fut = pool.submit(
                    self._search_one, idx, query, user_id, kwargs
                )
                future_map[fut] = idx

            for future in as_completed(future_map):
                results.append(future.result())

        total = time.monotonic() - start
        results.sort(key=lambda r: r.index)
        return BatchResult(items=results, total_time=total)

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
