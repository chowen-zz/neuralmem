"""Lightweight metrics collector for neuralmem operations."""
from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


class MetricsCollector:
    """Lightweight metrics collector for neuralmem operations.

    No external dependencies. Collects counters, histograms, and timing data.
    When ``enabled=False`` (default), timer context managers still record data
    but skip logging; counter/histogram recording is a no-op.
    """

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled
        self._logger = logging.getLogger("neuralmem.metrics")
        self._counters: dict[str, int] = {}
        self._histograms: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_counter(self, name: str, value: int = 1, **tags: str) -> None:
        """Increment a named counter."""
        key = self._make_key(name, tags)
        self._counters[key] = self._counters.get(key, 0) + value

    def record_histogram(self, name: str, value: float, **tags: str) -> None:
        """Append a value to a named histogram."""
        key = self._make_key(name, tags)
        self._histograms.setdefault(key, []).append(value)

    @contextmanager
    def timer(self, name: str, **tags: str) -> Generator[None, None, None]:
        """Context manager that measures wall-clock time.

        Always records a histogram entry for ``{name}.duration``.
        When logging is enabled, also emits an INFO log line.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.record_histogram(f"{name}.duration", elapsed, **tags)
            if self._enabled:
                self._logger.info("%s completed in %.3fs", name, elapsed)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Return a snapshot of all collected metrics.

        Returns a dict with two keys:
        - ``counters``: mapping of counter_key -> int
        - ``histograms``: mapping of histogram_key -> summary stats
        """
        result: dict[str, Any] = {
            "counters": dict(self._counters),
            "histograms": {},
        }
        for key, values in self._histograms.items():
            if values:
                sorted_v = sorted(values)
                n = len(sorted_v)
                result["histograms"][key] = {
                    "count": n,
                    "mean": sum(sorted_v) / n,
                    "min": sorted_v[0],
                    "max": sorted_v[-1],
                    "p50": sorted_v[n // 2],
                    "p95": sorted_v[int(n * 0.95)] if n > 1 else sorted_v[0],
                    "p99": sorted_v[int(n * 0.99)] if n > 1 else sorted_v[0],
                }
        return result

    def reset(self) -> None:
        """Clear all collected metrics."""
        self._counters.clear()
        self._histograms.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(name: str, tags: dict[str, str]) -> str:
        """Build a canonical metric key from a name and optional tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"
