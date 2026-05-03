"""Prometheus-compatible metrics collector for NeuralMem.

Pure-Python implementation with zero external dependencies. Thread-safe
using a lock. Tracks counters, gauges, and histograms and can expose
them in Prometheus text exposition format.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager


class MetricsCollector:
    """Thread-safe, zero-dependency metrics collector.

    Tracks:
    - **Counters**: ``total_recalls``, ``total_remembers``, ``error_count``
    - **Gauges**: ``memory_count``, ``cache_hit_rate``
    - **Histograms**: ``recall_latency_ms``

    Use ``expose()`` to generate a Prometheus text-format string suitable
    for scraping.

    Example::

        mc = MetricsCollector()
        mc.inc("total_recalls")
        mc.set_gauge("memory_count", 42)
        mc.observe("recall_latency_ms", 12.5)
        print(mc.expose())
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Counters (int, monotonically increasing)
        self._counters: dict[str, int] = defaultdict(int)
        # Gauges (float, can go up or down)
        self._gauges: dict[str, float] = defaultdict(float)
        # Histograms (list of observed values)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        # Metadata
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def inc(self, name: str, value: int = 1) -> None:
        """Increment a counter by *value*."""
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge to *value*."""
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        """Record an observation in a histogram."""
        with self._lock:
            self._histograms[name].append(value)

    @contextmanager
    def timer(self, name: str) -> Generator[None, None, None]:
        """Context manager that records elapsed wall-clock time in *name*
        histogram (in milliseconds)."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.observe(name, elapsed_ms)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def record_recall(self, latency_ms: float) -> None:
        """Record a recall operation."""
        self.inc("total_recalls")
        self.observe("recall_latency_ms", latency_ms)

    def record_remember(self) -> None:
        """Record a remember operation."""
        self.inc("total_remembers")

    def record_error(self) -> None:
        """Increment the error counter."""
        self.inc("error_count")

    def update_cache_hit_rate(self, rate: float) -> None:
        """Update the cache hit-rate gauge (0.0 – 1.0)."""
        self.set_gauge("cache_hit_rate", max(0.0, min(1.0, rate)))

    def update_memory_count(self, count: int) -> None:
        """Update the memory-count gauge."""
        self.set_gauge("memory_count", float(count))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_counter(self, name: str) -> int:
        """Return the current value of a counter."""
        with self._lock:
            return self._counters[name]

    def get_gauge(self, name: str) -> float:
        """Return the current value of a gauge."""
        with self._lock:
            return self._gauges[name]

    def get_histogram(self, name: str) -> list[float]:
        """Return a copy of all observations for a histogram."""
        with self._lock:
            return list(self._histograms[name])

    def snapshot(self) -> dict[str, object]:
        """Return a plain-dict snapshot of all metrics."""
        with self._lock:
            hist_summary: dict[str, dict[str, float]] = {}
            for name, values in self._histograms.items():
                if not values:
                    continue
                sorted_v = sorted(values)
                n = len(sorted_v)
                hist_summary[name] = {
                    "count": n,
                    "sum": sum(sorted_v),
                    "min": sorted_v[0],
                    "max": sorted_v[-1],
                    "mean": sum(sorted_v) / n,
                    "p50": sorted_v[n // 2],
                    "p95": sorted_v[int(n * 0.95)] if n > 1 else sorted_v[0],
                    "p99": sorted_v[int(n * 0.99)] if n > 1 else sorted_v[0],
                }
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": hist_summary,
                "uptime_seconds": time.time() - self._start_time,
            }

    # ------------------------------------------------------------------
    # Prometheus text exposition
    # ------------------------------------------------------------------

    def expose(self) -> str:
        """Return metrics in Prometheus text exposition format.

        The output follows the OpenMetrics/Prometheus convention:
        - Counters use ``_total`` suffix
        - Gauges have no suffix
        - Histograms expose ``_bucket``, ``_sum``, ``_count`` series
        """
        lines: list[str] = []
        uptime = time.time() - self._start_time

        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = {
                k: list(v) for k, v in self._histograms.items()
            }

        # Counters
        lines.append("# HELP neuralmem_uptime_seconds Process uptime.")
        lines.append("# TYPE neuralmem_uptime_seconds gauge")
        lines.append(
            f"neuralmem_uptime_seconds {uptime:.3f}"
        )

        for name, value in sorted(counters.items()):
            safe = self._safe_name(name)
            lines.append(f"# HELP neuralmem_{safe} Auto-generated counter.")
            lines.append(f"# TYPE neuralmem_{safe} counter")
            lines.append(f"neuralmem_{safe}_total {value}")

        # Gauges
        for name, value in sorted(gauges.items()):
            safe = self._safe_name(name)
            lines.append(f"# HELP neuralmem_{safe} Auto-generated gauge.")
            lines.append(f"# TYPE neuralmem_{safe} gauge")
            lines.append(f"neuralmem_{safe} {value}")

        # Histograms
        for name, values in sorted(histograms.items()):
            if not values:
                continue
            safe = self._safe_name(name)
            lines.append(
                f"# HELP neuralmem_{safe} Auto-generated histogram."
            )
            lines.append(f"# TYPE neuralmem_{safe} histogram")
            buckets = [5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
            for le in buckets:
                count = sum(1 for v in values if v <= le)
                lines.append(
                    f'neuralmem_{safe}_bucket{{le="{le:.1f}"}} {count}'
                )
            lines.append(
                f'neuralmem_{safe}_bucket{{le="+Inf"}} {len(values)}'
            )
            lines.append(f"neuralmem_{safe}_sum {sum(values):.3f}")
            lines.append(f"neuralmem_{safe}_count {len(values)}")

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._start_time = time.time()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_name(name: str) -> str:
        """Convert a metric name to a Prometheus-safe identifier."""
        return name.replace(".", "_").replace("-", "_").replace(" ", "_")
