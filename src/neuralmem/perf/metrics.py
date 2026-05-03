"""PerformanceMetrics: latency-aware metrics collection for sub-100ms P99.

Collects per-operation timing histograms, throughput counters, and
latency-percentile tracking. Integrates with NeuralMem's existing
MetricsCollector while adding P99-focused instrumentation.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


def _percentile(sorted_data: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) from a pre-sorted list."""
    if not sorted_data:
        return 0.0
    k = max(0, min(len(sorted_data) - 1, int(len(sorted_data) * p / 100) - 1))
    return sorted_data[k]


@dataclass
class LatencyHistogram:
    """Histogram bucket for latency distribution."""
    bucket_ms: float
    count: int = 0


@dataclass
class OperationMetrics:
    """Metrics for a single operation type (e.g., recall, remember)."""
    op_name: str
    total_calls: int = 0
    total_errors: int = 0
    latencies: deque[float] = field(
        default_factory=lambda: deque(maxlen=10_000)
    )
    last_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def p50_ms(self) -> float:
        return _percentile(sorted(self.latencies), 50)

    @property
    def p95_ms(self) -> float:
        return _percentile(sorted(self.latencies), 95)

    @property
    def p99_ms(self) -> float:
        return _percentile(sorted(self.latencies), 99)

    @property
    def error_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_errors / self.total_calls

    @property
    def throughput_per_sec(self) -> float:
        """Approximate throughput based on recent window."""
        if len(self.latencies) < 2:
            return 0.0
        # Use last 100 samples to estimate throughput
        recent = list(self.latencies)[-100:]
        total_time_s = sum(recent) / 1000
        if total_time_s <= 0:
            return 0.0
        return len(recent) / total_time_s

    def record(self, latency_ms: float, success: bool = True) -> None:
        self.total_calls += 1
        if not success:
            self.total_errors += 1
        self.latencies.append(latency_ms)
        self.last_latency_ms = latency_ms
        if latency_ms < self.min_latency_ms:
            self.min_latency_ms = latency_ms
        if latency_ms > self.max_latency_ms:
            self.max_latency_ms = latency_ms


@dataclass
class SystemMetrics:
    """Aggregated system-level performance snapshot."""
    timestamp: float = field(default_factory=time.monotonic)
    operations: dict[str, OperationMetrics] = field(default_factory=dict)
    cache_hit_rate: float = 0.0
    prefetch_hit_rate: float = 0.0
    active_threads: int = 0
    memory_mb: float = 0.0

    @property
    def overall_p99_ms(self) -> float:
        """P99 across all operations."""
        all_latencies: list[float] = []
        for op in self.operations.values():
            all_latencies.extend(op.latencies)
        if not all_latencies:
            return 0.0
        return _percentile(sorted(all_latencies), 99)


class PerformanceMetrics:
    """Centralized performance metrics collector for NeuralMem V1.2.

    Tracks per-operation latencies, computes percentiles, and provides
    P99-focused health checks. Thread-safe.
    """

    def __init__(
        self,
        *,
        p99_target_ms: float = 100.0,
        max_latency_history: int = 10_000,
        enable_histograms: bool = True,
    ) -> None:
        self._p99_target_ms = p99_target_ms
        self._max_latency_history = max_latency_history
        self._enable_histograms = enable_histograms
        self._lock = threading.Lock()
        self._ops: dict[str, OperationMetrics] = {}
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[LatencyHistogram]] = {}
        self._start_time = time.monotonic()

    # ---- recording ----

    def record_latency(
        self,
        op_name: str,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Record a latency sample for an operation."""
        with self._lock:
            op = self._ops.setdefault(
                op_name, OperationMetrics(op_name=op_name)
            )
            op.record(latency_ms, success)

    def record_counter(self, name: str, delta: int = 1) -> None:
        """Increment a named counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + delta

    def record_gauge(self, name: str, value: float) -> None:
        """Set a named gauge."""
        with self._lock:
            self._gauges[name] = value

    def record_histogram(
        self,
        op_name: str,
        latency_ms: float,
        buckets: list[float] | None = None,
    ) -> None:
        """Record latency into histogram buckets."""
        if not self._enable_histograms:
            return
        default_buckets = [1, 5, 10, 25, 50, 75, 100, 150, 200, 500, 1000]
        buckets = buckets or default_buckets
        with self._lock:
            if op_name not in self._histograms:
                self._histograms[op_name] = [
                    LatencyHistogram(bucket_ms=b) for b in buckets
                ]
            for h in self._histograms[op_name]:
                if latency_ms <= h.bucket_ms:
                    h.count += 1
                    break

    def time_operation(self, op_name: str):
        """Context manager for timing an operation."""
        return _TimedOperation(self, op_name)

    # ---- querying ----

    def get_operation_metrics(self, op_name: str) -> OperationMetrics | None:
        """Get metrics for a specific operation."""
        with self._lock:
            op = self._ops.get(op_name)
            if op is None:
                return None
            # Return a copy to avoid external mutation
            return OperationMetrics(
                op_name=op.op_name,
                total_calls=op.total_calls,
                total_errors=op.total_errors,
                latencies=deque(op.latencies, maxlen=self._max_latency_history),
                last_latency_ms=op.last_latency_ms,
                min_latency_ms=op.min_latency_ms,
                max_latency_ms=op.max_latency_ms,
            )

    def get_all_operations(self) -> dict[str, OperationMetrics]:
        """Get metrics for all operations."""
        with self._lock:
            return {
                name: OperationMetrics(
                    op_name=op.op_name,
                    total_calls=op.total_calls,
                    total_errors=op.total_errors,
                    latencies=deque(op.latencies, maxlen=self._max_latency_history),
                    last_latency_ms=op.last_latency_ms,
                    min_latency_ms=op.min_latency_ms,
                    max_latency_ms=op.max_latency_ms,
                )
                for name, op in self._ops.items()
            }

    def get_system_metrics(self) -> SystemMetrics:
        """Get aggregated system-level metrics snapshot."""
        with self._lock:
            ops = {
                name: OperationMetrics(
                    op_name=op.op_name,
                    total_calls=op.total_calls,
                    total_errors=op.total_errors,
                    latencies=deque(op.latencies, maxlen=self._max_latency_history),
                    last_latency_ms=op.last_latency_ms,
                    min_latency_ms=op.min_latency_ms,
                    max_latency_ms=op.max_latency_ms,
                )
                for name, op in self._ops.items()
            }
            return SystemMetrics(
                timestamp=time.monotonic(),
                operations=ops,
                cache_hit_rate=self._gauges.get("cache_hit_rate", 0.0),
                prefetch_hit_rate=self._gauges.get("prefetch_hit_rate", 0.0),
                active_threads=self._gauges.get("active_threads", 0.0),
                memory_mb=self._gauges.get("memory_mb", 0.0),
            )

    def is_healthy(self) -> tuple[bool, str]:
        """Check if system meets P99 latency target.

        Returns (is_healthy, reason).
        """
        system = self.get_system_metrics()
        overall_p99 = system.overall_p99_ms
        if overall_p99 > self._p99_target_ms * 1.5:
            return (
                False,
                f"P99 latency {overall_p99:.1f}ms exceeds "
                f"target {self._p99_target_ms:.1f}ms by 50%+",
            )
        if overall_p99 > self._p99_target_ms:
            return (
                True,
                f"P99 latency {overall_p99:.1f}ms slightly above "
                f"target {self._p99_target_ms:.1f}ms",
            )
        return (
            True,
            f"P99 latency {overall_p99:.1f}ms within target "
            f"{self._p99_target_ms:.1f}ms",
        )

    def summary(self) -> dict[str, Any]:
        """Human-readable performance summary."""
        system = self.get_system_metrics()
        healthy, reason = self.is_healthy()
        ops_summary = {}
        for name, op in system.operations.items():
            ops_summary[name] = {
                "calls": op.total_calls,
                "errors": op.total_errors,
                "error_rate": round(op.error_rate, 4),
                "avg_ms": round(op.avg_latency_ms, 2),
                "p50_ms": round(op.p50_ms, 2),
                "p95_ms": round(op.p95_ms, 2),
                "p99_ms": round(op.p99_ms, 2),
                "min_ms": round(op.min_latency_ms, 2) if op.latencies else 0,
                "max_ms": round(op.max_latency_ms, 2) if op.latencies else 0,
                "throughput": round(op.throughput_per_sec, 2),
            }
        return {
            "healthy": healthy,
            "health_reason": reason,
            "overall_p99_ms": round(system.overall_p99_ms, 2),
            "uptime_seconds": round(time.monotonic() - self._start_time, 2),
            "operations": ops_summary,
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
        }

    def reset(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._ops.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._start_time = time.monotonic()


class _TimedOperation:
    """Context manager used by PerformanceMetrics.time_operation()."""

    def __init__(self, metrics: PerformanceMetrics, op_name: str) -> None:
        self._metrics = metrics
        self._op_name = op_name
        self._t0: float = 0.0
        self._success: bool = True

    def __enter__(self) -> "_TimedOperation":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        latency_ms = (time.monotonic() - self._t0) * 1000
        self._success = exc_type is None
        self._metrics.record_latency(
            self._op_name, latency_ms, success=self._success
        )
        self._metrics.record_histogram(self._op_name, latency_ms)
        if not self._success:
            self._metrics.record_counter(f"{self._op_name}.errors")
