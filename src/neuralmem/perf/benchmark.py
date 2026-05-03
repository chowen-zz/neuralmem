"""Latency benchmarking for NeuralMem recall() and bulk write."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


def _percentile(sorted_data: list[float], p: float) -> float:
    """Compute p-th percentile (0-100) from a pre-sorted list.

    Nearest-rank method, numpy-free.
    """
    if not sorted_data:
        return 0.0
    k = max(0, min(len(sorted_data) - 1, int(len(sorted_data) * p / 100) - 1))
    return sorted_data[k]


@dataclass
class BenchmarkReport:
    """Statistical summary of collected latency samples."""
    samples: list[float] = field(default_factory=list)
    label: str = ""

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def max(self) -> float:
        return max(self.samples) if self.samples else 0.0

    @property
    def min(self) -> float:
        return min(self.samples) if self.samples else 0.0

    @property
    def p50(self) -> float:
        return _percentile(sorted(self.samples), 50)

    @property
    def p95(self) -> float:
        return _percentile(sorted(self.samples), 95)

    @property
    def p99(self) -> float:
        return _percentile(sorted(self.samples), 99)

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"[{self.label}] n={self.count}  "
            f"mean={self.mean*1000:.1f}ms  "
            f"p50={self.p50*1000:.1f}ms  "
            f"p95={self.p95*1000:.1f}ms  "
            f"p99={self.p99*1000:.1f}ms  "
            f"max={self.max*1000:.1f}ms"
        )


class LatencyBenchmark:
    """Run recall() N times and report latency percentiles.

    Also supports ``benchmark_bulk_write`` for write-throughput measurement.
    """

    def __init__(self, neural_mem: Any) -> None:
        self._mem = neural_mem
        self.timings: list[float] = []

    def benchmark_recall(
        self,
        query: str,
        *,
        iterations: int = 50,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> BenchmarkReport:
        """Run ``recall()`` *iterations* times and collect latencies.

        Args:
            query: Search query string.
            iterations: Number of times to run recall().
            user_id: Optional user scope.
            **kwargs: Extra args forwarded to recall().

        Returns:
            BenchmarkReport with raw samples and computed percentiles.
        """
        self.timings.clear()
        for _ in range(iterations):
            t0 = time.monotonic()
            self._mem.recall(query, user_id=user_id, **kwargs)
            elapsed = time.monotonic() - t0
            self.timings.append(elapsed)

        return BenchmarkReport(
            samples=list(self.timings), label="recall"
        )

    def benchmark_bulk_write(
        self,
        count: int = 100,
        *,
        user_id: str | None = None,
        prefix: str = "benchmark",
        **kwargs: Any,
    ) -> BenchmarkReport:
        """Write *count* memories sequentially and measure throughput.

        Args:
            count: Number of memories to write.
            user_id: Optional user scope.
            prefix: Content prefix for generated items.
            **kwargs: Extra args forwarded to remember().

        Returns:
            BenchmarkReport with per-write latencies.
        """
        timings: list[float] = []
        for i in range(count):
            content = f"{prefix} item {i}"
            t0 = time.monotonic()
            self._mem.remember(content, user_id=user_id, **kwargs)
            elapsed = time.monotonic() - t0
            timings.append(elapsed)

        return BenchmarkReport(samples=timings, label="bulk_write")
