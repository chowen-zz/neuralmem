"""Built-in plugins for NeuralMem V1.3 plugin system.

Provides LoggingPlugin, MetricsPlugin, and ValidationPlugin.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from neuralmem.core.types import Memory, SearchResult
from neuralmem.plugins.base import Plugin

_logger = logging.getLogger(__name__)


class LoggingPlugin(Plugin):
    """Logs every remember / recall / reflect event.

    Useful for audit trails and debugging the plugin pipeline.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or _logger

    @property
    def name(self) -> str:
        return "logging"

    @property
    def priority(self) -> int:
        return 5

    def on_remember(self, memory: Memory) -> Memory:
        self._logger.info(
            "[LoggingPlugin] remember: id=%s type=%s user=%s",
            memory.id,
            memory.memory_type,
            memory.user_id,
        )
        return memory

    def on_recall(self, results: list[SearchResult]) -> list[SearchResult]:
        self._logger.info(
            "[LoggingPlugin] recall: %d results",
            len(results),
        )
        return results

    def on_reflect(self, memory: Memory) -> Memory:
        self._logger.info(
            "[LoggingPlugin] reflect: id=%s",
            memory.id,
        )
        return memory


class MetricsPlugin(Plugin):
    """Collects timing and count metrics for memory operations.

    Stores counters and latencies in an internal dict that can be
    flushed to an external metrics backend.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, Any] = {
            "remember_count": 0,
            "recall_count": 0,
            "reflect_count": 0,
            "remember_latency_ms": [],
            "recall_latency_ms": [],
            "reflect_latency_ms": [],
        }

    @property
    def name(self) -> str:
        return "metrics"

    @property
    def priority(self) -> int:
        return 10

    def on_remember(self, memory: Memory) -> Memory:
        start = time.perf_counter()
        self._metrics["remember_count"] += 1
        self._metrics["remember_latency_ms"].append(
            (time.perf_counter() - start) * 1000
        )
        return memory

    def on_recall(self, results: list[SearchResult]) -> list[SearchResult]:
        start = time.perf_counter()
        self._metrics["recall_count"] += 1
        self._metrics["recall_latency_ms"].append(
            (time.perf_counter() - start) * 1000
        )
        return results

    def on_reflect(self, memory: Memory) -> Memory:
        start = time.perf_counter()
        self._metrics["reflect_count"] += 1
        self._metrics["reflect_latency_ms"].append(
            (time.perf_counter() - start) * 1000
        )
        return memory

    def flush(self) -> dict[str, Any]:
        """Return current metrics and reset counters."""
        snapshot = dict(self._metrics)
        # Reset latencies but keep counts so they are monotonic
        self._metrics["remember_latency_ms"] = []
        self._metrics["recall_latency_ms"] = []
        self._metrics["reflect_latency_ms"] = []
        return snapshot


class ValidationPlugin(Plugin):
    """Validates memory content before storage and after recall.

    Rejects empty content, overly long content, or memories missing
    required fields.  Can be configured with custom validators.
    """

    def __init__(
        self,
        max_content_length: int = 100_000,
        required_fields: tuple[str, ...] = ("content",),
    ) -> None:
        self._max_content_length = max_content_length
        self._required_fields = required_fields

    @property
    def name(self) -> str:
        return "validation"

    @property
    def priority(self) -> int:
        return 1

    def on_remember(self, memory: Memory) -> Memory:
        for field in self._required_fields:
            value = getattr(memory, field, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(
                    f"ValidationPlugin: memory field '{field}' is empty"
                )
        if len(memory.content) > self._max_content_length:
            raise ValueError(
                f"ValidationPlugin: content exceeds {self._max_content_length} chars"
            )
        return memory

    def on_recall(self, results: list[SearchResult]) -> list[SearchResult]:
        # Filter out results with invalid memories
        valid: list[SearchResult] = []
        for result in results:
            mem = result.memory
            if mem.content and len(mem.content) <= self._max_content_length:
                valid.append(result)
        return valid

    def on_reflect(self, memory: Memory) -> Memory:
        # Same rules as remember for reflection
        return self.on_remember(memory)
