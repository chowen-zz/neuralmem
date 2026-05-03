"""Health checker for NeuralMem operational monitoring.

Checks storage connectivity, embedding model availability, graph snapshot
loadability, and memory count bounds. Returns a structured HealthReport.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Overall system health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthReport:
    """Result of a health check."""
    status: HealthStatus
    checks: dict[str, Any] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)


@dataclass
class HealthThresholds:
    """Configurable thresholds for health checks."""
    max_memory_count: int = 1_000_000
    min_memory_count: int = 0


class HealthChecker:
    """Operational health checker for NeuralMem.

    Supports both synchronous ``check()`` and asynchronous ``acheck()``
    invocations. Configurable thresholds allow operators to tune
    what constitutes "healthy" vs "degraded".

    Parameters
    ----------
    storage:
        A storage backend instance (e.g. ``SQLiteStorage``).
    embedder:
        The embedding model / provider object.  Checked via
        ``hasattr(embedder, 'encode')`` and a probe encode call.
    graph:
        A ``KnowledgeGraph`` instance whose snapshot is loadable.
    thresholds:
        Optional custom health thresholds.
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedder: object = None,
        graph: object = None,
        thresholds: HealthThresholds | None = None,
    ) -> None:
        self.storage = storage
        self.embedder = embedder
        self.graph = graph
        self.thresholds = thresholds or HealthThresholds()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> HealthReport:
        """Run all health checks synchronously and return a report."""
        checks: dict[str, Any] = {}
        details: dict[str, str] = {}
        worst = HealthStatus.HEALTHY

        # 1. Storage connectivity
        status, detail = self._check_storage()
        checks["storage"] = status.value
        details["storage"] = detail
        worst = self._worsen(worst, status)

        # 2. Embedding model
        status, detail = self._check_embedder()
        checks["embedder"] = status.value
        details["embedder"] = detail
        worst = self._worsen(worst, status)

        # 3. Graph snapshot
        status, detail = self._check_graph()
        checks["graph"] = status.value
        details["graph"] = detail
        worst = self._worsen(worst, status)

        # 4. Memory count
        status, detail = self._check_memory_count()
        checks["memory_count"] = status.value
        details["memory_count"] = detail
        worst = self._worsen(worst, status)

        return HealthReport(status=worst, checks=checks, details=details)

    async def acheck(self) -> HealthReport:
        """Run health checks asynchronously (delegates to sync checks in a
        thread pool so storage I/O doesn't block the event loop)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.check)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_storage(self) -> tuple[HealthStatus, str]:
        """Verify storage is reachable by calling ``get_stats``."""
        try:
            stats = self.storage.get_stats()
            total = stats.get("total", "?")
            return HealthStatus.HEALTHY, (
                f"Storage OK — {total} memories"
            )
        except Exception as exc:
            _logger.warning("Storage health check failed: %s", exc)
            return HealthStatus.UNHEALTHY, f"Storage unreachable: {exc}"

    def _check_embedder(self) -> tuple[HealthStatus, str]:
        """Check that the embedding model is available and functional."""
        if self.embedder is None:
            return HealthStatus.DEGRADED, "No embedder configured"

        if not hasattr(self.embedder, "encode"):
            return HealthStatus.UNHEALTHY, (
                "Embedder missing 'encode' method"
            )

        try:
            vec = self.embedder.encode(["health-check-probe"])
            if vec is None or len(vec) == 0:
                return HealthStatus.UNHEALTHY, (
                    "Embedder returned empty result"
                )
            dim = len(vec[0]) if isinstance(vec[0], (list, tuple)) else "?"
            return HealthStatus.HEALTHY, (
                f"Embedder OK — dimension={dim}"
            )
        except Exception as exc:
            _logger.warning("Embedder health check failed: %s", exc)
            return HealthStatus.UNHEALTHY, f"Embedder error: {exc}"

    def _check_graph(self) -> tuple[HealthStatus, str]:
        """Check that the graph snapshot is loadable."""
        if self.graph is None:
            return HealthStatus.DEGRADED, "No graph configured"

        try:
            # KnowledgeGraph exposes load_graph_snapshot via storage
            if hasattr(self.graph, "_storage"):
                snap = self.graph._storage.load_graph_snapshot()
            elif hasattr(self.graph, "load_snapshot"):
                snap = self.graph.load_snapshot()
            else:
                snap = None

            if snap is not None:
                node_count = len(snap.get("nodes", {}))
                return HealthStatus.HEALTHY, (
                    f"Graph snapshot loaded — {node_count} nodes"
                )
            return HealthStatus.HEALTHY, "Graph snapshot empty or not yet persisted"
        except Exception as exc:
            _logger.warning("Graph health check failed: %s", exc)
            return HealthStatus.DEGRADED, f"Graph snapshot error: {exc}"

    def _check_memory_count(self) -> tuple[HealthStatus, str]:
        """Verify memory count is within configured bounds."""
        try:
            stats = self.storage.get_stats()
            total = int(stats.get("total", 0))
        except Exception as exc:
            return HealthStatus.UNHEALTHY, (
                f"Cannot read memory count: {exc}"
            )

        th = self.thresholds
        if total > th.max_memory_count:
            return HealthStatus.DEGRADED, (
                f"Memory count {total} exceeds max {th.max_memory_count}"
            )
        if total < th.min_memory_count:
            return HealthStatus.DEGRADED, (
                f"Memory count {total} below min {th.min_memory_count}"
            )
        return HealthStatus.HEALTHY, (
            f"Memory count {total} within bounds"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _worsen(
        current: HealthStatus, candidate: HealthStatus
    ) -> HealthStatus:
        """Return the worse of two statuses."""
        order = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.UNHEALTHY: 2,
        }
        if order[candidate] > order[current]:
            return candidate
        return current
