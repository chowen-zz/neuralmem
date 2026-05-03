"""Tests for neuralmem.ops.health — HealthChecker."""
from __future__ import annotations

import pytest

from neuralmem.ops.health import HealthChecker, HealthReport, HealthStatus, HealthThresholds

# ---------------------------------------------------------------------------
# Helpers / mocks
# ---------------------------------------------------------------------------


class MockStorage:
    """Minimal storage mock for health checks."""

    def __init__(
        self,
        stats: dict | None = None,
        raise_on_stats: bool = False,
    ) -> None:
        self._stats = stats or {"total": 42, "by_type": {}, "entity_count": 5}
        self._raise = raise_on_stats

    def get_stats(self, user_id=None):
        if self._raise:
            raise ConnectionError("DB locked")
        return self._stats

    def load_graph_snapshot(self):
        return {"nodes": {"n1": {}, "n2": {}}, "edges": []}


class MockEmbedder:
    """Minimal embedder mock."""

    def __init__(self, raise_on_encode: bool = False, empty: bool = False):
        self._raise = raise_on_encode
        self._empty = empty

    def encode(self, texts: list[str]):
        if self._raise:
            raise RuntimeError("Model not loaded")
        if self._empty:
            return []
        return [[0.1, 0.2, 0.3]]


class MockGraph:
    """Minimal graph mock."""

    def __init__(self, storage=None):
        self._storage = storage


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthChecker:
    def test_healthy_all_pass(self):
        storage = MockStorage()
        embedder = MockEmbedder()
        graph = MockGraph(storage)
        checker = HealthChecker(storage=storage, embedder=embedder, graph=graph)
        report = checker.check()
        assert report.status is HealthStatus.HEALTHY
        assert report.checks["storage"] == "healthy"
        assert report.checks["embedder"] == "healthy"
        assert report.checks["graph"] == "healthy"
        assert report.checks["memory_count"] == "healthy"

    def test_unhealthy_storage_down(self):
        storage = MockStorage(raise_on_stats=True)
        checker = HealthChecker(storage=storage)
        report = checker.check()
        assert report.status is HealthStatus.UNHEALTHY
        assert report.checks["storage"] == "unhealthy"

    def test_degraded_no_embedder(self):
        storage = MockStorage()
        checker = HealthChecker(storage=storage, embedder=None)
        report = checker.check()
        assert report.checks["embedder"] == "degraded"
        # Storage is healthy, so overall is degraded
        assert report.status is HealthStatus.DEGRADED

    def test_unhealthy_embedder_broken(self):
        storage = MockStorage()
        embedder = MockEmbedder(raise_on_encode=True)
        checker = HealthChecker(storage=storage, embedder=embedder)
        report = checker.check()
        assert report.checks["embedder"] == "unhealthy"
        assert report.status is HealthStatus.UNHEALTHY

    def test_unhealthy_embedder_empty_result(self):
        storage = MockStorage()
        embedder = MockEmbedder(empty=True)
        checker = HealthChecker(storage=storage, embedder=embedder)
        report = checker.check()
        assert report.checks["embedder"] == "unhealthy"

    def test_degraded_no_graph(self):
        storage = MockStorage()
        checker = HealthChecker(storage=storage, embedder=MockEmbedder(), graph=None)
        report = checker.check()
        assert report.checks["graph"] == "degraded"

    def test_degraded_memory_count_exceeds_max(self):
        storage = MockStorage(stats={"total": 2_000_000})
        thresholds = HealthThresholds(max_memory_count=1_000_000)
        checker = HealthChecker(storage=storage, thresholds=thresholds)
        report = checker.check()
        assert report.checks["memory_count"] == "degraded"
        assert "exceeds" in report.details["memory_count"]

    def test_healthy_memory_count_within_bounds(self):
        storage = MockStorage(stats={"total": 100})
        thresholds = HealthThresholds(max_memory_count=500, min_memory_count=1)
        checker = HealthChecker(storage=storage, thresholds=thresholds)
        report = checker.check()
        assert report.checks["memory_count"] == "healthy"

    def test_degraded_memory_count_below_min(self):
        storage = MockStorage(stats={"total": 0})
        thresholds = HealthThresholds(min_memory_count=1)
        checker = HealthChecker(storage=storage, thresholds=thresholds)
        report = checker.check()
        assert report.checks["memory_count"] == "degraded"

    def test_report_has_details(self):
        storage = MockStorage()
        checker = HealthChecker(storage=storage)
        report = checker.check()
        assert isinstance(report.details, dict)
        assert "storage" in report.details

    def test_worst_status_wins(self):
        """UNHEALTHY storage overrides HEALTHY everything else."""
        storage = MockStorage(raise_on_stats=True)
        checker = HealthChecker(storage=storage, embedder=MockEmbedder())
        report = checker.check()
        assert report.status is HealthStatus.UNHEALTHY

    def test_default_thresholds(self):
        th = HealthThresholds()
        assert th.max_memory_count == 1_000_000
        assert th.min_memory_count == 0

    @pytest.mark.asyncio
    async def test_acheck_returns_same_as_sync(self):
        storage = MockStorage()
        graph = MockGraph(storage)
        checker = HealthChecker(
            storage=storage, embedder=MockEmbedder(), graph=graph
        )
        report = await checker.acheck()
        assert isinstance(report, HealthReport)
        assert report.status is HealthStatus.HEALTHY

    def test_embedder_missing_encode_method(self):
        storage = MockStorage()
        checker = HealthChecker(storage=storage, embedder=object())
        report = checker.check()
        assert report.checks["embedder"] == "unhealthy"
        assert "missing" in report.details["embedder"].lower()

    def test_graph_exception_becomes_degraded(self):
        class BrokenGraph:
            def __init__(self):
                self._storage = self

            def load_graph_snapshot(self):
                raise RuntimeError("corrupt")

        storage = MockStorage()
        checker = HealthChecker(
            storage=storage, graph=BrokenGraph()
        )
        report = checker.check()
        assert report.checks["graph"] == "degraded"
