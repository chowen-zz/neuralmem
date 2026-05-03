"""Tests for the NeuralMem Dashboard Web UI (DashboardServer routes).

All tests use mocked NeuralMem internals — no real database or embedding
model is required.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from neuralmem.core.types import (
    Memory,
    MemoryScope,
    MemoryType,
    SearchResult,
)
from neuralmem.dashboard.server import DashboardServer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_memory(
    content: str = "test memory",
    mem_type: MemoryType = MemoryType.SEMANTIC,
    user_id: str | None = "user-1",
    importance: float = 0.5,
) -> Memory:
    """Create a frozen Memory instance for testing."""
    now = datetime.now(timezone.utc)
    return Memory(
        id="01TEST1234567890ABCDEF01",
        content=content,
        memory_type=mem_type,
        scope=MemoryScope.USER,
        user_id=user_id,
        importance=importance,
        tags=("test",),
        is_active=True,
        created_at=now,
        updated_at=now,
        last_accessed=now,
    )


@pytest.fixture
def mock_mem():
    """Create a mocked NeuralMem instance."""
    mem = MagicMock()
    mem.config = MagicMock(
        db_path="/tmp/test.db",
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="local",
        embedding_dim=384,
        conflict_threshold_low=0.75,
        conflict_threshold_high=0.95,
        enable_importance_reinforcement=True,
        reinforcement_boost=0.05,
        enable_reranker=False,
        default_search_limit=10,
        min_score=0.3,
        enable_metrics=False,
        llm_extractor="none",
    )
    mem.storage = MagicMock()
    mem.embedding = MagicMock()
    mem.graph = MagicMock()
    mem.metrics = MagicMock()
    mem.metrics.get_metrics.return_value = {
        "counters": {"neuralmem.recall.calls": 5},
        "histograms": {},
    }
    return mem


@pytest.fixture
def client(mock_mem):
    """Create a TestClient wrapping the DashboardServer."""
    server = DashboardServer(mock_mem)
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# Tests: GET /
# ---------------------------------------------------------------------------


class TestIndexRoute:
    """Tests for the root index route."""

    def test_index_returns_html(self, client):
        """GET / should return the dashboard HTML page."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "NeuralMem Dashboard" in resp.text

    def test_index_contains_sidebar(self, client):
        """Index page should contain sidebar navigation."""
        resp = client.get("/")
        assert "Overview" in resp.text
        assert "Memories" in resp.text
        assert "Graph" in resp.text
        assert "Settings" in resp.text


# ---------------------------------------------------------------------------
# Tests: GET /api/health
# ---------------------------------------------------------------------------


class TestHealthRoute:
    """Tests for the health check endpoint."""

    @patch("neuralmem.ops.health.HealthChecker")
    def test_health_returns_status(self, mock_checker_cls, client):
        """GET /api/health should return health status."""
        mock_report = MagicMock()
        mock_report.status.value = "healthy"
        mock_report.checks = {"storage": "healthy"}
        mock_report.details = {"storage": "OK"}
        mock_checker_cls.return_value.check.return_value = mock_report

        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "storage" in data["checks"]

    @patch("neuralmem.ops.health.HealthChecker")
    def test_health_degraded_status(self, mock_checker_cls, client):
        """Health endpoint should propagate degraded status."""
        mock_report = MagicMock()
        mock_report.status.value = "degraded"
        mock_report.checks = {"embedder": "degraded"}
        mock_report.details = {"embedder": "No embedder"}
        mock_checker_cls.return_value.check.return_value = mock_report

        resp = client.get("/api/health")
        data = resp.json()
        assert data["status"] == "degraded"


# ---------------------------------------------------------------------------
# Tests: GET /api/metrics
# ---------------------------------------------------------------------------


class TestMetricsRoute:
    """Tests for the metrics endpoint."""

    def test_metrics_returns_json(self, client, mock_mem):
        """GET /api/metrics should return metrics JSON."""
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "counters" in data
        assert data["counters"]["neuralmem.recall.calls"] == 5

    def test_metrics_calls_get_metrics(self, client, mock_mem):
        """Metrics endpoint should call mem.metrics.get_metrics()."""
        client.get("/api/metrics")
        mock_mem.metrics.get_metrics.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: GET /api/memories
# ---------------------------------------------------------------------------


class TestMemoriesRoute:
    """Tests for the memories listing endpoint."""

    def test_memories_returns_list(self, client, mock_mem):
        """GET /api/memories should return a paginated list."""
        mock_mem.storage.list_memories.return_value = [
            _make_memory("memory one"),
            _make_memory("memory two"),
        ]
        resp = client.get("/api/memories?limit=10&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["memories"]) == 2
        assert data["memories"][0]["content"] == "memory one"

    def test_memories_pagination(self, client, mock_mem):
        """Memories endpoint should respect limit and offset."""
        memories = [_make_memory(f"mem {i}") for i in range(10)]
        mock_mem.storage.list_memories.return_value = memories

        resp = client.get("/api/memories?limit=3&offset=2")
        data = resp.json()
        assert len(data["memories"]) == 3
        assert data["memories"][0]["content"] == "mem 2"

    def test_memories_empty(self, client, mock_mem):
        """Memories endpoint should handle empty storage."""
        mock_mem.storage.list_memories.return_value = []
        resp = client.get("/api/memories")
        data = resp.json()
        assert data["total"] == 0
        assert data["memories"] == []

    def test_memories_serializes_fields(self, client, mock_mem):
        """Memory items should include expected fields."""
        mem = _make_memory("hello world", MemoryType.FACT, importance=0.8)
        mock_mem.storage.list_memories.return_value = [mem]

        resp = client.get("/api/memories")
        item = resp.json()["memories"][0]
        assert item["id"] == mem.id
        assert item["memory_type"] == "fact"
        assert item["importance"] == 0.8
        assert item["tags"] == ["test"]
        assert item["is_active"] is True


# ---------------------------------------------------------------------------
# Tests: GET /api/graph/stats
# ---------------------------------------------------------------------------


class TestGraphStatsRoute:
    """Tests for the graph stats endpoint."""

    def test_graph_stats_returns_counts(self, client, mock_mem):
        """GET /api/graph/stats should return node and edge counts."""
        mock_mem.graph.get_stats.return_value = {
            "node_count": 42,
            "edge_count": 78,
        }
        resp = client.get("/api/graph/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_count"] == 42
        assert data["edge_count"] == 78

    def test_graph_stats_empty(self, client, mock_mem):
        """Graph stats should handle empty graph."""
        mock_mem.graph.get_stats.return_value = {
            "node_count": 0,
            "edge_count": 0,
        }
        resp = client.get("/api/graph/stats")
        data = resp.json()
        assert data["node_count"] == 0


# ---------------------------------------------------------------------------
# Tests: POST /api/recall
# ---------------------------------------------------------------------------


class TestRecallRoute:
    """Tests for the recall search endpoint."""

    def test_recall_returns_results(self, client, mock_mem):
        """POST /api/recall should return search results."""
        result = SearchResult(
            memory=_make_memory("relevant fact"),
            score=0.92,
            retrieval_method="semantic",
            explanation="high similarity",
        )
        mock_mem.recall.return_value = [result]

        resp = client.post(
            "/api/recall",
            json={"query": "test query", "user_id": "user-1", "limit": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["results"][0]["score"] == 0.92
        assert data["results"][0]["memory"]["content"] == "relevant fact"

    def test_recall_empty_query_returns_400(self, client):
        """POST /api/recall with empty query should return 400."""
        resp = client.post("/api/recall", json={"query": ""})
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_recall_calls_mem_recall(self, client, mock_mem):
        """Recall endpoint should call mem.recall with correct args."""
        mock_mem.recall.return_value = []
        client.post(
            "/api/recall",
            json={"query": "hello", "user_id": "u1", "limit": 3},
        )
        mock_mem.recall.assert_called_once_with(
            "hello", user_id="u1", limit=3,
        )

    def test_recall_no_results(self, client, mock_mem):
        """Recall endpoint should handle no results gracefully."""
        mock_mem.recall.return_value = []
        resp = client.post("/api/recall", json={"query": "nothing"})
        data = resp.json()
        assert data["count"] == 0
        assert data["results"] == []


# ---------------------------------------------------------------------------
# Tests: GET /api/config
# ---------------------------------------------------------------------------


class TestConfigRoute:
    """Tests for the config endpoint."""

    def test_config_returns_fields(self, client):
        """GET /api/config should return safe config fields."""
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "db_path" in data
        assert "embedding_model" in data
        assert data["embedding_model"] == "all-MiniLM-L6-v2"

    def test_config_excludes_secrets(self, client):
        """Config endpoint should not expose API keys."""
        resp = client.get("/api/config")
        data = resp.json()
        assert "openai_api_key" not in data
        assert "anthropic_api_key" not in data


# ---------------------------------------------------------------------------
# Tests: Static files
# ---------------------------------------------------------------------------


class TestStaticFiles:
    """Tests for static file serving."""

    def test_static_css_accessible(self, client):
        """Static CSS file should be accessible."""
        resp = client.get("/static/style.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"] or \
               "text/plain" in resp.headers["content-type"]

    def test_static_js_accessible(self, client):
        """Static JS file should be accessible."""
        resp = client.get("/static/app.js")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests: DashboardServer class
# ---------------------------------------------------------------------------


class TestDashboardServer:
    """Tests for the DashboardServer class itself."""

    def test_creates_fastapi_app(self, mock_mem):
        """DashboardServer should create a FastAPI application."""
        server = DashboardServer(mock_mem)
        assert server.app is not None
        assert server.app.title == "NeuralMem Dashboard"

    def test_get_app_returns_app(self, mock_mem):
        """get_app() should return the FastAPI app."""
        server = DashboardServer(mock_mem)
        app = server.get_app()
        assert app is server.app

    def test_stores_mem_reference(self, mock_mem):
        """DashboardServer should store the NeuralMem reference."""
        server = DashboardServer(mock_mem)
        assert server.mem is mock_mem
