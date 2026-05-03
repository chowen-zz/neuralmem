"""NeuralMem Dashboard Server — FastAPI application with REST API routes.

Provides health checks, memory browsing, knowledge graph stats,
metrics exposure, and recall endpoints for the web dashboard.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

if TYPE_CHECKING:
    from neuralmem.core.memory import NeuralMem

_logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


class DashboardServer:
    """Dashboard server that creates a FastAPI application with API routes.

    Parameters
    ----------
    mem:
        A ``NeuralMem`` instance used to back all API endpoints.
    """

    def __init__(self, mem: NeuralMem) -> None:
        self.mem = mem
        self.app = FastAPI(
            title="NeuralMem Dashboard",
            version="0.7.0",
        )
        self._setup_routes()

    # ------------------------------------------------------------------
    # Route setup
    # ------------------------------------------------------------------

    def _setup_routes(self) -> None:
        """Register all API and static-file routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            """Serve the dashboard single-page application."""
            html_path = _STATIC_DIR / "index.html"
            return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

        @self.app.get("/api/health")
        async def api_health() -> JSONResponse:
            """Return health check results."""
            from neuralmem.ops.health import HealthChecker

            checker = HealthChecker(
                storage=self.mem.storage,
                embedder=self.mem.embedding,
                graph=self.mem.graph,
            )
            report = checker.check()
            return JSONResponse(
                {
                    "status": report.status.value,
                    "checks": report.checks,
                    "details": report.details,
                }
            )

        @self.app.get("/api/metrics")
        async def api_metrics() -> JSONResponse:
            """Return collected metrics as JSON."""
            metrics_data = self.mem.metrics.get_metrics()
            return JSONResponse(metrics_data)

        @self.app.get("/api/memories")
        async def api_memories(
            limit: int = Query(default=50, ge=1, le=500),
            offset: int = Query(default=0, ge=0),
        ) -> JSONResponse:
            """List memories with pagination."""
            all_memories = self.mem.storage.list_memories(limit=limit + offset)
            page = all_memories[offset : offset + limit]
            items = []
            for m in page:
                items.append(
                    {
                        "id": m.id,
                        "content": m.content,
                        "memory_type": m.memory_type.value,
                        "scope": m.scope.value,
                        "user_id": m.user_id,
                        "importance": m.importance,
                        "is_active": m.is_active,
                        "tags": list(m.tags),
                        "created_at": m.created_at.isoformat(),
                        "access_count": m.access_count,
                    }
                )
            return JSONResponse(
                {
                    "memories": items,
                    "total": len(all_memories),
                    "limit": limit,
                    "offset": offset,
                }
            )

        @self.app.get("/api/graph/stats")
        async def api_graph_stats() -> JSONResponse:
            """Return knowledge graph statistics."""
            stats = self.mem.graph.get_stats()
            return JSONResponse(stats)

        @self.app.post("/api/recall")
        async def api_recall(request: Request) -> JSONResponse:
            """Search memories via recall endpoint."""
            body = await request.json()
            query = body.get("query", "")
            user_id = body.get("user_id")
            limit = body.get("limit", 10)

            if not query:
                return JSONResponse(
                    {"error": "query is required"}, status_code=400
                )

            results = self.mem.recall(
                query, user_id=user_id, limit=limit
            )
            items = []
            for r in results:
                items.append(
                    {
                        "memory": {
                            "id": r.memory.id,
                            "content": r.memory.content,
                            "memory_type": r.memory.memory_type.value,
                            "importance": r.memory.importance,
                            "tags": list(r.memory.tags),
                        },
                        "score": r.score,
                        "retrieval_method": r.retrieval_method,
                        "explanation": r.explanation,
                    }
                )
            return JSONResponse({"results": items, "count": len(items)})

        @self.app.get("/api/config")
        async def api_config() -> JSONResponse:
            """Return current NeuralMem configuration (non-sensitive fields)."""
            cfg = self.mem.config
            safe_fields = {
                "db_path": cfg.db_path,
                "embedding_model": cfg.embedding_model,
                "embedding_provider": cfg.embedding_provider,
                "embedding_dim": cfg.embedding_dim,
                "conflict_threshold_low": cfg.conflict_threshold_low,
                "conflict_threshold_high": cfg.conflict_threshold_high,
                "enable_importance_reinforcement": cfg.enable_importance_reinforcement,
                "reinforcement_boost": cfg.reinforcement_boost,
                "enable_reranker": cfg.enable_reranker,
                "default_search_limit": cfg.default_search_limit,
                "min_score": cfg.min_score,
                "enable_metrics": cfg.enable_metrics,
                "llm_extractor": cfg.llm_extractor,
            }
            return JSONResponse(safe_fields)

        # Mount static files last so API routes take priority
        self.app.mount(
            "/static",
            StaticFiles(directory=str(_STATIC_DIR)),
            name="dashboard-static",
        )

    def get_app(self) -> FastAPI:
        """Return the configured FastAPI application."""
        return self.app
