"""NeuralMem Dashboard API — FastAPI wrapper around NeuralMem core.

Routes:
  GET  /api/health        — Health check
  GET  /api/memories      — Paginated memory list
  POST /api/search        — Semantic search via recall
  GET  /api/stats         — Storage + graph + metrics stats
  GET  /api/graph         — Knowledge graph nodes + edges
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from neuralmem.core.memory import NeuralMem

_logger = logging.getLogger(__name__)

app = FastAPI(title="NeuralMem Dashboard API", version="1.6.0")

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global NeuralMem instance (injected at startup)
_mem: NeuralMem | None = None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    user_id: str | None = None
    agent_id: str | None = None
    memory_types: list[str] | None = None
    tags: list[str] | None = None
    limit: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    results: list[dict]
    count: int
    query: str


class MemoryListResponse(BaseModel):
    memories: list[dict]
    total: int
    limit: int
    offset: int


class StatsResponse(BaseModel):
    memory_count: int
    node_count: int
    edge_count: int
    p99_latency_ms: float
    p95_latency_ms: float
    p50_latency_ms: float
    mean_latency_ms: float
    recall_calls: int
    remember_calls: int
    cache_hit_rate: float
    active_memories: int
    superseded_memories: int


class GraphResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]
    node_count: int
    edge_count: int


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_memory(m) -> dict:
    """Serialize a Memory object to a JSON-safe dict."""
    return {
        "id": m.id,
        "content": m.content,
        "memory_type": m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type),
        "scope": m.scope.value if hasattr(m.scope, "value") else str(m.scope),
        "user_id": m.user_id,
        "agent_id": m.agent_id,
        "session_id": m.session_id,
        "tags": list(m.tags) if m.tags else [],
        "importance": m.importance,
        "is_active": m.is_active,
        "access_count": m.access_count,
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "updated_at": m.updated_at.isoformat() if m.updated_at else None,
        "last_accessed": m.last_accessed.isoformat() if m.last_accessed else None,
        "entity_ids": list(m.entity_ids) if m.entity_ids else [],
    }


def _compute_latency_stats(metrics_data: dict) -> dict:
    """Extract P99/P95/P50/mean from metrics histograms."""
    histograms = metrics_data.get("histograms", {})
    recall_hist = histograms.get("neuralmem.recall.duration", {})
    if not recall_hist:
        return {"p99": 0.0, "p95": 0.0, "p50": 0.0, "mean": 0.0}
    return {
        "p99": recall_hist.get("p99", 0.0) * 1000,
        "p95": recall_hist.get("p95", 0.0) * 1000,
        "p50": recall_hist.get("p50", 0.0) * 1000,
        "mean": recall_hist.get("mean", 0.0) * 1000,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy" if _mem else "no_engine", "version": "1.6.0"}


@app.get("/api/memories", response_model=MemoryListResponse)
async def list_memories(
    limit: int = Query(default=20, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    user_id: str | None = None,
    memory_type: str | None = None,
    active_only: bool = Query(default=True),
) -> dict:
    """Paginated memory list with optional filtering."""
    if _mem is None:
        raise HTTPException(status_code=503, detail="NeuralMem engine not initialized")

    all_memories = _mem.storage.list_memories(user_id=user_id, limit=limit + offset + 500)

    # Filter by type and active status
    filtered = []
    for m in all_memories:
        if active_only and not m.is_active:
            continue
        if memory_type and str(m.memory_type) != memory_type:
            continue
        filtered.append(m)

    total = len(filtered)
    page = filtered[offset : offset + limit]
    items = [_serialize_memory(m) for m in page]

    return {"memories": items, "total": total, "limit": limit, "offset": offset}


@app.get("/api/memories/{memory_id}")
async def get_memory(memory_id: str) -> dict:
    """Get a single memory by ID."""
    if _mem is None:
        raise HTTPException(status_code=503, detail="NeuralMem engine not initialized")

    memory = _mem.storage.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _serialize_memory(memory)


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> dict:
    """Semantic search via NeuralMem recall."""
    if _mem is None:
        raise HTTPException(status_code=503, detail="NeuralMem engine not initialized")

    from neuralmem.core.types import MemoryType

    memory_types = None
    if request.memory_types:
        memory_types = [MemoryType(mt) for mt in request.memory_types if mt in MemoryType._value2member_map_]

    results = _mem.recall(
        request.query,
        user_id=request.user_id,
        agent_id=request.agent_id,
        memory_types=memory_types,
        tags=request.tags,
        limit=request.limit,
        min_score=request.min_score,
    )

    items = []
    for r in results:
        items.append({
            "memory": _serialize_memory(r.memory),
            "score": r.score,
            "retrieval_method": r.retrieval_method,
            "explanation": r.explanation,
        })

    return {"results": items, "count": len(items), "query": request.query}


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats() -> dict:
    """Aggregated stats: storage, graph, and performance metrics."""
    if _mem is None:
        raise HTTPException(status_code=503, detail="NeuralMem engine not initialized")

    storage_stats = _mem.storage.get_stats()
    graph_stats = _mem.graph.get_stats()
    metrics_data = _mem.metrics.get_metrics()
    latency = _compute_latency_stats(metrics_data)

    counters = metrics_data.get("counters", {})
    all_memories = _mem.storage.list_memories(limit=100_000)
    active = sum(1 for m in all_memories if m.is_active)
    superseded = len(all_memories) - active

    # Compute cache hit rate if available
    cache_hits = counters.get("neuralmem.cache.hits", 0)
    cache_misses = counters.get("neuralmem.cache.misses", 0)
    total_cache = cache_hits + cache_misses
    hit_rate = cache_hits / total_cache if total_cache > 0 else 0.0

    return {
        "memory_count": storage_stats.get("total_memories", len(all_memories)),
        "node_count": graph_stats.get("node_count", 0),
        "edge_count": graph_stats.get("edge_count", 0),
        "p99_latency_ms": latency["p99"],
        "p95_latency_ms": latency["p95"],
        "p50_latency_ms": latency["p50"],
        "mean_latency_ms": latency["mean"],
        "recall_calls": counters.get("neuralmem.recall.calls", 0),
        "remember_calls": counters.get("neuralmem.remember.calls", 0),
        "cache_hit_rate": hit_rate,
        "active_memories": active,
        "superseded_memories": superseded,
    }


@app.get("/api/graph", response_model=GraphResponse)
async def get_graph(
    user_id: str | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict:
    """Export knowledge graph as nodes + edges for D3 visualization."""
    if _mem is None:
        raise HTTPException(status_code=503, detail="NeuralMem engine not initialized")

    entities = _mem.graph.get_entities(user_id=user_id)
    nodes = []
    for e in entities[:limit]:
        nodes.append({
            "id": e.id,
            "name": e.name,
            "type": e.entity_type,
            "aliases": list(e.aliases) if e.aliases else [],
            "attributes": e.attributes,
        })

    # Build edges from relations
    edges = []
    seen_edges = set()
    for e in entities[:limit]:
        neighbors = _mem.graph.get_neighbors([e.id], depth=1)
        for neighbor in neighbors:
            edge_key = tuple(sorted([e.id, neighbor.id]))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append({
                    "source": e.id,
                    "target": neighbor.id,
                    "relation_type": "related",
                    "weight": 1.0,
                })

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


# ---------------------------------------------------------------------------
# Startup helper
# ---------------------------------------------------------------------------

def set_engine(mem: NeuralMem) -> None:
    """Inject the NeuralMem engine into the FastAPI app."""
    global _mem
    _mem = mem
    _logger.info("NeuralMem engine injected into Dashboard API")


def get_app() -> FastAPI:
    """Return the configured FastAPI application."""
    return app
