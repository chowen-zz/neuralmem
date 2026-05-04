"""HTTP route handler for NeuralMem edge API.

Exposes REST endpoints:
  POST /memories        → remember
  GET  /memories        → list memories
  GET  /memories/{id}   → get memory
  DELETE /memories/{id} → delete memory
  POST /search          → recall (vector + keyword + temporal)
  GET  /stats           → storage stats
  POST /sync            → trigger connector sync

All methods are async-ready and mock-testable.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from neuralmem.core.types import Memory, MemoryType, SearchQuery, SearchResult
from neuralmem.edge.config import EdgeConfig
from neuralmem.edge.storage import EdgeStorage

_logger = logging.getLogger(__name__)


class HTTPRouteHandler:
    """Route incoming HTTP requests to NeuralMem operations.

    In Workers runtime ``request`` is a real Request object.
    In tests it can be a plain dict with ``method``, ``url``, and ``body``.
    """

    def __init__(
        self,
        storage: EdgeStorage,
        config: EdgeConfig | None = None,
    ) -> None:
        self.storage = storage
        self.config = config or EdgeConfig.from_env()

    # --- public API ---------------------------------------------------------

    async def handle(self, request: Any) -> Any:
        """Dispatch request to the appropriate handler."""
        method = _get_method(request)
        url = _get_url(request)
        path = _extract_path(url)

        # CORS preflight
        if method == "OPTIONS":
            return _cors_response(self.config.cors_origins)

        # Auth check
        if self.config.require_auth:
            auth_result = self._check_auth(request)
            if auth_result is not None:
                return auth_result

        # Route dispatch
        if path == "/memories" or path == "/memories/":
            if method == "POST":
                return await self._handle_remember(request)
            if method == "GET":
                return await self._handle_list_memories(request)
            return _method_not_allowed()

        if path.startswith("/memories/"):
            mem_id = path[len("/memories/"):]
            if method == "GET":
                return await self._handle_get_memory(mem_id)
            if method == "DELETE":
                return await self._handle_delete_memory(mem_id)
            return _method_not_allowed()

        if path == "/search" or path == "/search/":
            if method == "POST":
                return await self._handle_search(request)
            return _method_not_allowed()

        if path == "/stats" or path == "/stats/":
            if method == "GET":
                return await self._handle_stats(request)
            return _method_not_allowed()

        if path == "/sync" or path == "/sync/":
            if method == "POST":
                return await self._handle_sync(request)
            return _method_not_allowed()

        return _not_found()

    # --- route handlers -----------------------------------------------------

    async def _handle_remember(self, request: Any) -> Any:
        """POST /memories — store new memories."""
        body = await _read_json_body(request)
        if body is None:
            return _bad_request("JSON body required")

        content = body.get("content")
        if not content or not str(content).strip():
            return _bad_request("'content' is required")

        mem_type_str = body.get("memory_type", "semantic")
        try:
            mem_type = MemoryType(mem_type_str)
        except ValueError:
            return _bad_request(f"Invalid memory_type: {mem_type_str}")

        memory = Memory(
            content=str(content),
            memory_type=mem_type,
            user_id=body.get("user_id"),
            agent_id=body.get("agent_id"),
            session_id=body.get("session_id"),
            tags=tuple(body.get("tags", [])),
            source=body.get("source"),
            importance=float(body.get("importance", 0.5)),
        )
        memory_id = self.storage.save_memory(memory)
        return _json_response({"id": memory_id, "status": "created"}, status=201)

    async def _handle_list_memories(self, request: Any) -> Any:
        """GET /memories — list memories with optional filters."""
        query_params = _get_query_params(request)
        user_id = query_params.get("user_id")
        limit = int(query_params.get("limit", self.config.default_search_limit))
        memories = self.storage.list_memories(user_id=user_id, limit=limit)
        return _json_response(
            {
                "memories": [m.model_dump(mode="json") for m in memories],
                "count": len(memories),
            }
        )

    async def _handle_get_memory(self, memory_id: str) -> Any:
        """GET /memories/{id} — retrieve a single memory."""
        mem = self.storage.get_memory(memory_id)
        if mem is None:
            return _not_found(f"Memory {memory_id} not found")
        return _json_response({"memory": mem.model_dump(mode="json")})

    async def _handle_delete_memory(self, memory_id: str) -> Any:
        """DELETE /memories/{id} — delete a memory."""
        deleted = self.storage.delete_memories(memory_id=memory_id)
        if deleted == 0:
            return _not_found(f"Memory {memory_id} not found")
        return _json_response({"deleted": deleted})

    async def _handle_search(self, request: Any) -> Any:
        """POST /search — vector / keyword / temporal search."""
        body = await _read_json_body(request)
        if body is None:
            return _bad_request("JSON body required")

        query = body.get("query")
        if not query or not str(query).strip():
            return _bad_request("'query' is required")

        user_id = body.get("user_id")
        limit = int(body.get("limit", self.config.default_search_limit))
        min_score = float(body.get("min_score", self.config.min_search_score))
        search_type = body.get("type", "keyword")  # keyword | vector | temporal

        # For keyword search (no embedding required on edge)
        if search_type == "keyword":
            results = self.storage.keyword_search(
                str(query), user_id=user_id, limit=limit
            )
        elif search_type == "vector":
            # Expect pre-computed embedding in request (edge has no local embedder)
            vector = body.get("embedding")
            if not vector or not isinstance(vector, list):
                return _bad_request("'embedding' list required for vector search")
            results = self.storage.vector_search(
                vector, user_id=user_id, limit=limit
            )
        elif search_type == "temporal":
            vector = body.get("embedding")
            if not vector or not isinstance(vector, list):
                return _bad_request("'embedding' list required for temporal search")
            recency_weight = float(body.get("recency_weight", 0.3))
            results = self.storage.temporal_search(
                vector, user_id=user_id, recency_weight=recency_weight, limit=limit
            )
        else:
            return _bad_request(f"Invalid search type: {search_type}")

        # Hydrate results
        output: list[dict[str, Any]] = []
        for mem_id, score in results:
            if score < min_score:
                continue
            mem = self.storage.get_memory(mem_id)
            if mem is not None:
                output.append(
                    {
                        "memory": mem.model_dump(mode="json"),
                        "score": round(score, 4),
                    }
                )

        return _json_response({"results": output, "count": len(output)})

    async def _handle_stats(self, request: Any) -> Any:
        """GET /stats — storage and access statistics."""
        query_params = _get_query_params(request)
        user_id = query_params.get("user_id")
        stats = self.storage.get_stats(user_id=user_id)
        return _json_response({"stats": stats})

    async def _handle_sync(self, request: Any) -> Any:
        """POST /sync — trigger connector sync (delegates to CronScheduler)."""
        from neuralmem.edge.cron import CronScheduler

        body = await _read_json_body(request)
        connector_name = body.get("connector") if body else None
        scheduler = CronScheduler(storage=self.storage, config=self.config)
        result = await scheduler.run_sync(connector_name=connector_name)
        return _json_response(result)

    # --- auth ---------------------------------------------------------------

    def _check_auth(self, request: Any) -> Any | None:
        """Validate API key from Authorization header. Returns error response or None."""
        headers = _get_headers(request)
        auth = headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return _json_response({"error": "Unauthorized"}, status=401)
        token = auth[7:]
        if self.config.api_key is not None and token != self.config.api_key:
            return _json_response({"error": "Invalid API key"}, status=403)
        return None


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------


def _get_method(request: Any) -> str:
    if hasattr(request, "method"):
        return str(request.method).upper()
    return str(request.get("method", "GET")).upper()


def _get_url(request: Any) -> str:
    if hasattr(request, "url"):
        return str(request.url)
    return str(request.get("url", ""))


def _get_headers(request: Any) -> dict[str, str]:
    if hasattr(request, "headers"):
        h = request.headers
        if hasattr(h, "get"):
            return dict(h)  # type: ignore[arg-type]
    return dict(request.get("headers", {}))


def _extract_path(url: str) -> str:
    """Extract path from URL, stripping query string."""
    if "?" in url:
        url = url.split("?")[0]
    # Strip scheme/host if present
    if "://" in url:
        url = "/" + url.split("/", 3)[3] if len(url.split("/", 3)) > 3 else "/"
    return url


def _get_query_params(request: Any) -> dict[str, str]:
    """Parse query string into dict."""
    url = _get_url(request)
    if "?" not in url:
        return {}
    qs = url.split("?", 1)[1]
    params: dict[str, str] = {}
    for part in qs.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            params[k] = v
    return params


async def _read_json_body(request: Any) -> dict[str, Any] | None:
    """Read and parse JSON body from request."""
    try:
        if hasattr(request, "json"):
            # Async method on Workers Request
            body = await request.json()
            if isinstance(body, dict):
                return body
        body = request.get("body")
        if isinstance(body, str):
            return json.loads(body)
        if isinstance(body, dict):
            return body
    except Exception:
        pass
    return None


def _json_response(body: dict[str, Any], status: int = 200) -> dict[str, Any]:
    return {
        "status": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }


def _cors_response(origins: tuple[str, ...]) -> dict[str, Any]:
    origin = origins[0] if origins else "*"
    return {
        "status": 204,
        "headers": {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
        "body": "",
    }


def _bad_request(message: str) -> dict[str, Any]:
    return _json_response({"error": "Bad request", "detail": message}, status=400)


def _not_found(message: str = "Not found") -> dict[str, Any]:
    return _json_response({"error": "Not found", "detail": message}, status=404)


def _method_not_allowed() -> dict[str, Any]:
    return _json_response({"error": "Method not allowed"}, status=405)
