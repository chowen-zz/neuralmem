"""Cloudflare Workers fetch adapter for NeuralMem.

Wraps the core NeuralMem engine so it can respond to Workers ``fetch`` events.
All heavy dependencies are lazy-imported to keep cold-start times minimal.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from neuralmem.edge.config import EdgeConfig
from neuralmem.edge.handler import HTTPRouteHandler
from neuralmem.edge.storage import EdgeStorage

_logger = logging.getLogger(__name__)


class CloudflareWorkersAdapter:
    """Adapter that exposes NeuralMem via Cloudflare Workers ``fetch`` handler.

    Usage in a Workers script::

        from neuralmem.edge import CloudflareWorkersAdapter, EdgeConfig

        config = EdgeConfig.from_env()
        adapter = CloudflareWorkersAdapter(config)

        async def on_fetch(request, env, ctx):
            return await adapter.fetch(request, env, ctx)

    In tests, pass a mock ``env`` dict with KV bindings and a mock ``request``.
    """

    def __init__(self, config: EdgeConfig | None = None) -> None:
        self.config = config or EdgeConfig.from_env()
        self._handler: HTTPRouteHandler | None = None
        self._storage: EdgeStorage | None = None

    # --- lazy initialisation ------------------------------------------------

    def _get_storage(self, env: dict[str, Any]) -> EdgeStorage:
        """Resolve KV bindings from env and return EdgeStorage."""
        if self._storage is None:
            # Workers KV namespace is passed via ``env`` binding
            kv = env.get(self.config.kv_memories, {})
            self._storage = EdgeStorage(config=self.config, kv=kv)
        return self._storage

    def _get_handler(self, env: dict[str, Any]) -> HTTPRouteHandler:
        """Lazy-build the HTTP route handler."""
        if self._handler is None:
            storage = self._get_storage(env)
            self._handler = HTTPRouteHandler(storage=storage, config=self.config)
        return self._handler

    # --- fetch entrypoint ---------------------------------------------------

    async def fetch(
        self,
        request: Any,
        env: dict[str, Any],
        ctx: Any | None = None,
    ) -> Any:
        """Main Workers fetch handler.

        Args:
            request: A Workers Request-like object (has ``method``, ``url``, ``json()``).
            env: Workers environment bindings (KV namespaces, secrets, etc.).
            ctx: Optional Workers execution context (waitUntil, passThroughOnException).

        Returns:
            A Response-like object (dict with ``status``, ``body``, ``headers`` in tests).
        """
        try:
            handler = self._get_handler(env)
            return await handler.handle(request)
        except Exception as exc:  # pragma: no cover
            _logger.exception("Unhandled error in fetch handler")
            return _json_response(
                {"error": "Internal server error", "detail": str(exc)},
                status=500,
            )

    # --- convenience methods for non-Workers use ----------------------------

    def remember(self, content: str, env: dict[str, Any], **kwargs: Any) -> list[Any]:
        """Synchronous convenience wrapper for remember (testing / local)."""
        from neuralmem.core.memory import NeuralMem

        storage = self._get_storage(env)
        mem = NeuralMem(config=None, embedder=None)
        mem.storage = storage  # type: ignore[assignment]
        return mem.remember(content, **kwargs)

    def recall(self, query: str, env: dict[str, Any], **kwargs: Any) -> list[Any]:
        """Synchronous convenience wrapper for recall (testing / local)."""
        from neuralmem.core.memory import NeuralMem

        storage = self._get_storage(env)
        mem = NeuralMem(config=None, embedder=None)
        mem.storage = storage  # type: ignore[assignment]
        return mem.recall(query, **kwargs)


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _json_response(body: dict[str, Any], status: int = 200) -> dict[str, Any]:
    """Build a JSON response dict (compatible with Workers Response and tests)."""
    return {
        "status": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
