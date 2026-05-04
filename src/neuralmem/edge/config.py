"""Edge configuration for Cloudflare Workers / serverless runtimes.

All values can be overridden via environment variables with NEURALMEM_EDGE_ prefix.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EdgeConfig:
    """Configuration tailored for edge / serverless environments.

    Fields are read from environment variables at instantiation time,
    falling back to sensible defaults for Workers KV deployments.
    """

    # KV binding names (used in Workers runtime)
    kv_memories: str = field(default="NEURALMEM_MEMORIES")
    kv_index: str = field(default="NEURALMEM_INDEX")
    kv_graph: str = field(default="NEURALMEM_GRAPH")
    kv_stats: str = field(default="NEURALMEM_STATS")
    kv_connectors: str = field(default="NEURALMEM_CONNECTORS")

    # Embedding (edge-friendly: small model or external API)
    embedding_provider: str = field(default="local")
    embedding_model: str = field(default="all-MiniLM-L6-v2")
    embedding_dimension: int = field(default=384)
    embedding_api_url: str | None = field(default=None)
    embedding_api_key: str | None = field(default=None)

    # Search / retrieval defaults
    default_search_limit: int = field(default=10)
    min_search_score: float = field(default=0.3)
    enable_reranker: bool = field(default=False)

    # Connector sync
    sync_interval_minutes: int = field(default=15)
    max_sync_items_per_run: int = field(default=100)
    connector_timeout_seconds: int = field(default=30)

    # Security
    api_key: str | None = field(default=None)
    require_auth: bool = field(default=False)
    cors_origins: tuple[str, ...] = field(default_factory=lambda: ("*",))

    # Observability
    enable_metrics: bool = field(default=False)
    log_level: str = field(default="INFO")

    @classmethod
    def from_env(cls, prefix: str = "NEURALMEM_EDGE_") -> "EdgeConfig":
        """Build config from environment variables.

        Example:
            NEURALMEM_EDGE_KV_MEMORIES=MY_KV_NAMESPACE
            NEURALMEM_EDGE_EMBEDDING_PROVIDER=openai
            NEURALMEM_EDGE_API_KEY=secret
        """
        def _get(key: str, default: Any) -> Any:
            val = os.getenv(f"{prefix}{key}")
            if val is None:
                return default
            # Coerce booleans
            if isinstance(default, bool):
                return val.lower() in ("1", "true", "yes", "on")
            if isinstance(default, int):
                return int(val)
            if isinstance(default, float):
                return float(val)
            if isinstance(default, tuple):
                return tuple(v.strip() for v in val.split(","))
            return val

        return cls(
            kv_memories=_get("KV_MEMORIES", "NEURALMEM_MEMORIES"),
            kv_index=_get("KV_INDEX", "NEURALMEM_INDEX"),
            kv_graph=_get("KV_GRAPH", "NEURALMEM_GRAPH"),
            kv_stats=_get("KV_STATS", "NEURALMEM_STATS"),
            kv_connectors=_get("KV_CONNECTORS", "NEURALMEM_CONNECTORS"),
            embedding_provider=_get("EMBEDDING_PROVIDER", "local"),
            embedding_model=_get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            embedding_dimension=_get("EMBEDDING_DIMENSION", 384),
            embedding_api_url=_get("EMBEDDING_API_URL", None),
            embedding_api_key=_get("EMBEDDING_API_KEY", None),
            default_search_limit=_get("DEFAULT_SEARCH_LIMIT", 10),
            min_search_score=_get("MIN_SEARCH_SCORE", 0.3),
            enable_reranker=_get("ENABLE_RERANKER", False),
            sync_interval_minutes=_get("SYNC_INTERVAL_MINUTES", 15),
            max_sync_items_per_run=_get("MAX_SYNC_ITEMS_PER_RUN", 100),
            connector_timeout_seconds=_get("CONNECTOR_TIMEOUT_SECONDS", 30),
            api_key=_get("API_KEY", None),
            require_auth=_get("REQUIRE_AUTH", False),
            cors_origins=_get("CORS_ORIGINS", ("*",)),
            enable_metrics=_get("ENABLE_METRICS", False),
            log_level=_get("LOG_LEVEL", "INFO"),
        )
