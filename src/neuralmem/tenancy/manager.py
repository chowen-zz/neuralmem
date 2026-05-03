"""Tenant manager — CRUD, isolation, rate limiting, memory limits."""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

from neuralmem.auth.ratelimit import RateLimitConfig, RateLimiter
from neuralmem.tenancy.models import TenantConfig

if TYPE_CHECKING:
    from neuralmem.core.memory import NeuralMem

_logger = logging.getLogger(__name__)


class TenantContext:
    """Thread-local tenant context holder.

    Use ``TenantManager.with_tenant`` to set the active tenant rather
    than manipulating this class directly.
    """

    _local = threading.local()

    @classmethod
    def get_tenant_id(cls) -> str | None:
        """Return the currently active tenant id (or ``None``)."""
        return getattr(cls._local, "tenant_id", None)

    @classmethod
    def set_tenant_id(cls, tenant_id: str | None) -> None:
        cls._local.tenant_id = tenant_id

    @classmethod
    def get_namespace(cls) -> str:
        """Return the storage namespace prefix for the active tenant."""
        return getattr(cls._local, "namespace", "")

    @classmethod
    def set_namespace(cls, namespace: str) -> None:
        cls._local.namespace = namespace

    @classmethod
    def clear(cls) -> None:
        cls._local.tenant_id = None
        cls._local.namespace = ""


class TenantManager:
    """Manage tenants with per-tenant rate limiting and memory limits.

    Parameters
    ----------
    neuralmem : NeuralMem | None
        Optional NeuralMem instance whose storage is used for tenant
        memory counting and deletion.
    """

    def __init__(self, neuralmem: NeuralMem | None = None) -> None:
        self._neuralmem = neuralmem
        self._tenants: dict[str, TenantConfig] = {}
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_tenant(self, config: TenantConfig) -> TenantConfig:
        """Register a new tenant.

        Raises ``ValueError`` if the tenant already exists.
        """
        with self._lock:
            if config.tenant_id in self._tenants:
                raise ValueError(
                    f"Tenant '{config.tenant_id}' already exists"
                )
            # Auto-generate storage_namespace if empty
            cfg = config
            if not cfg.storage_namespace:
                cfg = config.model_copy(
                    update={"storage_namespace": f"tenant:{config.tenant_id}:"}
                )
            self._tenants[config.tenant_id] = cfg
            self._rate_limiters[config.tenant_id] = RateLimiter(
                RateLimitConfig(
                    requests_per_minute=cfg.max_queries_per_minute,
                    burst_size=min(cfg.max_queries_per_minute, 10),
                )
            )
            _logger.info("Created tenant: %s", config.tenant_id)
            return cfg

    def delete_tenant(self, tenant_id: str) -> None:
        """Delete a tenant and all its associated memories.

        Raises ``KeyError`` if the tenant does not exist.
        """
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Tenant '{tenant_id}' not found")

        # Delete memories outside lock (may take time)
        if self._neuralmem is not None:
            ns = self._tenants[tenant_id].storage_namespace
            self._delete_tenant_memories(tenant_id, ns)

        with self._lock:
            del self._tenants[tenant_id]
            self._rate_limiters.pop(tenant_id, None)
            _logger.info("Deleted tenant: %s", tenant_id)

    def get_tenant(self, tenant_id: str) -> TenantConfig:
        """Return the configuration for a tenant.

        Raises ``KeyError`` if the tenant does not exist.
        """
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            return self._tenants[tenant_id]

    def list_tenants(self) -> list[TenantConfig]:
        """Return all registered tenant configurations."""
        with self._lock:
            return list(self._tenants.values())

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def with_tenant(self, tenant_id: str):
        """Context manager that sets the active tenant for the current thread.

        Validates tenant exists and checks rate limit before entering.
        Restores previous context on exit.

        Usage::

            with manager.with_tenant("acme"):
                mem.remember("important fact")
        """
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            cfg = self._tenants[tenant_id]
            limiter = self._rate_limiters[tenant_id]

        # Rate limit check
        allowed, info = limiter.consume(tenant_id)
        if not allowed:
            raise PermissionError(
                f"Rate limit exceeded for tenant '{tenant_id}'. "
                f"Retry after {info.get('retry_after', '?')}s"
            )

        # Save previous context
        prev_tenant = TenantContext.get_tenant_id()
        prev_ns = TenantContext.get_namespace()
        try:
            TenantContext.set_tenant_id(tenant_id)
            TenantContext.set_namespace(cfg.storage_namespace)
            yield cfg
        finally:
            TenantContext.set_tenant_id(prev_tenant)
            TenantContext.set_namespace(prev_ns)

    # ------------------------------------------------------------------
    # Rate limiting helpers
    # ------------------------------------------------------------------

    def check_rate_limit(self, tenant_id: str) -> tuple[bool, dict]:
        """Check whether a request is allowed for a tenant without consuming.

        Returns ``(allowed, info)``.
        """
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            limiter = self._rate_limiters[tenant_id]
        return limiter.check(tenant_id)

    # ------------------------------------------------------------------
    # Memory limits
    # ------------------------------------------------------------------

    def check_memory_limit(self, tenant_id: str) -> bool:
        """Return ``True`` if the tenant is under its memory limit.

        Requires a NeuralMem instance to count stored memories.
        """
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            cfg = self._tenants[tenant_id]

        if self._neuralmem is None:
            return True

        count = self._count_tenant_memories(tenant_id, cfg.storage_namespace)
        return count < cfg.max_memories

    def get_memory_count(self, tenant_id: str) -> int:
        """Return the number of memories stored for a tenant."""
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            cfg = self._tenants[tenant_id]

        if self._neuralmem is None:
            return 0
        return self._count_tenant_memories(tenant_id, cfg.storage_namespace)

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def validate_memory_type(self, tenant_id: str, memory_type: str) -> bool:
        """Return ``True`` if the memory type is allowed for the tenant.

        An empty ``allowed_memory_types`` means all types are allowed.
        """
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            cfg = self._tenants[tenant_id]

        if not cfg.allowed_memory_types:
            return True
        return memory_type in cfg.allowed_memory_types

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_tenant_memories(
        self, tenant_id: str, namespace: str
    ) -> int:
        """Count memories belonging to a tenant."""
        if self._neuralmem is None:
            return 0
        try:
            stats = self._neuralmem.storage.get_stats(user_id=tenant_id)
            return int(stats.get("total_memories", 0))
        except Exception:
            return 0

    def _delete_tenant_memories(
        self, tenant_id: str, namespace: str
    ) -> None:
        """Delete all memories belonging to a tenant."""
        if self._neuralmem is None:
            return
        try:
            self._neuralmem.storage.delete_memories(user_id=tenant_id)
        except Exception as exc:
            _logger.warning(
                "Failed to delete memories for tenant %s: %s",
                tenant_id,
                exc,
            )
