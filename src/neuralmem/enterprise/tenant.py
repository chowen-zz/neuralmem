"""Tenant manager — multi-tenant isolation with tenant-scoped in-memory storage.

No external database dependency; all state is held in memory with optional
namespace-based scoping for storage backends.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TenantConfig:
    """Configuration for a single tenant."""

    tenant_id: str
    name: str = ""
    storage_namespace: str = ""
    max_memories: int = 10_000
    max_queries_per_minute: int = 1_000
    allowed_memory_types: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.storage_namespace:
            object.__setattr__(
                self, "storage_namespace", f"tenant:{self.tenant_id}:"
            )


class TenantContext:
    """Thread-local tenant context holder.

    Use ``TenantManager.with_tenant`` to set the active tenant rather than
    manipulating this class directly.
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
    """Manage tenants with per-tenant isolation and memory limits.

    All state is kept in memory (no external DB).  A ``storage_backend``
    callable may be provided for namespace-scoped storage operations.
    """

    def __init__(
        self,
        storage_backend: Any | None = None,
    ) -> None:
        self._storage = storage_backend
        self._tenants: dict[str, TenantConfig] = {}
        self._tenant_data: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_tenant(self, config: TenantConfig) -> TenantConfig:
        """Register a new tenant.

        Raises ``ValueError`` if the tenant already exists.
        """
        with self._lock:
            if config.tenant_id in self._tenants:
                raise ValueError(f"Tenant '{config.tenant_id}' already exists")
            self._tenants[config.tenant_id] = config
            self._tenant_data[config.tenant_id] = {}
            return config

    def delete_tenant(self, tenant_id: str) -> None:
        """Delete a tenant and all its associated data.

        Raises ``KeyError`` if the tenant does not exist.
        """
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            del self._tenants[tenant_id]
            del self._tenant_data[tenant_id]

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

    def tenant_exists(self, tenant_id: str) -> bool:
        """Return ``True`` if the tenant is registered."""
        with self._lock:
            return tenant_id in self._tenants

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def with_tenant(self, tenant_id: str):
        """Context manager that sets the active tenant for the current thread.

        Validates tenant exists.  Restores previous context on exit.

        Usage::

            with manager.with_tenant("acme"):
                mem.remember("important fact")
        """
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            cfg = self._tenants[tenant_id]

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
    # Tenant-scoped storage (in-memory)
    # ------------------------------------------------------------------

    def put(self, tenant_id: str, key: str, value: Any) -> None:
        """Store a key/value pair scoped to a tenant."""
        with self._lock:
            if tenant_id not in self._tenant_data:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            self._tenant_data[tenant_id][key] = value

    def get(self, tenant_id: str, key: str, default: Any = None) -> Any:
        """Retrieve a value by key scoped to a tenant."""
        with self._lock:
            if tenant_id not in self._tenant_data:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            return self._tenant_data[tenant_id].get(key, default)

    def delete(self, tenant_id: str, key: str) -> bool:
        """Delete a key from tenant-scoped storage.  Returns ``True`` if deleted."""
        with self._lock:
            if tenant_id not in self._tenant_data:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            if key in self._tenant_data[tenant_id]:
                del self._tenant_data[tenant_id][key]
                return True
            return False

    def list_keys(self, tenant_id: str) -> list[str]:
        """Return all keys stored for a tenant."""
        with self._lock:
            if tenant_id not in self._tenant_data:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            return list(self._tenant_data[tenant_id].keys())

    def clear_tenant_data(self, tenant_id: str) -> None:
        """Clear all in-memory data for a tenant without deleting the tenant."""
        with self._lock:
            if tenant_id not in self._tenant_data:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            self._tenant_data[tenant_id].clear()

    # ------------------------------------------------------------------
    # Memory limits
    # ------------------------------------------------------------------

    def check_memory_limit(self, tenant_id: str) -> bool:
        """Return ``True`` if the tenant is under its memory limit."""
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            cfg = self._tenants[tenant_id]
            count = len(self._tenant_data.get(tenant_id, {}))
            return count < cfg.max_memories

    def get_memory_count(self, tenant_id: str) -> int:
        """Return the number of items stored for a tenant."""
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            return len(self._tenant_data.get(tenant_id, {}))

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
    # Namespace helpers
    # ------------------------------------------------------------------

    def get_namespace(self, tenant_id: str) -> str:
        """Return the storage namespace for a tenant."""
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Tenant '{tenant_id}' not found")
            return self._tenants[tenant_id].storage_namespace

    def scoped_key(self, tenant_id: str, key: str) -> str:
        """Prefix a key with the tenant's storage namespace."""
        ns = self.get_namespace(tenant_id)
        return f"{ns}{key}"
