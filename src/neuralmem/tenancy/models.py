"""Tenant configuration models."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TenantConfig(BaseModel):
    """Configuration for a single tenant.

    Attributes
    ----------
    tenant_id : str
        Unique identifier for the tenant.
    max_memories : int
        Maximum number of memories this tenant may store.
    max_queries_per_minute : int
        Rate limit for queries per minute.
    allowed_memory_types : list[str]
        Memory types the tenant is allowed to use (empty = all).
    storage_namespace : str
        Prefix for all DB keys belonging to this tenant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    max_memories: int = Field(default=10_000, ge=1)
    max_queries_per_minute: int = Field(default=60, ge=1)
    allowed_memory_types: tuple[str, ...] = Field(default_factory=tuple)
    storage_namespace: str = ""
