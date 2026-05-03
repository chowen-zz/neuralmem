"""Tenant middleware for MCP server integration.

Extracts tenant_id from MCP tool call metadata, validates tenant
existence and limits, and injects tenant context into the request lifecycle.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from neuralmem.tenancy.manager import TenantManager

_logger = logging.getLogger(__name__)


class TenantMiddleware:
    """MCP middleware that resolves and validates tenant context.

    Parameters
    ----------
    manager : TenantManager
        The tenant manager used for validation and context management.

    Usage::

        middleware = TenantMiddleware(manager)

        # In an MCP tool handler:
        with middleware.process_request(metadata) as ctx:
            mem.remember("hello", user_id=ctx["tenant_id"])
    """

    def __init__(self, manager: TenantManager) -> None:
        self._manager = manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def process_request(self, metadata: dict[str, Any]):
        """Process an incoming MCP tool call request.

        Extracts ``tenant_id`` from the metadata dict, validates the
        tenant, checks rate limits, and sets the tenant context for the
        duration of the yield.

        Parameters
        ----------
        metadata : dict
            MCP tool call metadata.  Expected to contain a ``tenant_id``
            key (or ``X-Tenant-Id`` header-style key).

        Yields
        ------
        dict
            A context dict with ``tenant_id`` and ``namespace``.

        Raises
        ------
        ValueError
            If no tenant_id is found in metadata.
        KeyError
            If the tenant does not exist.
        PermissionError
            If the tenant is rate-limited.
        """
        tenant_id = self._extract_tenant_id(metadata)
        if not tenant_id:
            raise ValueError(
                "Missing tenant_id in MCP tool call metadata"
            )

        with self._manager.with_tenant(tenant_id) as cfg:
            yield {
                "tenant_id": tenant_id,
                "namespace": cfg.storage_namespace,
                "config": cfg,
            }

    def validate_request(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Validate a request without entering a context manager.

        Returns a dict with ``tenant_id``, ``namespace``, and ``config``.

        Raises the same exceptions as ``process_request``.
        """
        tenant_id = self._extract_tenant_id(metadata)
        if not tenant_id:
            raise ValueError(
                "Missing tenant_id in MCP tool call metadata"
            )

        cfg = self._manager.get_tenant(tenant_id)

        # Rate limit check
        allowed, info = self._manager.check_rate_limit(tenant_id)
        if not allowed:
            raise PermissionError(
                f"Rate limit exceeded for tenant '{tenant_id}'. "
                f"Retry after {info.get('retry_after', '?')}s"
            )

        return {
            "tenant_id": tenant_id,
            "namespace": cfg.storage_namespace,
            "config": cfg,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tenant_id(metadata: dict[str, Any]) -> str | None:
        """Extract tenant_id from MCP metadata.

        Supports several key variants:
        - ``tenant_id``
        - ``tenant-id``
        - ``X-Tenant-Id``
        - ``x_tenant_id``
        """
        for key in ("tenant_id", "tenant-id", "X-Tenant-Id", "x_tenant_id"):
            val = metadata.get(key)
            if val:
                return str(val).strip()
        return None
