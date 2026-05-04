"""Cron scheduler for connector sync jobs on edge runtimes.

Runs connector sync at configurable intervals, persisting state in KV.
Compatible with Cloudflare Workers Cron Triggers and local test mocks.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from neuralmem.connectors.base import ConnectorProtocol, SyncItem
from neuralmem.edge.config import EdgeConfig
from neuralmem.edge.storage import EdgeStorage

_logger = logging.getLogger(__name__)


class CronScheduler:
    """Scheduler that drives connector sync jobs.

    In production, Workers Cron Triggers call ``run_all_syncs()``.
    In tests, call ``run_sync(connector_name)`` with a mock connector.
    """

    def __init__(
        self,
        storage: EdgeStorage,
        config: EdgeConfig | None = None,
    ) -> None:
        self.storage = storage
        self.config = config or EdgeConfig.from_env()
        self._connectors: dict[str, ConnectorProtocol] = {}

    # --- connector registry ---------------------------------------------------

    def register_connector(self, name: str, connector: ConnectorProtocol) -> None:
        """Register a connector instance for scheduled sync."""
        self._connectors[name] = connector
        _logger.info("Registered connector '%s' for cron sync", name)

    def unregister_connector(self, name: str) -> None:
        """Remove a connector from the scheduler."""
        self._connectors.pop(name, None)

    # --- sync execution ------------------------------------------------------

    async def run_all_syncs(self) -> dict[str, Any]:
        """Run sync for all registered connectors. Called by Cron Trigger."""
        results: dict[str, Any] = {}
        for name in self._connectors:
            results[name] = await self.run_sync(name)
        return results

    async def run_sync(self, connector_name: str | None = None) -> dict[str, Any]:
        """Run sync for a single connector (or all if name is None).

        Returns a result dict with counts and errors.
        """
        if connector_name is None:
            return await self.run_all_syncs()

        connector = self._connectors.get(connector_name)
        if connector is None:
            return {
                "connector": connector_name,
                "status": "skipped",
                "reason": "not_registered",
                "synced": 0,
                "errors": 0,
            }

        # Check if connector is available
        if hasattr(connector, "is_available") and not connector.is_available():
            return {
                "connector": connector_name,
                "status": "skipped",
                "reason": "not_available",
                "synced": 0,
                "errors": 0,
            }

        last_sync = self._get_last_sync(connector_name)
        _logger.info("Starting sync for '%s' (last sync: %s)", connector_name, last_sync)

        synced = 0
        errors = 0
        items: list[SyncItem] = []

        try:
            if hasattr(connector, "authenticate"):
                connector.authenticate()
            if hasattr(connector, "sync"):
                items = connector.sync(limit=self.config.max_sync_items_per_run)
        except Exception as exc:
            _logger.exception("Sync failed for connector '%s'", connector_name)
            errors += 1
            return {
                "connector": connector_name,
                "status": "error",
                "error": str(exc),
                "synced": synced,
                "errors": errors,
            }
        finally:
            if hasattr(connector, "disconnect"):
                try:
                    connector.disconnect()
                except Exception:
                    pass

        # Store each SyncItem as a Memory
        for item in items[: self.config.max_sync_items_per_run]:
            try:
                from neuralmem.core.types import Memory, MemoryType

                memory = Memory(
                    content=item.content,
                    memory_type=MemoryType.EPISODIC,
                    user_id=None,
                    tags=tuple(item.tags),
                    source=item.source,
                )
                self.storage.save_memory(memory)
                synced += 1
            except Exception:
                _logger.exception("Failed to store sync item from '%s'", connector_name)
                errors += 1

        self._set_last_sync(connector_name)
        return {
            "connector": connector_name,
            "status": "ok",
            "synced": synced,
            "errors": errors,
            "items": [self._item_to_dict(i) for i in items],
        }

    # --- state persistence ----------------------------------------------------

    def _get_last_sync(self, connector_name: str) -> str | None:
        """Read last sync timestamp from KV."""
        key = f"cron:last_sync:{connector_name}"
        raw = self.storage._kv_get(key)
        if isinstance(raw, str):
            return raw
        return None

    def _set_last_sync(self, connector_name: str) -> None:
        """Write last sync timestamp to KV."""
        key = f"cron:last_sync:{connector_name}"
        self.storage._kv_put(key, datetime.now(timezone.utc).isoformat())

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _item_to_dict(item: SyncItem) -> dict[str, Any]:
        return {
            "id": item.id,
            "source": item.source,
            "title": item.title,
            "author": item.author,
            "tags": item.tags,
        }
