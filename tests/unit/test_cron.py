"""Unit tests for CronScheduler connector sync jobs.

All tests use mock connectors; no external services required.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from neuralmem.connectors.base import ConnectorState, SyncItem
from neuralmem.core.types import Memory, MemoryType
from neuralmem.edge.config import EdgeConfig
from neuralmem.edge.cron import CronScheduler
from neuralmem.edge.storage import EdgeStorage


@pytest.fixture
def config():
    return EdgeConfig(
        max_sync_items_per_run=10,
        sync_interval_minutes=5,
    )


@pytest.fixture
def storage(config):
    return EdgeStorage(config=config, kv={})


@pytest.fixture
def scheduler(storage, config):
    return CronScheduler(storage=storage, config=config)


@pytest.fixture
def mock_connector():
    conn = MagicMock()
    conn.name = "mock_connector"
    conn.is_available = MagicMock(return_value=True)
    conn.authenticate = MagicMock(return_value=True)
    conn.sync = MagicMock(return_value=[
        SyncItem(
            id="s1",
            content="synced item 1",
            source="mock",
            title="Title 1",
            author="Alice",
            tags=["docs"],
        ),
        SyncItem(
            id="s2",
            content="synced item 2",
            source="mock",
            title="Title 2",
            author="Bob",
            tags=["code"],
        ),
    ])
    conn.disconnect = MagicMock()
    return conn


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

def test_register_connector(scheduler, mock_connector):
    scheduler.register_connector("mock", mock_connector)
    assert "mock" in scheduler._connectors
    assert scheduler._connectors["mock"] is mock_connector


def test_unregister_connector(scheduler, mock_connector):
    scheduler.register_connector("mock", mock_connector)
    scheduler.unregister_connector("mock")
    assert "mock" not in scheduler._connectors


# --------------------------------------------------------------------------- #
# Sync execution
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_run_sync_success(scheduler, mock_connector, storage):
    scheduler.register_connector("mock", mock_connector)
    result = await scheduler.run_sync("mock")
    assert result["status"] == "ok"
    assert result["synced"] == 2
    assert result["errors"] == 0
    assert len(result["items"]) == 2
    # Verify memories stored
    memories = storage.list_memories(limit=10)
    assert len(memories) == 2


@pytest.mark.asyncio
async def test_run_sync_not_registered(scheduler):
    result = await scheduler.run_sync("missing")
    assert result["status"] == "skipped"
    assert result["reason"] == "not_registered"
    assert result["synced"] == 0


@pytest.mark.asyncio
async def test_run_sync_not_available(scheduler, mock_connector):
    mock_connector.is_available = MagicMock(return_value=False)
    scheduler.register_connector("mock", mock_connector)
    result = await scheduler.run_sync("mock")
    assert result["status"] == "skipped"
    assert result["reason"] == "not_available"


@pytest.mark.asyncio
async def test_run_sync_connector_error(scheduler, mock_connector):
    mock_connector.sync = MagicMock(side_effect=RuntimeError("boom"))
    scheduler.register_connector("mock", mock_connector)
    result = await scheduler.run_sync("mock")
    assert result["status"] == "error"
    assert "boom" in result["error"]
    assert result["errors"] == 1
    # disconnect should still be called
    mock_connector.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_run_sync_all_connectors(scheduler, mock_connector):
    scheduler.register_connector("c1", mock_connector)
    scheduler.register_connector("c2", mock_connector)
    results = await scheduler.run_all_syncs()
    assert "c1" in results
    assert "c2" in results
    assert results["c1"]["status"] == "ok"
    assert results["c2"]["status"] == "ok"


@pytest.mark.asyncio
async def test_run_sync_none_runs_all(scheduler, mock_connector):
    scheduler.register_connector("c1", mock_connector)
    result = await scheduler.run_sync(None)
    assert "c1" in result


@pytest.mark.asyncio
async def test_run_sync_respects_max_items(scheduler, mock_connector, config):
    config.max_sync_items_per_run = 1
    scheduler.config = config
    scheduler.register_connector("mock", mock_connector)
    result = await scheduler.run_sync("mock")
    assert result["synced"] == 1


@pytest.mark.asyncio
async def test_run_sync_store_failure(scheduler, mock_connector, storage):
    """Simulate a failure during memory storage."""
    original_save = storage.save_memory
    call_count = 0
    def flaky_save(mem):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("store error")
        return original_save(mem)
    storage.save_memory = flaky_save
    scheduler.register_connector("mock", mock_connector)
    result = await scheduler.run_sync("mock")
    assert result["synced"] == 1
    assert result["errors"] == 1


# --------------------------------------------------------------------------- #
# State persistence
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_last_sync_timestamp(scheduler, mock_connector, storage):
    scheduler.register_connector("mock", mock_connector)
    await scheduler.run_sync("mock")
    ts = scheduler._get_last_sync("mock")
    assert ts is not None
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None


def test_get_last_sync_missing(scheduler):
    assert scheduler._get_last_sync("never") is None


# --------------------------------------------------------------------------- #
# Item dict conversion
# --------------------------------------------------------------------------- #

def test_item_to_dict(scheduler):
    item = SyncItem(
        id="x",
        content="c",
        source="s",
        title="T",
        author="A",
        tags=["t1"],
    )
    d = scheduler._item_to_dict(item)
    assert d["id"] == "x"
    assert d["source"] == "s"
    assert d["title"] == "T"
    assert d["author"] == "A"
    assert d["tags"] == ["t1"]
    assert "content" not in d  # content is not included in dict output


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_run_sync_empty_items(scheduler, mock_connector):
    mock_connector.sync = MagicMock(return_value=[])
    scheduler.register_connector("mock", mock_connector)
    result = await scheduler.run_sync("mock")
    assert result["status"] == "ok"
    assert result["synced"] == 0
    assert result["errors"] == 0


@pytest.mark.asyncio
async def test_run_sync_disconnect_raises(scheduler, mock_connector):
    mock_connector.disconnect = MagicMock(side_effect=RuntimeError("disconnect fail"))
    scheduler.register_connector("mock", mock_connector)
    result = await scheduler.run_sync("mock")
    # Should still succeed even if disconnect raises
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_run_sync_no_authenticate(scheduler):
    """Connector without authenticate method."""
    conn = MagicMock()
    conn.sync = MagicMock(return_value=[
        SyncItem(id="s1", content="c", source="bare"),
    ])
    scheduler.register_connector("bare", conn)
    result = await scheduler.run_sync("bare")
    assert result["status"] == "ok"
    assert result["synced"] == 1
