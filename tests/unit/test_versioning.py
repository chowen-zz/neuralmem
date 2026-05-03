"""Tests for MemoryVersioner — version tracking and rollback."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neuralmem.versioning import MemoryVersioner

# ── Helpers ─────────────────────────────────────────────────────────────


def _make_mock_storage(
    history: list[dict] | None = None,
) -> MagicMock:
    """Create a mock storage backend with configurable history."""
    storage = MagicMock()
    storage.get_history.return_value = history or []
    return storage


def _history_entry(
    idx: int = 1,
    memory_id: str = "mem-1",
    old_content: str | None = None,
    new_content: str = "content",
    event: str = "UPDATE",
    changed_at: str = "2025-01-01T00:00:00+00:00",
) -> dict:
    return {
        "id": idx,
        "memory_id": memory_id,
        "old_content": old_content,
        "new_content": new_content,
        "event": event,
        "changed_at": changed_at,
        "metadata": {},
    }


# ── MemoryVersioner tests ──────────────────────────────────────────────


class TestMemoryVersioner:

    def test_save_version_delegates_to_storage(self):
        storage = _make_mock_storage()
        versioner = MemoryVersioner(storage)

        versioner.save_version("m1", "old", "new", event="UPDATE")

        storage.save_history.assert_called_once_with(
            "m1", "old", "new", event="UPDATE", metadata=None
        )

    def test_save_version_with_metadata(self):
        storage = _make_mock_storage()
        versioner = MemoryVersioner(storage)

        meta = {"reason": "correction"}
        versioner.save_version("m1", None, "new", metadata=meta)

        storage.save_history.assert_called_once_with(
            "m1", None, "new", event="UPDATE", metadata=meta
        )

    def test_save_version_custom_event(self):
        storage = _make_mock_storage()
        versioner = MemoryVersioner(storage)

        versioner.save_version("m1", None, "new", event="CREATE")

        storage.save_history.assert_called_once_with(
            "m1", None, "new", event="CREATE", metadata=None
        )

    def test_get_versions_returns_history(self):
        history = [
            _history_entry(1, "m1", None, "v1"),
            _history_entry(2, "m1", "v1", "v2"),
        ]
        storage = _make_mock_storage(history)
        versioner = MemoryVersioner(storage)

        versions = versioner.get_versions("m1")

        assert len(versions) == 2
        assert versions[0]["new_content"] == "v1"
        assert versions[1]["new_content"] == "v2"
        storage.get_history.assert_called_once_with("m1")

    def test_get_versions_empty(self):
        storage = _make_mock_storage([])
        versioner = MemoryVersioner(storage)

        assert versioner.get_versions("m1") == []

    def test_get_version_count(self):
        history = [_history_entry(i, "m1", None, f"v{i}") for i in range(5)]
        storage = _make_mock_storage(history)
        versioner = MemoryVersioner(storage)

        assert versioner.get_version_count("m1") == 5

    def test_get_version_count_empty(self):
        storage = _make_mock_storage([])
        versioner = MemoryVersioner(storage)

        assert versioner.get_version_count("m1") == 0

    def test_get_latest_returns_last_entry(self):
        history = [
            _history_entry(1, "m1", None, "v1"),
            _history_entry(2, "m1", "v1", "v2"),
        ]
        storage = _make_mock_storage(history)
        versioner = MemoryVersioner(storage)

        latest = versioner.get_latest("m1")
        assert latest is not None
        assert latest["new_content"] == "v2"

    def test_get_latest_empty_returns_none(self):
        storage = _make_mock_storage([])
        versioner = MemoryVersioner(storage)

        assert versioner.get_latest("m1") is None

    def test_rollback_restores_content(self):
        history = [
            _history_entry(1, "m1", None, "version one"),
            _history_entry(2, "m1", "version one", "version two"),
        ]
        storage = _make_mock_storage(history)
        mock_mem = MagicMock()
        mock_mem.content = "version two"
        storage.get_memory.return_value = mock_mem

        versioner = MemoryVersioner(storage)
        result = versioner.rollback("m1", version_index=0)

        assert result == "version one"
        storage.update_memory.assert_called_once_with(
            "m1", content="version one"
        )

    def test_rollback_records_rollback_event(self):
        history = [
            _history_entry(1, "m1", None, "v1"),
            _history_entry(2, "m1", "v1", "v2"),
        ]
        storage = _make_mock_storage(history)
        mock_mem = MagicMock()
        mock_mem.content = "v2"
        storage.get_memory.return_value = mock_mem

        versioner = MemoryVersioner(storage)
        versioner.rollback("m1", version_index=0)

        # save_history called for the rollback event
        rollback_call = storage.save_history.call_args
        assert rollback_call.args[0] == "m1"  # memory_id
        assert rollback_call.kwargs["new_content"] == "v1"  # restored
        assert rollback_call.kwargs["event"] == "ROLLBACK"

    def test_rollback_to_last_version(self):
        history = [
            _history_entry(1, "m1", None, "v1"),
            _history_entry(2, "m1", "v1", "v2"),
            _history_entry(3, "m1", "v2", "v3"),
        ]
        storage = _make_mock_storage(history)
        mock_mem = MagicMock()
        mock_mem.content = "v3"
        storage.get_memory.return_value = mock_mem

        versioner = MemoryVersioner(storage)
        result = versioner.rollback("m1", version_index=1)

        assert result == "v2"

    def test_rollback_empty_history_raises(self):
        storage = _make_mock_storage([])
        versioner = MemoryVersioner(storage)

        with pytest.raises(ValueError, match="No version history"):
            versioner.rollback("m1", version_index=0)

    def test_rollback_out_of_range_raises(self):
        history = [_history_entry(1, "m1", None, "v1")]
        storage = _make_mock_storage(history)
        versioner = MemoryVersioner(storage)

        with pytest.raises(IndexError, match="out of range"):
            versioner.rollback("m1", version_index=5)

    def test_rollback_negative_index_raises(self):
        history = [_history_entry(1, "m1", None, "v1")]
        storage = _make_mock_storage(history)
        versioner = MemoryVersioner(storage)

        with pytest.raises(IndexError, match="out of range"):
            versioner.rollback("m1", version_index=-1)

    def test_rollback_no_memory_still_restores(self):
        """If memory is deleted, rollback should still call update."""
        history = [_history_entry(1, "m1", None, "v1")]
        storage = _make_mock_storage(history)
        storage.get_memory.return_value = None

        versioner = MemoryVersioner(storage)
        result = versioner.rollback("m1", version_index=0)

        assert result == "v1"
        storage.update_memory.assert_called_once_with(
            "m1", content="v1"
        )

    def test_rollback_metadata_includes_version_info(self):
        history = [
            _history_entry(
                42, "m1", None, "v1",
                changed_at="2025-06-15T10:00:00+00:00",
            ),
        ]
        storage = _make_mock_storage(history)
        storage.get_memory.return_value = MagicMock(content="current")

        versioner = MemoryVersioner(storage)
        versioner.rollback("m1", version_index=0)

        rollback_call = storage.save_history.call_args
        metadata = rollback_call.kwargs["metadata"]
        assert metadata["rolled_back_to_index"] == 0
        assert "2025-06-15" in metadata["rolled_back_to_changed_at"]
