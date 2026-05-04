"""Tests for the V1.9 versioning module (MemoryVersion, VersionStore, VersionManager)."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from neuralmem.versioning.version import MemoryVersion
from neuralmem.versioning.store import VersionStore
from neuralmem.versioning.manager import VersionManager


# ── MemoryVersion tests ──────────────────────────────────────────────


class TestMemoryVersion:

    def test_basic_creation(self):
        v = MemoryVersion(
            version_number=1,
            memory_id="mem-1",
            content="hello",
            event="CREATE",
        )
        assert v.version_number == 1
        assert v.memory_id == "mem-1"
        assert v.content == "hello"
        assert v.parent is None
        assert v.is_latest is False
        assert v.event == "CREATE"

    def test_frozen_cannot_mutate(self):
        v = MemoryVersion(
            version_number=1,
            memory_id="mem-1",
            content="hello",
        )
        with pytest.raises(AttributeError):
            v.content = "changed"

    def test_invalid_version_number(self):
        with pytest.raises(ValueError, match="version_number"):
            MemoryVersion(version_number=0, memory_id="mem-1", content="x")

    def test_empty_memory_id(self):
        with pytest.raises(ValueError, match="memory_id"):
            MemoryVersion(version_number=1, memory_id="", content="x")

    def test_to_dict_roundtrip(self):
        v = MemoryVersion(
            version_number=2,
            memory_id="mem-1",
            content="hello world",
            parent=1,
            is_latest=True,
            changes={"content": "hello → hello world"},
            event="UPDATE",
            metadata={"reason": "correction"},
        )
        d = v.to_dict()
        restored = MemoryVersion.from_dict(d)
        assert restored == v

    def test_default_created_at(self):
        before = datetime.now(timezone.utc)
        v = MemoryVersion(version_number=1, memory_id="m", content="c")
        after = datetime.now(timezone.utc)
        assert before <= v.created_at <= after

    def test_from_dict_iso_timestamp(self):
        ts = "2025-06-15T10:30:00+00:00"
        d = {
            "version_number": 3,
            "memory_id": "mem-1",
            "content": "c",
            "parent": 2,
            "is_latest": True,
            "changes": {},
            "created_at": ts,
            "event": "DELETE",
            "metadata": {},
        }
        v = MemoryVersion.from_dict(d)
        assert v.created_at == datetime.fromisoformat(ts)
        assert v.event == "DELETE"


# ── VersionStore tests ───────────────────────────────────────────────


class TestVersionStore:

    def test_save_and_get_history(self):
        store = VersionStore()
        v1 = MemoryVersion(1, "mem-1", "first", is_latest=True, event="CREATE")
        store.save_version(v1)

        history = store.get_history("mem-1")
        assert len(history) == 1
        assert history[0].version_number == 1

    def test_get_version_found(self):
        store = VersionStore()
        v1 = MemoryVersion(1, "mem-1", "first", is_latest=True)
        store.save_version(v1)

        found = store.get_version("mem-1", 1)
        assert found is not None
        assert found.content == "first"

    def test_get_version_not_found(self):
        store = VersionStore()
        assert store.get_version("mem-1", 1) is None

    def test_get_latest(self):
        store = VersionStore()
        store.save_version(MemoryVersion(1, "mem-1", "v1", is_latest=True))
        store.save_version(MemoryVersion(2, "mem-1", "v2", parent=1, is_latest=True))

        latest = store.get_latest("mem-1")
        assert latest is not None
        assert latest.version_number == 2
        assert latest.content == "v2"

    def test_get_latest_empty(self):
        store = VersionStore()
        assert store.get_latest("mem-1") is None

    def test_multiple_memories_isolated(self):
        store = VersionStore()
        store.save_version(MemoryVersion(1, "mem-a", "a", is_latest=True))
        store.save_version(MemoryVersion(1, "mem-b", "b", is_latest=True))

        assert store.get_version_count("mem-a") == 1
        assert store.get_version_count("mem-b") == 1
        assert store.get_history("mem-a")[0].content == "a"
        assert store.get_history("mem-b")[0].content == "b"

    def test_is_latest_updated_on_save(self):
        store = VersionStore()
        v1 = MemoryVersion(1, "mem-1", "first", is_latest=True)
        store.save_version(v1)

        v2 = MemoryVersion(2, "mem-1", "second", parent=1, is_latest=True)
        store.save_version(v2)

        history = store.get_history("mem-1")
        assert history[0].is_latest is False
        assert history[1].is_latest is True

    def test_clear(self):
        store = VersionStore()
        store.save_version(MemoryVersion(1, "mem-1", "x", is_latest=True))
        store.clear("mem-1")
        assert store.get_history("mem-1") == []

    def test_backend_delegation(self):
        backend = MagicMock()
        store = VersionStore(storage=backend)
        v = MemoryVersion(1, "mem-1", "hello", is_latest=True, event="CREATE")
        store.save_version(v)

        backend.save_history.assert_called_once()
        args, kwargs = backend.save_history.call_args
        assert kwargs["event"] == "CREATE"
        assert kwargs["new_content"] == "hello"

    def test_backend_failure_graceful(self):
        backend = MagicMock()
        backend.save_history.side_effect = RuntimeError("db down")
        store = VersionStore(storage=backend)
        v = MemoryVersion(1, "mem-1", "hello", is_latest=True)
        store.save_version(v)  # should not raise
        assert store.get_version_count("mem-1") == 1


# ── VersionManager tests ─────────────────────────────────────────────


class TestVersionManagerCreate:

    def test_create_first_version(self):
        mgr = VersionManager(VersionStore())
        v = mgr.create_version("mem-1", "initial", event="CREATE")
        assert v.version_number == 1
        assert v.parent is None
        assert v.is_latest is True
        assert v.event == "CREATE"

    def test_create_subsequent_version(self):
        mgr = VersionManager(VersionStore())
        v1 = mgr.create_version("mem-1", "first", event="CREATE")
        v2 = mgr.create_version("mem-1", "second")
        assert v2.version_number == 2
        assert v2.parent == 1
        assert v2.is_latest is True

    def test_create_with_changes_and_metadata(self):
        mgr = VersionManager(VersionStore())
        v = mgr.create_version(
            "mem-1",
            "content",
            changes={"field": "old → new"},
            metadata={"author": "test"},
        )
        assert v.changes == {"field": "old → new"}
        assert v.metadata == {"author": "test"}

    def test_create_empty_memory_id_raises(self):
        mgr = VersionManager(VersionStore())
        with pytest.raises(ValueError, match="memory_id"):
            mgr.create_version("", "content")


class TestVersionManagerRollback:

    def test_rollback_restores_content(self):
        mgr = VersionManager(VersionStore())
        v1 = mgr.create_version("mem-1", "version one", event="CREATE")
        mgr.create_version("mem-1", "version two")

        rollback = mgr.rollback("mem-1", v1.version_number)
        assert rollback.content == "version one"
        assert rollback.event == "ROLLBACK"
        assert rollback.metadata["rolled_back_to"] == v1.version_number

    def test_rollback_creates_new_version(self):
        mgr = VersionManager(VersionStore())
        mgr.create_version("mem-1", "v1", event="CREATE")
        mgr.create_version("mem-1", "v2")
        rollback = mgr.rollback("mem-1", 1)

        assert rollback.version_number == 3
        assert mgr.get_version_count("mem-1") == 3

    def test_rollback_empty_history_raises(self):
        mgr = VersionManager(VersionStore())
        with pytest.raises(ValueError, match="No version history"):
            mgr.rollback("mem-1", 1)

    def test_rollback_missing_version_raises(self):
        mgr = VersionManager(VersionStore())
        mgr.create_version("mem-1", "v1", event="CREATE")
        with pytest.raises(ValueError, match="Version 99 not found"):
            mgr.rollback("mem-1", 99)

    def test_rollback_sets_latest(self):
        mgr = VersionManager(VersionStore())
        v1 = mgr.create_version("mem-1", "v1", event="CREATE")
        mgr.create_version("mem-1", "v2")
        rollback = mgr.rollback("mem-1", 1)

        latest = mgr.get_latest("mem-1")
        assert latest is not None
        assert latest.version_number == rollback.version_number
        assert latest.is_latest is True


class TestVersionManagerDiff:

    def test_diff_between_versions(self):
        mgr = VersionManager(VersionStore())
        mgr.create_version("mem-1", "line one\nline two", event="CREATE")
        mgr.create_version("mem-1", "line one\nline three")

        diff = mgr.diff("mem-1", 1, 2)
        assert "-line two" in diff
        assert "+line three" in diff

    def test_diff_same_content(self):
        mgr = VersionManager(VersionStore())
        mgr.create_version("mem-1", "same", event="CREATE")
        mgr.create_version("mem-1", "same")

        diff = mgr.diff("mem-1", 1, 2)
        # unified diff of identical content has no +/- lines
        assert "-same" not in diff or "+same" not in diff

    def test_diff_version_not_found(self):
        mgr = VersionManager(VersionStore())
        mgr.create_version("mem-1", "x", event="CREATE")
        with pytest.raises(ValueError, match="Version 99 not found"):
            mgr.diff("mem-1", 1, 99)

    def test_diff_single_line(self):
        mgr = VersionManager(VersionStore())
        mgr.create_version("mem-1", "hello world", event="CREATE")
        mgr.create_version("mem-1", "hello universe")

        diff = mgr.diff("mem-1", 1, 2)
        assert "-hello world" in diff
        assert "+hello universe" in diff


class TestVersionManagerList:

    def test_list_versions(self):
        mgr = VersionManager(VersionStore())
        mgr.create_version("mem-1", "a", event="CREATE")
        mgr.create_version("mem-1", "b")

        versions = mgr.list_versions("mem-1")
        assert len(versions) == 2
        assert versions[0]["version_number"] == 1
        assert versions[1]["version_number"] == 2
        assert all(isinstance(v, dict) for v in versions)

    def test_list_versions_empty(self):
        mgr = VersionManager(VersionStore())
        assert mgr.list_versions("mem-1") == []

    def test_get_version_count(self):
        mgr = VersionManager(VersionStore())
        assert mgr.get_version_count("mem-1") == 0
        mgr.create_version("mem-1", "a", event="CREATE")
        assert mgr.get_version_count("mem-1") == 1


# ── Integration / end-to-end ─────────────────────────────────────────


class TestVersioningIntegration:

    def test_full_lifecycle(self):
        """Create → update → diff → rollback → list."""
        mgr = VersionManager(VersionStore())

        v1 = mgr.create_version("mem-1", "The sky is blue.", event="CREATE")
        v2 = mgr.create_version("mem-1", "The sky is azure.")
        v3 = mgr.create_version("mem-1", "The sky is cyan.")

        assert mgr.get_version_count("mem-1") == 3
        assert mgr.get_latest("mem-1").version_number == 3

        diff = mgr.diff("mem-1", v1.version_number, v3.version_number)
        assert "-The sky is blue." in diff
        assert "+The sky is cyan." in diff

        rollback = mgr.rollback("mem-1", v1.version_number)
        assert rollback.content == "The sky is blue."
        assert rollback.event == "ROLLBACK"
        assert mgr.get_latest("mem-1").version_number == 4

        versions = mgr.list_versions("mem-1")
        assert len(versions) == 4
        assert versions[-1]["event"] == "ROLLBACK"

    def test_multiple_memories(self):
        mgr = VersionManager(VersionStore())
        mgr.create_version("mem-a", "alpha", event="CREATE")
        mgr.create_version("mem-b", "beta", event="CREATE")
        mgr.create_version("mem-a", "alpha2")

        assert mgr.get_version_count("mem-a") == 2
        assert mgr.get_version_count("mem-b") == 1
        assert mgr.get_latest("mem-a").content == "alpha2"
        assert mgr.get_latest("mem-b").content == "beta"

    def test_store_with_backend_integration(self):
        backend = MagicMock()
        store = VersionStore(storage=backend)
        mgr = VersionManager(store)

        mgr.create_version("mem-1", "hello", event="CREATE")
        mgr.create_version("mem-1", "world")

        assert backend.save_history.call_count == 2
        assert mgr.get_version_count("mem-1") == 2
