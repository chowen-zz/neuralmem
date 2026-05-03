"""Tests for MilvusVectorStore using mocked pymilvus."""
from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from neuralmem.core.types import Memory, MemoryType
from neuralmem.storage.base import StorageBackend


def _make_memory(**overrides: Any) -> Memory:
    """Return a Memory with sensible defaults."""
    defaults: dict[str, Any] = {
        "id": "test-mem-001",
        "content": "hello world",
        "memory_type": MemoryType.SEMANTIC,
        "user_id": "u1",
        "importance": 0.7,
        "embedding": [0.1] * 384,
    }
    defaults.update(overrides)
    return Memory(**defaults)


def _make_row(mem: Memory | None = None) -> dict[str, Any]:
    """Return a Milvus-style row dict."""
    if mem is None:
        mem = _make_memory()
    return {
        "id": mem.id,
        "embedding": mem.embedding or [0.0] * 384,
        "content": mem.content,
        "memory_type": mem.memory_type.value,
        "scope": mem.scope.value,
        "user_id": mem.user_id or "",
        "agent_id": mem.agent_id or "",
        "session_id": mem.session_id or "",
        "tags": json.dumps(list(mem.tags)),
        "source": mem.source or "",
        "importance": float(mem.importance),
        "entity_ids": json.dumps(list(mem.entity_ids)),
        "is_active": int(mem.is_active),
        "superseded_by": mem.superseded_by or "",
        "supersedes": json.dumps(list(mem.supersedes)),
        "created_at": mem.created_at.isoformat(),
        "updated_at": mem.updated_at.isoformat(),
        "last_accessed": mem.last_accessed.isoformat(),
        "access_count": mem.access_count,
        "expires_at": (
            mem.expires_at.isoformat()
            if mem.expires_at
            else ""
        ),
    }


@pytest.fixture(autouse=True)
def _patch_milvus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a mock pymilvus module."""
    mock_milvus = MagicMock()
    mock_collection = MagicMock()
    mock_collection.num_entities = 10
    mock_milvus.Collection.return_value = mock_collection
    mock_milvus.CollectionSchema = MagicMock()
    mock_milvus.FieldSchema = MagicMock()
    mock_milvus.DataType = MagicMock(
        VARCHAR="VARCHAR",
        FLOAT_VECTOR="FLOAT_VECTOR",
        FLOAT="FLOAT",
        INT64="INT64",
    )

    monkeypatch.setitem(
        sys.modules, "pymilvus", mock_milvus
    )

    if "neuralmem.storage.milvus_store" in sys.modules:
        del sys.modules["neuralmem.storage.milvus_store"]


def _make_store() -> Any:
    from neuralmem.storage.milvus_store import (
        MilvusVectorStore,
    )
    return MilvusVectorStore(config={})


class TestMilvusVectorStore:
    """Tests for MilvusVectorStore."""

    def test_is_storage_backend(self) -> None:
        from neuralmem.storage.milvus_store import (
            MilvusVectorStore,
        )
        assert issubclass(
            MilvusVectorStore, StorageBackend
        )

    def test_init_creates_collection(self) -> None:
        store = _make_store()
        assert store._collection is not None

    def test_save_memory(self) -> None:
        store = _make_store()
        mem = _make_memory()
        result = store.save_memory(mem)
        assert result == "test-mem-001"
        store._collection.insert.assert_called_once()

    def test_save_memory_with_default_embedding(self) -> None:
        store = _make_store()
        mem = _make_memory(embedding=None)
        result = store.save_memory(mem)
        assert result == "test-mem-001"

    def test_get_memory_returns_none_when_missing(
        self,
    ) -> None:
        store = _make_store()
        store._collection.query.return_value = []
        assert store.get_memory("missing") is None

    def test_get_memory_returns_memory(self) -> None:
        store = _make_store()
        mem = _make_memory()
        store._collection.query.return_value = [
            _make_row(mem)
        ]
        result = store.get_memory("test-mem-001")
        assert result is not None
        assert result.id == "test-mem-001"
        assert result.content == "hello world"

    def test_update_memory_not_found(self) -> None:
        store = _make_store()
        store._collection.query.return_value = []
        with pytest.raises(Exception):
            store.update_memory("missing", content="new")

    def test_update_memory_success(self) -> None:
        store = _make_store()
        mem = _make_memory()
        store._collection.query.return_value = [
            _make_row(mem)
        ]
        store.update_memory(
            "test-mem-001", content="updated"
        )
        store._collection.upsert.assert_called()

    def test_delete_memory_by_id(self) -> None:
        store = _make_store()
        count = store.delete_memories(memory_id="m1")
        assert count == 1
        store._collection.delete.assert_called_once()

    def test_delete_memories_by_user(self) -> None:
        store = _make_store()
        mem1 = _make_memory(id="m1", user_id="u1")
        mem2 = _make_memory(id="m2", user_id="u2")
        store._collection.query.return_value = [
            _make_row(mem1),
            _make_row(mem2),
        ]
        count = store.delete_memories(user_id="u1")
        assert count == 1

    def test_vector_search_empty(self) -> None:
        store = _make_store()
        store._collection.search.return_value = []
        results = store.vector_search([0.1] * 384)
        assert results == []

    def test_vector_search_returns_scores(self) -> None:
        store = _make_store()
        hit1 = MagicMock()
        hit1.score = 0.8
        hit1.entity = {"id": "m1"}
        hit2 = MagicMock()
        hit2.score = 0.6
        hit2.entity = {"id": "m2"}
        store._collection.search.return_value = [
            [hit1, hit2]
        ]
        results = store.vector_search(
            [0.1] * 384, limit=2
        )
        assert len(results) == 2
        assert results[0][0] == "m1"
        assert results[0][1] == 0.8

    def test_keyword_search(self) -> None:
        store = _make_store()
        mem = _make_memory()
        store._collection.query.return_value = [
            _make_row(mem)
        ]
        results = store.keyword_search("hello")
        assert len(results) == 1
        assert results[0][1] == 1.0

    def test_list_memories(self) -> None:
        store = _make_store()
        mem = _make_memory()
        store._collection.query.return_value = [
            _make_row(mem)
        ]
        mems = store.list_memories(user_id="u1")
        assert len(mems) == 1

    def test_record_access(self) -> None:
        store = _make_store()
        mem = _make_memory()
        store._collection.query.return_value = [
            _make_row(mem)
        ]
        store.record_access("m1")
        store._collection.upsert.assert_called()

    def test_record_access_missing(self) -> None:
        store = _make_store()
        store._collection.query.return_value = []
        store.record_access("missing")
        store._collection.upsert.assert_not_called()

    def test_get_stats(self) -> None:
        store = _make_store()
        stats = store.get_stats()
        assert stats["backend"] == "milvus"
        assert stats["total_memories"] == 10

    def test_save_and_get_history(self) -> None:
        store = _make_store()
        store.save_history("m1", "old", "new", "UPDATE")
        assert store.get_history("m1") == []

    def test_graph_snapshot_noop(self) -> None:
        store = _make_store()
        store.save_graph_snapshot({"key": "val"})
        assert store.load_graph_snapshot() is None

    def test_batch_record_access(self) -> None:
        store = _make_store()
        mem = _make_memory()
        store._collection.query.return_value = [
            _make_row(mem)
        ]
        store.batch_record_access(["m1"])
        store._collection.upsert.assert_called()

    def test_find_similar(self) -> None:
        store = _make_store()
        hit = MagicMock()
        hit.score = 0.98
        hit.entity = {"id": "m1"}
        store._collection.search.return_value = [[hit]]
        mem = _make_memory(id="m1")
        store._collection.query.return_value = [
            _make_row(mem)
        ]
        similar = store.find_similar(
            [0.1] * 384, threshold=0.9
        )
        assert len(similar) == 1

    def test_import_guard(self) -> None:
        """Test that _check_milvus raises when None."""
        import neuralmem.storage.milvus_store as mod

        original = mod.Collection
        mod.Collection = None
        with pytest.raises(ImportError, match="pymilvus"):
            mod._check_milvus()
        mod.Collection = original
