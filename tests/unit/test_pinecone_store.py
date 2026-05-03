"""Tests for PineconeVectorStore using mocked pinecone."""
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


def _make_metadata(mem: Memory | None = None) -> dict[str, Any]:
    """Return Pinecone metadata dict for a memory."""
    if mem is None:
        mem = _make_memory()
    return {
        "content": mem.content,
        "memory_type": mem.memory_type.value,
        "scope": mem.scope.value,
        "user_id": mem.user_id or "",
        "agent_id": mem.agent_id or "",
        "session_id": mem.session_id or "",
        "tags": json.dumps(list(mem.tags)),
        "source": mem.source or "",
        "importance": mem.importance,
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
def _patch_pinecone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a mock pinecone module."""
    mock_pinecone = MagicMock()
    mock_index = MagicMock()
    mock_client = MagicMock()

    # list_indexes returns objects with .name attribute
    mock_idx_obj = MagicMock()
    mock_idx_obj.name = "neuralmem"
    mock_client.list_indexes.return_value = [mock_idx_obj]
    mock_client.Index.return_value = mock_index
    mock_pinecone.Pinecone.return_value = mock_client

    monkeypatch.setitem(sys.modules, "pinecone", mock_pinecone)

    # Re-import so the module-level try/except picks up mock
    if "neuralmem.storage.pinecone_store" in sys.modules:
        del sys.modules["neuralmem.storage.pinecone_store"]


def _make_store() -> Any:
    from neuralmem.storage.pinecone_store import (
        PineconeVectorStore,
    )
    return PineconeVectorStore(config={"api_key": "test"})


class TestPineconeVectorStore:
    """Tests for PineconeVectorStore."""

    def test_is_storage_backend(self) -> None:
        from neuralmem.storage.pinecone_store import (
            PineconeVectorStore,
        )
        assert issubclass(
            PineconeVectorStore, StorageBackend
        )

    def test_init_creates_client(self) -> None:
        store = _make_store()
        assert store._index is not None

    def test_save_memory(self) -> None:
        store = _make_store()
        mem = _make_memory()
        result = store.save_memory(mem)
        assert result == "test-mem-001"
        store._index.upsert.assert_called_once()

    def test_save_memory_with_default_embedding(self) -> None:
        store = _make_store()
        mem = _make_memory(embedding=None)
        result = store.save_memory(mem)
        assert result == "test-mem-001"
        call_args = store._index.upsert.call_args
        vectors = call_args.kwargs["vectors"]
        assert len(vectors[0]["values"]) == 384

    def test_get_memory_returns_none_when_missing(
        self,
    ) -> None:
        store = _make_store()
        store._index.fetch.return_value = {"vectors": {}}
        assert store.get_memory("missing") is None

    def test_get_memory_returns_memory(self) -> None:
        store = _make_store()
        mem = _make_memory()
        store._index.fetch.return_value = {
            "vectors": {
                "test-mem-001": {
                    "values": [0.1] * 384,
                    "metadata": _make_metadata(mem),
                }
            }
        }
        result = store.get_memory("test-mem-001")
        assert result is not None
        assert result.id == "test-mem-001"
        assert result.content == "hello world"

    def test_update_memory_not_found(self) -> None:
        store = _make_store()
        store._index.fetch.return_value = {"vectors": {}}
        with pytest.raises(Exception):
            store.update_memory("missing", content="new")

    def test_update_memory_success(self) -> None:
        store = _make_store()
        mem = _make_memory()
        store._index.fetch.return_value = {
            "vectors": {
                "test-mem-001": {
                    "values": [0.1] * 384,
                    "metadata": _make_metadata(mem),
                }
            }
        }
        store.update_memory(
            "test-mem-001", content="updated"
        )
        store._index.upsert.assert_called()

    def test_delete_memory_by_id(self) -> None:
        store = _make_store()
        count = store.delete_memories(memory_id="m1")
        assert count == 1
        store._index.delete.assert_called_once()

    def test_delete_memories_by_user(self) -> None:
        store = _make_store()
        mem1 = _make_memory(id="m1", user_id="u1")
        mem2 = _make_memory(id="m2", user_id="u2")
        store._index.query.return_value = {
            "matches": [
                {
                    "id": "m1",
                    "metadata": _make_metadata(mem1),
                },
                {
                    "id": "m2",
                    "metadata": _make_metadata(mem2),
                },
            ]
        }
        count = store.delete_memories(user_id="u1")
        assert count == 1

    def test_vector_search_empty(self) -> None:
        store = _make_store()
        store._index.query.return_value = {"matches": []}
        results = store.vector_search([0.1] * 384)
        assert results == []

    def test_vector_search_returns_scores(self) -> None:
        store = _make_store()
        store._index.query.return_value = {
            "matches": [
                {"id": "m1", "score": 0.8},
                {"id": "m2", "score": 0.6},
            ]
        }
        results = store.vector_search([0.1] * 384, limit=2)
        assert len(results) == 2
        assert results[0][0] == "m1"
        assert results[0][1] == 0.8

    def test_vector_search_with_filters(self) -> None:
        store = _make_store()
        store._index.query.return_value = {
            "matches": [{"id": "m1", "score": 0.9}]
        }
        results = store.vector_search(
            [0.1] * 384,
            user_id="u1",
            memory_types=[MemoryType.SEMANTIC],
        )
        assert len(results) == 1
        call_args = store._index.query.call_args
        assert call_args.kwargs.get("filter") is not None

    def test_keyword_search(self) -> None:
        store = _make_store()
        mem = _make_memory()
        store._index.query.return_value = {
            "matches": [
                {
                    "id": "m1",
                    "metadata": _make_metadata(mem),
                }
            ]
        }
        results = store.keyword_search("hello")
        assert len(results) == 1
        assert results[0][1] == 1.0

    def test_list_memories(self) -> None:
        store = _make_store()
        mem = _make_memory()
        store._index.query.return_value = {
            "matches": [
                {
                    "id": "m1",
                    "values": [0.1] * 384,
                    "metadata": _make_metadata(mem),
                }
            ]
        }
        mems = store.list_memories(user_id="u1")
        assert len(mems) == 1

    def test_record_access(self) -> None:
        store = _make_store()
        mem = _make_memory()
        store._index.fetch.return_value = {
            "vectors": {
                "m1": {
                    "values": [0.1] * 384,
                    "metadata": _make_metadata(mem),
                }
            }
        }
        store.record_access("m1")
        store._index.upsert.assert_called()

    def test_record_access_missing(self) -> None:
        store = _make_store()
        store._index.fetch.return_value = {"vectors": {}}
        store.record_access("missing")
        store._index.upsert.assert_not_called()

    def test_get_stats(self) -> None:
        store = _make_store()
        store._index.describe_index_stats.return_value = {
            "total_vector_count": 42
        }
        stats = store.get_stats()
        assert stats["backend"] == "pinecone"
        assert stats["total_memories"] == 42

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
        store._index.fetch.return_value = {
            "vectors": {
                "m1": {
                    "values": [0.1] * 384,
                    "metadata": _make_metadata(mem),
                }
            }
        }
        store.batch_record_access(["m1"])
        store._index.upsert.assert_called()

    def test_find_similar(self) -> None:
        store = _make_store()
        store._index.query.return_value = {
            "matches": [
                {"id": "m1", "score": 0.98}
            ]
        }
        mem = _make_memory(id="m1")
        store._index.fetch.return_value = {
            "vectors": {
                "m1": {
                    "values": [0.1] * 384,
                    "metadata": _make_metadata(mem),
                }
            }
        }
        similar = store.find_similar(
            [0.1] * 384, threshold=0.9
        )
        assert len(similar) == 1

    def test_temporal_search(self) -> None:
        store = _make_store()
        store._index.query.return_value = {
            "matches": [
                {"id": "m1", "score": 0.9}
            ]
        }
        mem = _make_memory(id="m1")
        store._index.fetch.return_value = {
            "vectors": {
                "m1": {
                    "values": [0.1] * 384,
                    "metadata": _make_metadata(mem),
                }
            }
        }
        result = store.temporal_search([0.1] * 384)
        assert len(result) == 1

    def test_import_guard(self) -> None:
        """Test that _check_pinecone raises when None."""
        import neuralmem.storage.pinecone_store as mod

        original = mod.Pinecone
        mod.Pinecone = None
        with pytest.raises(ImportError, match="pinecone"):
            mod._check_pinecone()
        mod.Pinecone = original
