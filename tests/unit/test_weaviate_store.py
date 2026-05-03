"""Tests for WeaviateVectorStore using mocked weaviate."""
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


def _make_weaviate_obj(
    mem: Memory | None = None,
) -> dict[str, Any]:
    """Return a Weaviate-style object."""
    if mem is None:
        mem = _make_memory()
    return {
        "properties": {
            "content": mem.content,
            "memory_type": mem.memory_type.value,
            "scope": mem.scope.value,
            "user_id": mem.user_id or "",
            "agent_id": mem.agent_id or "",
            "session_id": mem.session_id or "",
            "tags_json": json.dumps(list(mem.tags)),
            "source": mem.source or "",
            "importance": float(mem.importance),
            "entity_ids_json": json.dumps(
                list(mem.entity_ids)
            ),
            "is_active": mem.is_active,
            "superseded_by": mem.superseded_by or "",
            "supersedes_json": json.dumps(
                list(mem.supersedes)
            ),
            "created_at": mem.created_at.isoformat(),
            "updated_at": mem.updated_at.isoformat(),
            "last_accessed": mem.last_accessed.isoformat(),
            "access_count": mem.access_count,
            "expires_at": (
                mem.expires_at.isoformat()
                if mem.expires_at
                else ""
            ),
        },
        "_additional": {
            "id": mem.id,
            "vector": mem.embedding,
        },
        "vector": mem.embedding,
    }


@pytest.fixture(autouse=True)
def _patch_weaviate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provide a mock weaviate module."""
    mock_weaviate = MagicMock()
    mock_client = MagicMock()
    mock_client.schema.exists.return_value = True
    mock_weaviate.Client.return_value = mock_client
    mock_weaviate.auth.AuthApiKey = MagicMock()

    monkeypatch.setitem(
        sys.modules, "weaviate", mock_weaviate
    )

    if "neuralmem.storage.weaviate_store" in sys.modules:
        del sys.modules["neuralmem.storage.weaviate_store"]


def _make_store() -> Any:
    from neuralmem.storage.weaviate_store import (
        WeaviateVectorStore,
    )
    return WeaviateVectorStore(config={})


class TestWeaviateVectorStore:
    """Tests for WeaviateVectorStore."""

    def test_is_storage_backend(self) -> None:
        from neuralmem.storage.weaviate_store import (
            WeaviateVectorStore,
        )
        assert issubclass(
            WeaviateVectorStore, StorageBackend
        )

    def test_init_creates_client(self) -> None:
        store = _make_store()
        assert store._client is not None

    def test_save_memory(self) -> None:
        store = _make_store()
        mem = _make_memory()
        result = store.save_memory(mem)
        assert result == "test-mem-001"
        store._client.data_object.create.assert_called_once()

    def test_save_memory_with_default_embedding(self) -> None:
        store = _make_store()
        mem = _make_memory(embedding=None)
        result = store.save_memory(mem)
        assert result == "test-mem-001"
        call_args = (
            store._client.data_object.create.call_args
        )
        vector = call_args.kwargs.get("vector")
        assert len(vector) == 384

    def test_get_memory_returns_none_when_missing(
        self,
    ) -> None:
        store = _make_store()
        store._client.data_object.get_by_id.return_value = (
            None
        )
        assert store.get_memory("missing") is None

    def test_get_memory_returns_memory(self) -> None:
        store = _make_store()
        mem = _make_memory()
        store._client.data_object.get_by_id.return_value = (
            _make_weaviate_obj(mem)
        )
        result = store.get_memory("test-mem-001")
        assert result is not None
        assert result.id == "test-mem-001"
        assert result.content == "hello world"

    def test_update_memory_not_found(self) -> None:
        store = _make_store()
        store._client.data_object.get_by_id.return_value = (
            None
        )
        with pytest.raises(Exception):
            store.update_memory("missing", content="new")

    def test_update_memory_success(self) -> None:
        store = _make_store()
        mem = _make_memory()
        store._client.data_object.get_by_id.return_value = (
            _make_weaviate_obj(mem)
        )
        store.update_memory(
            "test-mem-001", content="updated"
        )
        store._client.data_object.update.assert_called()

    def test_delete_memory_by_id(self) -> None:
        store = _make_store()
        count = store.delete_memories(memory_id="m1")
        assert count == 1
        store._client.data_object.delete.assert_called_once()

    def test_delete_memories_by_user(self) -> None:
        store = _make_store()
        mem1 = _make_memory(id="m1", user_id="u1")
        mem2 = _make_memory(id="m2", user_id="u2")
        query_mock = MagicMock()
        query_mock.with_additional.return_value = query_mock
        query_mock.with_limit.return_value = query_mock
        query_mock.do.return_value = {
            "data": {
                "Get": {
                    "NeuralMem": [
                        {
                            **_make_weaviate_obj(mem1)[
                                "properties"
                            ],
                            "_additional": {"id": "m1"},
                        },
                        {
                            **_make_weaviate_obj(mem2)[
                                "properties"
                            ],
                            "_additional": {"id": "m2"},
                        },
                    ]
                }
            }
        }
        store._client.query.get.return_value = query_mock
        count = store.delete_memories(user_id="u1")
        assert count == 1

    def test_vector_search_empty(self) -> None:
        store = _make_store()
        query_mock = MagicMock()
        query_mock.with_near_vector.return_value = query_mock
        query_mock.with_additional.return_value = query_mock
        query_mock.with_limit.return_value = query_mock
        query_mock.with_where.return_value = query_mock
        query_mock.do.return_value = {
            "data": {"Get": {"NeuralMem": []}}
        }
        store._client.query.get.return_value = query_mock
        results = store.vector_search([0.1] * 384)
        assert results == []

    def test_vector_search_returns_scores(self) -> None:
        store = _make_store()
        query_mock = MagicMock()
        query_mock.with_near_vector.return_value = query_mock
        query_mock.with_additional.return_value = query_mock
        query_mock.with_limit.return_value = query_mock
        query_mock.with_where.return_value = query_mock
        query_mock.do.return_value = {
            "data": {
                "Get": {
                    "NeuralMem": [
                        {
                            "_additional": {
                                "id": "m1",
                                "distance": 0.2,
                            }
                        },
                        {
                            "_additional": {
                                "id": "m2",
                                "distance": 0.5,
                            }
                        },
                    ]
                }
            }
        }
        store._client.query.get.return_value = query_mock
        results = store.vector_search(
            [0.1] * 384, limit=2
        )
        assert len(results) == 2
        assert results[0][0] == "m1"
        assert abs(results[0][1] - 0.8) < 1e-6

    def test_keyword_search(self) -> None:
        store = _make_store()
        mem = _make_memory()
        # Mock list_memories which keyword_search uses
        store.list_memories = MagicMock(  # type: ignore
            return_value=[mem]
        )
        results = store.keyword_search("hello")
        assert len(results) == 1
        assert results[0][1] == 1.0

    def test_list_memories(self) -> None:
        store = _make_store()
        mem = _make_memory()
        query_mock = MagicMock()
        query_mock.with_additional.return_value = query_mock
        query_mock.with_limit.return_value = query_mock
        query_mock.with_where.return_value = query_mock
        query_mock.do.return_value = {
            "data": {
                "Get": {
                    "NeuralMem": [
                        {
                            **_make_weaviate_obj(mem)[
                                "properties"
                            ],
                            "_additional": {
                                "id": "m1",
                                "vector": [0.1] * 384,
                            },
                        }
                    ]
                }
            }
        }
        store._client.query.get.return_value = query_mock
        mems = store.list_memories(user_id="u1")
        assert len(mems) == 1

    def test_record_access(self) -> None:
        store = _make_store()
        mem = _make_memory()
        store._client.data_object.get_by_id.return_value = (
            _make_weaviate_obj(mem)
        )
        store.record_access("m1")
        store._client.data_object.update.assert_called()

    def test_record_access_missing(self) -> None:
        store = _make_store()
        store._client.data_object.get_by_id.return_value = (
            None
        )
        store.record_access("missing")
        store._client.data_object.update.assert_not_called()

    def test_get_stats(self) -> None:
        store = _make_store()
        agg_mock = MagicMock()
        agg_mock.with_meta_count.return_value = agg_mock
        agg_mock.do.return_value = {
            "data": {
                "Aggregate": {
                    "NeuralMem": [
                        {"meta": {"count": 42}}
                    ]
                }
            }
        }
        store._client.query.aggregate.return_value = (
            agg_mock
        )
        stats = store.get_stats()
        assert stats["backend"] == "weaviate"
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
        store._client.data_object.get_by_id.return_value = (
            _make_weaviate_obj(mem)
        )
        store.batch_record_access(["m1"])
        store._client.data_object.update.assert_called()

    def test_find_similar(self) -> None:
        store = _make_store()
        store.vector_search = MagicMock(  # type: ignore
            return_value=[("m1", 0.98)]
        )
        mem = _make_memory(id="m1")
        store.get_memory = MagicMock(  # type: ignore
            return_value=mem
        )
        similar = store.find_similar(
            [0.1] * 384, threshold=0.9
        )
        assert len(similar) == 1

    def test_import_guard(self) -> None:
        """Test that _check_weaviate raises when None."""
        import neuralmem.storage.weaviate_store as mod

        original = mod.weaviate
        mod.weaviate = None
        with pytest.raises(ImportError, match="weaviate"):
            mod._check_weaviate()
        mod.weaviate = original
