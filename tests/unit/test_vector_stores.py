"""Tests for vector store backends and VectorStoreFactory."""
from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from neuralmem.core.types import Memory, MemoryType
from neuralmem.storage.base import StorageBackend

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


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


def _is_subclass_of_backend(cls: type) -> bool:
    return issubclass(cls, StorageBackend)


# ==================================================================
# Factory Tests
# ==================================================================


class TestVectorStoreFactory:
    """Tests for the VectorStoreFactory class."""

    def test_register_and_list(self) -> None:
        from neuralmem.storage.factory import VectorStoreFactory

        backends = VectorStoreFactory.list_backends()
        assert "chroma" in backends
        assert "qdrant" in backends
        assert "faiss" in backends
        assert "redis" in backends

    def test_create_unknown_raises(self) -> None:
        from neuralmem.storage.factory import VectorStoreFactory

        with pytest.raises(ValueError, match="Unknown vector store"):
            VectorStoreFactory.create("nonexistent")

    def test_check_dependency_returns_false_for_missing(
        self,
    ) -> None:
        from neuralmem.storage.factory import VectorStoreFactory

        assert VectorStoreFactory.check_dependency("nonexistent") is False

    def test_registry_info_structure(self) -> None:
        from neuralmem.storage.factory import VectorStoreFactory

        info = VectorStoreFactory.registry_info()
        assert "chroma" in info
        assert "module" in info["chroma"]
        assert "class" in info["chroma"]
        assert "extra" in info["chroma"]
        assert info["chroma"]["extra"] == "chroma"

    def test_factory_register_custom_backend(self) -> None:
        from neuralmem.storage.factory import VectorStoreFactory

        # Register a dummy module that we control
        dummy_mod = types.ModuleType("tests._dummy_store")

        class DummyStore(StorageBackend):
            def __init__(self, **kw: Any) -> None:
                pass

            def save_memory(self, memory: Memory) -> str:
                return memory.id

            def get_memory(self, memory_id: str) -> Memory | None:
                return None

            def update_memory(
                self, memory_id: str, **kwargs: object
            ) -> None:
                pass

            def delete_memories(
                self,
                memory_id: str | None = None,
                user_id: str | None = None,
                before: object = None,
                tags: list[str] | None = None,
                max_importance: float | None = None,
            ) -> int:
                return 0

            def vector_search(
                self,
                vector: list[float],
                user_id: str | None = None,
                memory_types: list[MemoryType] | None = None,
                limit: int = 10,
            ) -> list[tuple[str, float]]:
                return []

            def keyword_search(
                self,
                query: str,
                user_id: str | None = None,
                memory_types: list[MemoryType] | None = None,
                limit: int = 10,
            ) -> list[tuple[str, float]]:
                return []

            def temporal_search(
                self,
                vector: list[float],
                user_id: str | None = None,
                time_range: tuple[object, object] | None = None,
                recency_weight: float = 0.3,
                limit: int = 10,
            ) -> list[tuple[str, float]]:
                return []

            def find_similar(
                self,
                vector: list[float],
                user_id: str | None = None,
                threshold: float = 0.95,
            ) -> list[Memory]:
                return []

            def record_access(self, memory_id: str) -> None:
                pass

            def save_history(
                self,
                memory_id: str,
                old_content: str | None,
                new_content: str,
                event: str,
                metadata: dict | None = None,
            ) -> None:
                pass

            def get_history(self, memory_id: str) -> list[dict]:
                return []

            def batch_record_access(
                self, memory_ids: list[str]
            ) -> None:
                pass

            def get_stats(
                self, user_id: str | None = None
            ) -> dict[str, object]:
                return {}

            def list_memories(
                self, user_id: str | None = None, limit: int = 10_000
            ) -> list[Memory]:
                return []

            def load_graph_snapshot(self) -> dict | None:
                return None

            def save_graph_snapshot(self, data: dict) -> None:
                pass

        dummy_mod.DummyStore = DummyStore  # type: ignore[attr-defined]
        sys.modules["tests._dummy_store"] = dummy_mod

        try:
            VectorStoreFactory.register(
                "dummy",
                "tests._dummy_store",
                "DummyStore",
                "dummy-extra",
            )
            backend = VectorStoreFactory.create("dummy")
            assert isinstance(backend, DummyStore)
        finally:
            del sys.modules["tests._dummy_store"]

    def test_create_with_missing_dep_raises_import_error(
        self,
    ) -> None:
        from neuralmem.storage.factory import VectorStoreFactory

        # Register a backend whose module doesn't exist
        VectorStoreFactory.register(
            "ghost", "nonexistent.module", "Cls", "ghost"
        )
        with pytest.raises(ImportError, match="pip install"):
            VectorStoreFactory.create("ghost")

    def test_available_backends_returns_list(self) -> None:
        from neuralmem.storage.factory import VectorStoreFactory

        result = VectorStoreFactory.available_backends()
        assert isinstance(result, list)


# ==================================================================
# ChromaBackend Tests (mocked)
# ==================================================================


class TestChromaVectorStore:
    """Tests for ChromaVectorStore using mocked chromadb."""

    @pytest.fixture(autouse=True)
    def _patch_chroma(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provide a mock chromadb module."""
        mock_chroma = MagicMock()
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = (
            mock_collection
        )
        mock_chroma.Client.return_value = mock_client
        mock_chroma.PersistentClient.return_value = mock_client

        monkeypatch.setitem(sys.modules, "chromadb", mock_chroma)

        # Re-import so the module-level try/except picks up the mock
        if "neuralmem.storage.chroma" in sys.modules:
            del sys.modules["neuralmem.storage.chroma"]

    def _make_store(self) -> Any:
        from neuralmem.storage.chroma import ChromaVectorStore

        return ChromaVectorStore(config={})

    def test_save_memory(self) -> None:
        store = self._make_store()
        mem = _make_memory()
        result = store.save_memory(mem)
        assert result == "test-mem-001"
        store._collection.upsert.assert_called_once()

    def test_get_memory_returns_none_when_empty(self) -> None:
        store = self._make_store()
        store._collection.get.return_value = {
            "ids": [],
            "metadatas": [],
            "embeddings": [],
        }
        assert store.get_memory("missing") is None

    def test_get_memory_returns_memory(self) -> None:
        store = self._make_store()
        now = datetime.now(timezone.utc).isoformat()
        store._collection.get.return_value = {
            "ids": ["m1"],
            "metadatas": [
                {
                    "content": "test",
                    "memory_type": "semantic",
                    "scope": "user",
                    "user_id": "u1",
                    "agent_id": "",
                    "session_id": "",
                    "tags": "[]",
                    "source": "",
                    "importance": 0.5,
                    "entity_ids": "[]",
                    "is_active": 1,
                    "superseded_by": "",
                    "supersedes": "[]",
                    "created_at": now,
                    "updated_at": now,
                    "last_accessed": now,
                    "access_count": 0,
                    "expires_at": "",
                }
            ],
            "embeddings": [[0.1] * 384],
        }
        mem = store.get_memory("m1")
        assert mem is not None
        assert mem.id == "m1"
        assert mem.content == "test"

    def test_update_memory_not_found(self) -> None:
        store = self._make_store()
        store._collection.get.return_value = {
            "ids": [],
            "metadatas": [],
            "embeddings": [],
        }
        with pytest.raises(Exception):
            store.update_memory("missing", content="new")

    def test_delete_memories_by_id(self) -> None:
        store = self._make_store()
        count = store.delete_memories(memory_id="m1")
        assert count == 1
        store._collection.delete.assert_called_once()

    def test_delete_memories_by_user(self) -> None:
        store = self._make_store()
        now = datetime.now(timezone.utc).isoformat()
        store._collection.get.return_value = {
            "ids": ["m1", "m2"],
            "metadatas": [
                {"user_id": "u1", "created_at": now},
                {"user_id": "u2", "created_at": now},
            ],
        }
        count = store.delete_memories(user_id="u1")
        assert count == 1

    def test_vector_search_empty(self) -> None:
        store = self._make_store()
        store._collection.query.return_value = {
            "ids": [],
            "distances": [],
        }
        results = store.vector_search([0.1] * 384)
        assert results == []

    def test_vector_search_returns_scores(self) -> None:
        store = self._make_store()
        store._collection.query.return_value = {
            "ids": [["m1", "m2"]],
            "distances": [[0.2, 0.5]],
        }
        results = store.vector_search([0.1] * 384, limit=2)
        assert len(results) == 2
        assert results[0][0] == "m1"
        assert abs(results[0][1] - 0.8) < 1e-6

    def test_keyword_search(self) -> None:
        store = self._make_store()
        store._collection.query.return_value = {
            "ids": [["m1"]],
            "distances": [[0.3]],
        }
        results = store.keyword_search("hello")
        assert len(results) == 1

    def test_record_access(self) -> None:
        store = self._make_store()
        now = datetime.now(timezone.utc).isoformat()
        store._collection.get.return_value = {
            "ids": ["m1"],
            "metadatas": [
                {
                    "content": "test",
                    "access_count": 0,
                    "last_accessed": now,
                }
            ],
        }
        store.record_access("m1")
        store._collection.update.assert_called_once()

    def test_record_access_missing(self) -> None:
        store = self._make_store()
        store._collection.get.return_value = {
            "ids": [],
            "metadatas": [],
        }
        store.record_access("missing")
        store._collection.update.assert_not_called()

    def test_list_memories(self) -> None:
        store = self._make_store()
        now = datetime.now(timezone.utc).isoformat()
        store._collection.get.return_value = {
            "ids": ["m1"],
            "metadatas": [
                {
                    "content": "test",
                    "memory_type": "semantic",
                    "scope": "user",
                    "user_id": "u1",
                    "agent_id": "",
                    "session_id": "",
                    "tags": "[]",
                    "source": "",
                    "importance": 0.5,
                    "entity_ids": "[]",
                    "is_active": 1,
                    "superseded_by": "",
                    "supersedes": "[]",
                    "created_at": now,
                    "updated_at": now,
                    "last_accessed": now,
                    "access_count": 0,
                    "expires_at": "",
                }
            ],
            "embeddings": [None],
        }
        mems = store.list_memories(user_id="u1")
        assert len(mems) == 1

    def test_get_stats(self) -> None:
        store = self._make_store()
        store._collection.get.return_value = {
            "ids": ["m1", "m2"],
            "metadatas": [{}, {}],
        }
        stats = store.get_stats()
        assert stats["backend"] == "chroma"
        assert stats["total_memories"] == 2

    def test_save_and_get_history(self) -> None:
        store = self._make_store()
        store.save_history("m1", "old", "new", "UPDATE")
        # No-op returns None; get_history returns []
        assert store.get_history("m1") == []

    def test_graph_snapshot_noop(self) -> None:
        store = self._make_store()
        store.save_graph_snapshot({"key": "val"})
        assert store.load_graph_snapshot() is None

    def test_find_similar(self) -> None:
        store = self._make_store()
        store._collection.query.return_value = {
            "ids": [["m1"]],
            "distances": [[0.02]],
        }
        now = datetime.now(timezone.utc).isoformat()
        store._collection.get.return_value = {
            "ids": ["m1"],
            "metadatas": [
                {
                    "content": "test",
                    "memory_type": "semantic",
                    "scope": "user",
                    "user_id": "u1",
                    "agent_id": "",
                    "session_id": "",
                    "tags": "[]",
                    "source": "",
                    "importance": 0.5,
                    "entity_ids": "[]",
                    "is_active": 1,
                    "superseded_by": "",
                    "supersedes": "[]",
                    "created_at": now,
                    "updated_at": now,
                    "last_accessed": now,
                    "access_count": 0,
                    "expires_at": "",
                }
            ],
            "embeddings": [[0.1] * 384],
        }
        similar = store.find_similar(
            [0.1] * 384, threshold=0.9
        )
        assert len(similar) == 1

    def test_batch_record_access(self) -> None:
        store = self._make_store()
        now = datetime.now(timezone.utc).isoformat()
        store._collection.get.return_value = {
            "ids": ["m1"],
            "metadatas": [
                {
                    "content": "c",
                    "access_count": 0,
                    "last_accessed": now,
                }
            ],
        }
        store.batch_record_access(["m1"])

    def test_temporal_search(self) -> None:
        store = self._make_store()
        store._collection.query.return_value = {
            "ids": [["m1"]],
            "distances": [[0.1]],
        }
        mem = _make_memory(id="m1")
        store.list_memories = MagicMock(  # type: ignore[method-assign]
            return_value=[mem]
        )
        store.get_memory = MagicMock(  # type: ignore[method-assign]
            return_value=mem
        )
        result = store.temporal_search([0.1] * 384, limit=5)
        assert len(result) == 1

    def test_import_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure helpful error when chromadb is missing."""
        monkeypatch.setitem(sys.modules, "chromadb", None)
        # Force re-import
        if "neuralmem.storage.chroma" in sys.modules:
            del sys.modules["neuralmem.storage.chroma"]
        with pytest.raises(ImportError, match="pip install"):
            from neuralmem.storage.chroma import ChromaVectorStore

            ChromaVectorStore(config={})


# ==================================================================
# FAISSBackend Tests (mocked)
# ==================================================================


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore using mocked faiss."""

    @pytest.fixture(autouse=True)
    def _patch_faiss(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_faiss.IndexFlatIP.return_value = mock_index
        monkeypatch.setitem(sys.modules, "faiss", mock_faiss)
        if "neuralmem.storage.faiss_store" in sys.modules:
            del sys.modules["neuralmem.storage.faiss_store"]

    def _make_store(self) -> Any:
        from neuralmem.storage.faiss_store import FAISSVectorStore

        return FAISSVectorStore(config={})

    def test_save_memory(self) -> None:
        store = self._make_store()
        mem = _make_memory()
        result = store.save_memory(mem)
        assert result == "test-mem-001"
        assert "test-mem-001" in store._memories

    def test_get_memory(self) -> None:
        store = self._make_store()
        mem = _make_memory()
        store.save_memory(mem)
        retrieved = store.get_memory("test-mem-001")
        assert retrieved is not None
        assert retrieved.content == "hello world"

    def test_get_memory_missing(self) -> None:
        store = self._make_store()
        assert store.get_memory("nope") is None

    def test_update_memory(self) -> None:
        store = self._make_store()
        mem = _make_memory()
        store.save_memory(mem)
        store.update_memory("test-mem-001", importance=0.9)
        updated = store.get_memory("test-mem-001")
        assert updated is not None
        assert updated.importance == 0.9

    def test_update_memory_not_found(self) -> None:
        store = self._make_store()
        with pytest.raises(Exception):
            store.update_memory("nope", content="x")

    def test_delete_memories_by_id(self) -> None:
        store = self._make_store()
        store.save_memory(_make_memory(id="m1"))
        count = store.delete_memories(memory_id="m1")
        assert count == 1
        assert store.get_memory("m1") is None

    def test_delete_memories_by_user(self) -> None:
        store = self._make_store()
        store.save_memory(_make_memory(id="m1", user_id="u1"))
        store.save_memory(_make_memory(id="m2", user_id="u2"))
        count = store.delete_memories(user_id="u1")
        assert count == 1

    def test_list_memories(self) -> None:
        store = self._make_store()
        store.save_memory(_make_memory(id="m1", user_id="u1"))
        store.save_memory(_make_memory(id="m2", user_id="u2"))
        all_mems = store.list_memories()
        assert len(all_mems) == 2
        u1 = store.list_memories(user_id="u1")
        assert len(u1) == 1

    def test_vector_search_empty_index(self) -> None:
        store = self._make_store()
        results = store.vector_search([0.1] * 384)
        assert results == []

    def test_keyword_search(self) -> None:
        store = self._make_store()
        store.save_memory(_make_memory(id="m1", content="foo bar"))
        store.save_memory(_make_memory(id="m2", content="baz qux"))
        results = store.keyword_search("foo")
        assert len(results) == 1
        assert results[0][0] == "m1"

    def test_record_access(self) -> None:
        store = self._make_store()
        store.save_memory(_make_memory(id="m1"))
        store.record_access("m1")
        mem = store.get_memory("m1")
        assert mem is not None
        assert mem.access_count == 1

    def test_batch_record_access(self) -> None:
        store = self._make_store()
        store.save_memory(_make_memory(id="m1"))
        store.save_memory(_make_memory(id="m2"))
        store.batch_record_access(["m1", "m2"])
        assert store.get_memory("m1").access_count == 1  # type: ignore[union-attr]
        assert store.get_memory("m2").access_count == 1  # type: ignore[union-attr]

    def test_save_and_get_history(self) -> None:
        store = self._make_store()
        store.save_history("m1", "old", "new", "UPDATE")
        history = store.get_history("m1")
        assert len(history) == 1
        assert history[0]["event"] == "UPDATE"

    def test_get_stats(self) -> None:
        store = self._make_store()
        store.save_memory(_make_memory(id="m1"))
        stats = store.get_stats()
        assert stats["backend"] == "faiss"
        assert stats["total_memories"] == 1

    def test_graph_snapshot(self) -> None:
        store = self._make_store()
        store.save_graph_snapshot({"key": "val"})
        assert store.load_graph_snapshot() == {"key": "val"}

    def test_find_similar(self) -> None:
        store = self._make_store()
        store.save_memory(
            _make_memory(id="m1", embedding=[1.0] * 384)
        )
        store._index.ntotal = 1
        store._index.search.return_value = (
            [[0.99]],
            [[0]],
        )
        result = store.find_similar(
            [1.0] * 384, threshold=0.9
        )
        assert len(result) == 1

    def test_temporal_search(self) -> None:
        store = self._make_store()
        mem = _make_memory(id="m1")
        store.save_memory(mem)
        store._index.ntotal = 1
        store._index.search.return_value = (
            [[0.95]],
            [[0]],
        )
        result = store.temporal_search([0.1] * 384, limit=5)
        assert len(result) == 1

    def test_update_memory_tags(self) -> None:
        store = self._make_store()
        store.save_memory(_make_memory(id="m1"))
        store.update_memory("m1", tags=["new_tag"])
        mem = store.get_memory("m1")
        assert "new_tag" in mem.tags  # type: ignore[union-attr]

    def test_import_error_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "faiss", None)
        if "neuralmem.storage.faiss_store" in sys.modules:
            del sys.modules["neuralmem.storage.faiss_store"]
        with pytest.raises(ImportError, match="pip install"):
            from neuralmem.storage.faiss_store import (
                FAISSVectorStore,
            )

            FAISSVectorStore(config={})


# ==================================================================
# QdrantBackend Tests (mocked)
# ==================================================================


class TestQdrantVectorStore:
    """Tests for QdrantVectorStore using mocked qdrant-client."""

    @pytest.fixture(autouse=True)
    def _patch_qdrant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_qdrant = MagicMock()
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(
            collections=[]
        )
        mock_qdrant.QdrantClient.return_value = mock_client

        # Provide models module with expected classes
        models = types.ModuleType("qdrant_client.models")
        models.Distance = MagicMock()
        models.VectorParams = MagicMock()
        models.PointStruct = MagicMock(
            side_effect=lambda id, vector, payload: MagicMock(
                id=id, vector=vector, payload=payload
            )
        )
        models.PointIdsList = MagicMock()
        models.Filter = MagicMock(side_effect=lambda must: {"must": must})
        models.FieldCondition = MagicMock()
        models.MatchValue = MagicMock(side_effect=lambda value: value)

        mock_qdrant.models = models
        monkeypatch.setitem(sys.modules, "qdrant_client", mock_qdrant)
        monkeypatch.setitem(
            sys.modules, "qdrant_client.models", models
        )

        if "neuralmem.storage.qdrant_store" in sys.modules:
            del sys.modules["neuralmem.storage.qdrant_store"]

    def _make_store(self) -> Any:
        from neuralmem.storage.qdrant_store import (
            QdrantVectorStore,
        )

        return QdrantVectorStore(config={":memory:": True})

    def test_save_memory(self) -> None:
        store = self._make_store()
        mem = _make_memory()
        result = store.save_memory(mem)
        assert result == "test-mem-001"

    def test_get_memory_not_found(self) -> None:
        store = self._make_store()
        store._client.retrieve.return_value = []
        assert store.get_memory("missing") is None

    def test_update_memory_not_found(self) -> None:
        store = self._make_store()
        store._client.retrieve.return_value = []
        with pytest.raises(Exception):
            store.update_memory("missing", content="x")

    def test_delete_memories_by_id(self) -> None:
        store = self._make_store()
        count = store.delete_memories(memory_id="m1")
        assert count == 1

    def test_vector_search_empty(self) -> None:
        store = self._make_store()
        store._client.query_points.return_value = MagicMock(
            points=[]
        )
        result = store.vector_search([0.1] * 384)
        assert result == []

    def test_keyword_search_empty(self) -> None:
        store = self._make_store()
        store._client.scroll.return_value = ([], None)
        result = store.keyword_search("hello")
        assert result == []

    def test_record_access(self) -> None:
        store = self._make_store()
        store._client.retrieve.return_value = []
        store.record_access("m1")
        store._client.upsert.assert_not_called()

    def test_get_stats(self) -> None:
        store = self._make_store()
        store._client.get_collection.return_value = MagicMock(
            points_count=5
        )
        stats = store.get_stats()
        assert stats["backend"] == "qdrant"
        assert stats["total_memories"] == 5

    def test_list_memories(self) -> None:
        store = self._make_store()
        store._client.scroll.return_value = ([], None)
        result = store.list_memories()
        assert result == []

    def test_batch_record_access(self) -> None:
        store = self._make_store()
        store._client.retrieve.return_value = []
        store.batch_record_access(["m1"])

    def test_import_error_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "qdrant_client", None)
        if "neuralmem.storage.qdrant_store" in sys.modules:
            del sys.modules["neuralmem.storage.qdrant_store"]
        with pytest.raises(ImportError, match="pip install"):
            from neuralmem.storage.qdrant_store import (
                QdrantVectorStore,
            )

            QdrantVectorStore(config={})
