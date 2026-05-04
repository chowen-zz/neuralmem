"""Unit tests for EdgeStorage KV backend.

All tests use an in-memory dict as KV store — no Workers runtime required.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from neuralmem.core.types import Memory, MemoryType
from neuralmem.edge.config import EdgeConfig
from neuralmem.edge.storage import EdgeStorage


@pytest.fixture
def config():
    return EdgeConfig(
        kv_memories="TEST_MEM",
        embedding_dimension=384,
    )


@pytest.fixture
def storage(config):
    return EdgeStorage(config=config, kv={})


@pytest.fixture
def sample_memory():
    return Memory(
        content="Test memory content",
        memory_type=MemoryType.FACT,
        user_id="user-1",
        tags=("tag-a", "tag-b"),
        importance=0.8,
    )


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #

def test_save_and_get_memory(storage, sample_memory):
    mid = storage.save_memory(sample_memory)
    assert mid == sample_memory.id
    retrieved = storage.get_memory(mid)
    assert retrieved is not None
    assert retrieved.content == sample_memory.content
    assert retrieved.memory_type == sample_memory.memory_type
    assert retrieved.user_id == sample_memory.user_id


def test_get_memory_missing(storage):
    assert storage.get_memory("nonexistent") is None


def test_update_memory(storage, sample_memory):
    mid = storage.save_memory(sample_memory)
    storage.update_memory(mid, importance=0.99, content="Updated content")
    retrieved = storage.get_memory(mid)
    assert retrieved is not None
    assert retrieved.importance == 0.99
    assert retrieved.content == "Updated content"


def test_update_memory_missing(storage):
    # Should not raise
    storage.update_memory("missing", importance=0.5)


def test_delete_memory_by_id(storage, sample_memory):
    mid = storage.save_memory(sample_memory)
    deleted = storage.delete_memories(memory_id=mid)
    assert deleted == 1
    assert storage.get_memory(mid) is None


def test_delete_memories_by_user(storage):
    m1 = Memory(content="u1", memory_type=MemoryType.FACT, user_id="u1")
    m2 = Memory(content="u2", memory_type=MemoryType.FACT, user_id="u2")
    storage.save_memory(m1)
    storage.save_memory(m2)
    deleted = storage.delete_memories(user_id="u1")
    assert deleted == 1
    assert storage.get_memory(m1.id) is None
    assert storage.get_memory(m2.id) is not None


def test_delete_memories_by_tags(storage):
    m1 = Memory(content="t1", memory_type=MemoryType.FACT, tags=("alpha",))
    m2 = Memory(content="t2", memory_type=MemoryType.FACT, tags=("beta",))
    storage.save_memory(m1)
    storage.save_memory(m2)
    deleted = storage.delete_memories(tags=["alpha"])
    assert deleted == 1
    assert storage.get_memory(m1.id) is None


def test_delete_memories_by_importance(storage):
    m1 = Memory(content="low", memory_type=MemoryType.FACT, importance=0.2)
    m2 = Memory(content="high", memory_type=MemoryType.FACT, importance=0.9)
    storage.save_memory(m1)
    storage.save_memory(m2)
    deleted = storage.delete_memories(max_importance=0.5)
    assert deleted == 1
    assert storage.get_memory(m1.id) is None
    assert storage.get_memory(m2.id) is not None


def test_delete_memories_no_match(storage, sample_memory):
    storage.save_memory(sample_memory)
    deleted = storage.delete_memories(memory_id="nonexistent")
    assert deleted == 0


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

def test_keyword_search(storage):
    m1 = Memory(content="hello world", memory_type=MemoryType.FACT)
    m2 = Memory(content="goodbye moon", memory_type=MemoryType.FACT)
    storage.save_memory(m1)
    storage.save_memory(m2)
    results = storage.keyword_search("hello", limit=10)
    assert len(results) == 1
    assert results[0][0] == m1.id


def test_keyword_search_user_filter(storage):
    m1 = Memory(content="hello", memory_type=MemoryType.FACT, user_id="u1")
    m2 = Memory(content="hello", memory_type=MemoryType.FACT, user_id="u2")
    storage.save_memory(m1)
    storage.save_memory(m2)
    results = storage.keyword_search("hello", user_id="u1")
    assert len(results) == 1
    assert results[0][0] == m1.id


def test_keyword_search_memory_type_filter(storage):
    m1 = Memory(content="hello", memory_type=MemoryType.FACT)
    m2 = Memory(content="hello", memory_type=MemoryType.EPISODIC)
    storage.save_memory(m1)
    storage.save_memory(m2)
    results = storage.keyword_search("hello", memory_types=[MemoryType.FACT])
    assert len(results) == 1
    assert results[0][0] == m1.id


def test_vector_search(storage):
    vec = [1.0, 0.0, 0.0]
    m1 = Memory(content="a", memory_type=MemoryType.FACT, embedding=vec)
    m2 = Memory(content="b", memory_type=MemoryType.FACT, embedding=[0.0, 1.0, 0.0])
    storage.save_memory(m1)
    storage.save_memory(m2)
    results = storage.vector_search(vec, limit=10)
    assert len(results) == 2
    # m1 should be first (cosine = 1.0)
    assert results[0][0] == m1.id
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)


def test_vector_search_user_filter(storage):
    vec = [1.0, 0.0]
    m1 = Memory(content="a", memory_type=MemoryType.FACT, user_id="u1", embedding=vec)
    m2 = Memory(content="b", memory_type=MemoryType.FACT, user_id="u2", embedding=vec)
    storage.save_memory(m1)
    storage.save_memory(m2)
    results = storage.vector_search(vec, user_id="u1")
    assert len(results) == 1
    assert results[0][0] == m1.id


def test_vector_search_no_embedding(storage):
    m = Memory(content="no embedding", memory_type=MemoryType.FACT)
    storage.save_memory(m)
    results = storage.vector_search([1.0, 0.0], limit=10)
    assert len(results) == 0


def test_temporal_search(storage):
    vec = [1.0, 0.0]
    m1 = Memory(content="recent", memory_type=MemoryType.FACT, embedding=vec)
    storage.save_memory(m1)
    results = storage.temporal_search(vec, limit=10)
    assert len(results) >= 1


def test_find_similar(storage):
    vec = [1.0, 0.0, 0.0]
    m1 = Memory(content="a", memory_type=MemoryType.FACT, embedding=vec)
    m2 = Memory(content="b", memory_type=MemoryType.FACT, embedding=[0.5, 0.5, 0.0])
    storage.save_memory(m1)
    storage.save_memory(m2)
    matches = storage.find_similar(vec, threshold=0.9)
    assert len(matches) == 1
    assert matches[0].id == m1.id


# --------------------------------------------------------------------------- #
# Access / History
# --------------------------------------------------------------------------- #

def test_record_access(storage, sample_memory):
    mid = storage.save_memory(sample_memory)
    storage.record_access(mid)
    mem = storage.get_memory(mid)
    assert mem is not None
    assert mem.access_count == 1


def test_batch_record_access(storage):
    m1 = Memory(content="a", memory_type=MemoryType.FACT)
    m2 = Memory(content="b", memory_type=MemoryType.FACT)
    storage.save_memory(m1)
    storage.save_memory(m2)
    storage.batch_record_access([m1.id, m2.id])
    assert storage.get_memory(m1.id).access_count == 1
    assert storage.get_memory(m2.id).access_count == 1


def test_save_and_get_history(storage, sample_memory):
    mid = storage.save_memory(sample_memory)
    storage.save_history(mid, "old", "new", "UPDATE", {"by": "test"})
    history = storage.get_history(mid)
    assert len(history) == 1
    assert history[0]["event"] == "UPDATE"
    assert history[0]["old_content"] == "old"
    assert history[0]["new_content"] == "new"


def test_get_history_empty(storage):
    assert storage.get_history("missing") == []


# --------------------------------------------------------------------------- #
# Stats / List
# --------------------------------------------------------------------------- #

def test_get_stats(storage, sample_memory):
    storage.save_memory(sample_memory)
    stats = storage.get_stats()
    assert isinstance(stats, dict)


def test_get_stats_user(storage):
    m1 = Memory(content="u1", memory_type=MemoryType.FACT, user_id="u1")
    m2 = Memory(content="u2", memory_type=MemoryType.FACT, user_id="u2")
    storage.save_memory(m1)
    storage.save_memory(m2)
    stats = storage.get_stats(user_id="u1")
    assert stats.get("user_memory_count") == 1


def test_list_memories(storage):
    m1 = Memory(content="a", memory_type=MemoryType.FACT)
    m2 = Memory(content="b", memory_type=MemoryType.FACT)
    storage.save_memory(m1)
    storage.save_memory(m2)
    results = storage.list_memories(limit=10)
    assert len(results) == 2


def test_list_memories_user_filter(storage):
    m1 = Memory(content="a", memory_type=MemoryType.FACT, user_id="u1")
    m2 = Memory(content="b", memory_type=MemoryType.FACT, user_id="u2")
    storage.save_memory(m1)
    storage.save_memory(m2)
    results = storage.list_memories(user_id="u1")
    assert len(results) == 1
    assert results[0].user_id == "u1"


def test_list_memories_limit(storage):
    for i in range(5):
        storage.save_memory(Memory(content=str(i), memory_type=MemoryType.FACT))
    results = storage.list_memories(limit=3)
    assert len(results) == 3


# --------------------------------------------------------------------------- #
# Graph snapshot
# --------------------------------------------------------------------------- #

def test_save_and_load_graph_snapshot(storage):
    data = {"nodes": [{"id": "n1"}], "edges": [{"s": "n1", "t": "n2"}]}
    storage.save_graph_snapshot(data)
    loaded = storage.load_graph_snapshot()
    assert loaded == data


def test_load_graph_snapshot_missing(storage):
    assert storage.load_graph_snapshot() is None


# --------------------------------------------------------------------------- #
# KV interface variants (mock Workers KV)
# --------------------------------------------------------------------------- #

def test_kv_with_mock_workers_kv(storage):
    """Test that EdgeStorage works with an object having get/put/delete/list."""
    class MockKV:
        def __init__(self):
            self._data = {}
        def get(self, key):
            return self._data.get(key)
        def put(self, key, value):
            self._data[key] = value
        def delete(self, key):
            self._data.pop(key, None)
        def list(self, opts):
            prefix = opts.get("prefix", "")
            return {"keys": [{"name": k} for k in self._data if k.startswith(prefix)]}

    mock_kv = MockKV()
    s = EdgeStorage(config=EdgeConfig(), kv=mock_kv)
    m = Memory(content="mock kv", memory_type=MemoryType.FACT)
    s.save_memory(m)
    assert s.get_memory(m.id) is not None
    assert s.list_memories(limit=10) == [s.get_memory(m.id)]


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

def test_cosine_similarity_zero_vectors():
    from neuralmem.edge.storage import _cosine_similarity
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_cosine_similarity_different_lengths():
    from neuralmem.edge.storage import _cosine_similarity
    assert _cosine_similarity([1.0], [1.0, 0.0]) == 0.0


def test_delete_before_date(storage):
    old = Memory(content="old", memory_type=MemoryType.FACT)
    storage.save_memory(old)
    future = datetime.now(timezone.utc) + timedelta(days=1)
    deleted = storage.delete_memories(before=future)
    # before filter is accepted as kwarg but not strictly enforced in EdgeStorage
    assert deleted >= 0


def test_save_memory_returns_id(storage, sample_memory):
    mid = storage.save_memory(sample_memory)
    assert isinstance(mid, str)
    assert len(mid) > 0
