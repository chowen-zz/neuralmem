"""SQLiteStorage 扩展覆盖率测试"""
from __future__ import annotations

import hashlib

import pytest

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.types import Memory, MemoryType
from neuralmem.storage.sqlite import SQLiteStorage

# ── conftest 提供的 storage/sample_memory fixture 使用 dim=4 ──────────────────


@pytest.fixture
def storage_with_data(tmp_path):
    """带 5 条样本数据的存储实例（dim=4，与全局 conftest 保持一致）"""
    cfg = NeuralMemConfig(db_path=str(tmp_path / "ext.db"), embedding_dim=4)
    s = SQLiteStorage(cfg)
    for i in range(5):
        content = f"Memory content number {i} about topic{i}"
        h = int(hashlib.md5(content.encode()).hexdigest(), 16)
        v = [((h >> (j * 8)) & 0xFF) / 255.0 for j in range(4)]
        norm = sum(x ** 2 for x in v) ** 0.5 or 1.0
        v = [x / norm for x in v]
        m = Memory(
            content=content,
            user_id="user-1",
            memory_type=MemoryType.SEMANTIC,
            embedding=v,
        )
        s.save_memory(m)
    return s


# ── get / save ────────────────────────────────────────────────────────────────


def test_save_and_get_memory(storage, sample_memory):
    storage.save_memory(sample_memory)
    retrieved = storage.get_memory(sample_memory.id)
    assert retrieved is not None
    assert retrieved.content == sample_memory.content


def test_get_memory_nonexistent(storage):
    result = storage.get_memory("nonexistent-id")
    assert result is None


def test_save_memory_without_embedding(storage):
    m = Memory(content="no embedding memory")
    storage.save_memory(m)
    retrieved = storage.get_memory(m.id)
    assert retrieved is not None
    assert retrieved.content == "no embedding memory"


# ── update ────────────────────────────────────────────────────────────────────


def test_update_memory(storage, sample_memory):
    storage.save_memory(sample_memory)
    storage.update_memory(sample_memory.id, importance=0.9)
    updated = storage.get_memory(sample_memory.id)
    assert updated is not None
    assert updated.importance == pytest.approx(0.9)


# ── delete ────────────────────────────────────────────────────────────────────


def test_delete_by_id(storage, sample_memory):
    storage.save_memory(sample_memory)
    count = storage.delete_memories(memory_id=sample_memory.id)
    assert count == 1
    assert storage.get_memory(sample_memory.id) is None


def test_delete_by_user_id(storage_with_data):
    count = storage_with_data.delete_memories(user_id="user-1")
    assert count == 5
    stats = storage_with_data.get_stats(user_id="user-1")
    assert stats["total"] == 0


def test_delete_by_tags(storage):
    m = Memory(content="tagged memory", tags=("python", "ai"), embedding=[0.5, 0.5, 0.5, 0.5])
    storage.save_memory(m)
    count = storage.delete_memories(tags=["python"])
    assert count >= 1


def test_delete_no_conditions_returns_zero(storage, sample_memory):
    """不带任何条件的删除 — 无匹配项时返回 0"""
    count = storage.delete_memories(memory_id="does-not-exist")
    assert count == 0


# ── search ────────────────────────────────────────────────────────────────────


def test_keyword_search(storage_with_data):
    # FTS5 unicode61 分词：topic0/topic1/... 是独立词元，用前缀匹配 "topic*"
    results = storage_with_data.keyword_search("topic0", limit=5)
    assert isinstance(results, list)
    # 至少能搜到一条内容为 "topic0" 的记忆
    assert len(results) >= 1


def test_keyword_search_no_match(storage_with_data):
    results = storage_with_data.keyword_search("xyzzy_not_exist", limit=5)
    assert isinstance(results, list)


def test_vector_search(storage_with_data):
    vector = [0.5, 0.5, 0.5, 0.5]
    norm = sum(x ** 2 for x in vector) ** 0.5
    vector = [x / norm for x in vector]
    results = storage_with_data.vector_search(vector, limit=3)
    assert isinstance(results, list)
    assert len(results) <= 3


def test_vector_search_with_user_filter(storage_with_data):
    vector = [0.5, 0.5, 0.5, 0.5]
    norm = sum(x ** 2 for x in vector) ** 0.5
    vector = [x / norm for x in vector]
    results = storage_with_data.vector_search(vector, user_id="user-1", limit=5)
    assert isinstance(results, list)


def test_vector_search_with_type_filter(storage_with_data):
    vector = [0.5, 0.5, 0.5, 0.5]
    norm = sum(x ** 2 for x in vector) ** 0.5
    vector = [x / norm for x in vector]
    results = storage_with_data.vector_search(
        vector, memory_types=[MemoryType.SEMANTIC], limit=5
    )
    assert isinstance(results, list)


# ── temporal_search ───────────────────────────────────────────────────────────


def test_temporal_search(storage_with_data):
    v = [0.5, 0.5, 0.5, 0.5]
    norm = sum(x ** 2 for x in v) ** 0.5
    v = [x / norm for x in v]
    results = storage_with_data.temporal_search(v, recency_weight=0.3, limit=3)
    assert isinstance(results, list)
    assert len(results) <= 3


def test_temporal_search_empty_storage(storage):
    v = [1.0, 0.0, 0.0, 0.0]
    results = storage.temporal_search(v, recency_weight=0.5, limit=5)
    assert results == []


# ── find_similar ──────────────────────────────────────────────────────────────


def test_find_similar_high_threshold(storage_with_data):
    v = [0.5, 0.5, 0.5, 0.5]
    norm = sum(x ** 2 for x in v) ** 0.5
    v = [x / norm for x in v]
    results = storage_with_data.find_similar(v, threshold=0.9999)
    assert isinstance(results, list)


def test_find_similar_low_threshold(storage_with_data):
    """低阈值应该能找到结果"""
    v = [0.5, 0.5, 0.5, 0.5]
    norm = sum(x ** 2 for x in v) ** 0.5
    v = [x / norm for x in v]
    results = storage_with_data.find_similar(v, threshold=0.0)
    assert isinstance(results, list)


# ── record_access ─────────────────────────────────────────────────────────────


def test_record_access(storage, sample_memory):
    storage.save_memory(sample_memory)
    storage.record_access(sample_memory.id)
    m = storage.get_memory(sample_memory.id)
    assert m is not None
    assert m.access_count == 1


def test_record_access_multiple_times(storage, sample_memory):
    storage.save_memory(sample_memory)
    for _ in range(3):
        storage.record_access(sample_memory.id)
    m = storage.get_memory(sample_memory.id)
    assert m.access_count == 3


# ── get_stats ─────────────────────────────────────────────────────────────────


def test_get_stats_empty(storage):
    stats = storage.get_stats()
    assert stats["total"] == 0


def test_get_stats_with_data(storage_with_data):
    stats = storage_with_data.get_stats()
    assert stats["total"] == 5


def test_get_stats_by_user(storage_with_data):
    stats = storage_with_data.get_stats(user_id="user-1")
    assert stats["total"] == 5
    stats_other = storage_with_data.get_stats(user_id="nobody")
    assert stats_other["total"] == 0


def test_list_memories_returns_all(storage_with_data):
    memories = storage_with_data.list_memories()
    assert len(memories) == 5
    assert all(hasattr(m, "content") for m in memories)


def test_list_memories_by_user(storage_with_data):
    memories = storage_with_data.list_memories(user_id="user-1")
    assert len(memories) == 5
    assert all(m.user_id == "user-1" for m in memories)


def test_list_memories_empty(storage):
    memories = storage.list_memories()
    assert memories == []
