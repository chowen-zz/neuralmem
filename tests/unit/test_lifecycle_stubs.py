"""生命周期测试 — ImportanceScorer + MemoryConsolidator"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from neuralmem.core.protocols import LifecycleProtocol
from neuralmem.core.types import Memory, MemoryType
from neuralmem.lifecycle.consolidation import MemoryConsolidator, _cosine_similarity
from neuralmem.lifecycle.decay import DecayManager
from neuralmem.lifecycle.importance import (
    ImportanceScorer,
    _access_factor,
    _entity_factor,
    _recency_factor,
    _type_factor,
)

pytestmark = pytest.mark.stub


def _mock_storage():
    s = MagicMock()
    s.list_memories.return_value = []
    s.delete_memories.return_value = 0
    return s


# ==================== DecayManager ====================

def test_decay_manager_protocol():
    dm = DecayManager(_mock_storage())
    assert isinstance(dm, LifecycleProtocol)


def test_decay_apply_returns_zero():
    assert DecayManager(_mock_storage()).apply_decay() == 0
    assert DecayManager(_mock_storage()).apply_decay(user_id="test") == 0


def test_decay_remove_returns_zero():
    assert DecayManager(_mock_storage()).remove_forgotten() == 0


# ==================== Consolidator — stub fallback ====================

def test_consolidator_no_deps_returns_zero():
    """无 storage/embedder → 退化为 stub，返回 0"""
    assert MemoryConsolidator().merge_similar() == 0


def test_consolidator_empty_storage_returns_zero():
    """有 storage 但无记忆 → 返回 0"""
    s = _mock_storage()
    c = MemoryConsolidator(storage=s, embedder=MagicMock())
    assert c.merge_similar() == 0


# ==================== Consolidator — real merge ====================

def _make_memory(content: str, importance: float = 0.5, created_minutes_ago: int = 0) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        content=content,
        importance=importance,
        created_at=now - timedelta(minutes=created_minutes_ago),
        last_accessed=now,
    )


def test_consolidator_merges_similar():
    """两条高相似记忆应被合并（标记为 superseded，而非删除）"""
    m1 = _make_memory("用户喜欢 Python", importance=0.6, created_minutes_ago=10)
    m2 = _make_memory("用户偏好 Python 编程", importance=0.5, created_minutes_ago=5)

    # 高相似度的 embeddings
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.98, 0.0, 0.02]

    storage = MagicMock()
    storage.list_memories.return_value = [m1, m2]

    embedder = MagicMock()
    embedder.encode_one.side_effect = lambda text: vec1 if "喜欢" in text else vec2

    c = MemoryConsolidator(storage=storage, embedder=embedder, merge_threshold=0.9)
    result = c.merge_similar()
    assert result == 1
    # Should call update_memory twice: once for canonical, once to supersede m2
    assert storage.update_memory.call_count == 2
    # First call updates canonical (m1), second marks m2 as superseded
    calls = storage.update_memory.call_args_list
    # Second call should mark m2 as superseded
    assert calls[1][0][0] == m2.id
    assert calls[1][1]["is_active"] is False
    assert calls[1][1]["superseded_by"] == m1.id
    # Should NOT delete anything
    storage.delete_memories.assert_not_called()


def test_consolidator_skips_dissimilar():
    """两条不同记忆不应被合并"""
    m1 = _make_memory("天气很好")
    m2 = _make_memory("用户喜欢编程")

    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]

    storage = MagicMock()
    storage.list_memories.return_value = [m1, m2]

    embedder = MagicMock()
    embedder.encode_one.side_effect = lambda text: vec1 if "天气" in text else vec2

    c = MemoryConsolidator(storage=storage, embedder=embedder, merge_threshold=0.85)
    result = c.merge_similar()
    assert result == 0
    storage.update_memory.assert_not_called()


def test_consolidator_uses_existing_embedding():
    """如果记忆已有 embedding，应直接使用而不重新计算"""
    vec = [1.0, 0.0, 0.0]
    m1 = _make_memory("test A")
    m1 = m1.model_copy(update={"embedding": vec})
    m2 = _make_memory("test B")
    m2 = m2.model_copy(update={"embedding": vec})

    storage = MagicMock()
    storage.list_memories.return_value = [m1, m2]

    embedder = MagicMock()

    c = MemoryConsolidator(storage=storage, embedder=embedder, merge_threshold=0.85)
    result = c.merge_similar()
    # Should NOT call encode_one since embeddings exist
    embedder.encode_one.assert_not_called()
    assert result == 1
    # m2 should be superseded, not deleted
    calls = storage.update_memory.call_args_list
    assert calls[1][0][0] == m2.id
    assert calls[1][1]["is_active"] is False
    assert calls[1][1]["superseded_by"] == m1.id


def test_cosine_similarity_basic():
    assert _cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert _cosine_similarity([0, 0], [1, 0]) == pytest.approx(0.0)


# ==================== ImportanceScorer ====================

def test_scorer_basic_memory():
    """默认记忆（importance=0.5, 无访问, 无实体）应返回合理分数"""
    scorer = ImportanceScorer()
    m = Memory(content="test", importance=0.5)
    score = scorer.score(m)
    assert 0.0 <= score <= 1.0


def test_scorer_high_access_higher_score():
    """高频访问的记忆应比低频的得分高"""
    scorer = ImportanceScorer()
    m_low = Memory(content="test", importance=0.5, access_count=0)
    m_high = Memory(content="test", importance=0.5, access_count=50)
    assert scorer.score(m_high) > scorer.score(m_low)


def test_scorer_more_entities_higher_score():
    """关联更多实体的记忆应得分更高"""
    scorer = ImportanceScorer()
    m_few = Memory(content="test", importance=0.5, entity_ids=("a",))
    m_many = Memory(content="test", importance=0.5, entity_ids=("a", "b", "c", "d", "e"))
    assert scorer.score(m_many) > scorer.score(m_few)


def test_scorer_recent_access_higher_score():
    """最近访问的记忆应比久远的得分高"""
    scorer = ImportanceScorer()
    now = datetime.now(timezone.utc)
    m_recent = Memory(content="test", importance=0.5, last_accessed=now)
    m_old = Memory(content="test", importance=0.5, last_accessed=now - timedelta(days=90))
    assert scorer.score(m_recent) > scorer.score(m_old)


def test_scorer_semantic_beats_working():
    """语义记忆应比工作记忆得分高"""
    scorer = ImportanceScorer()
    base = dict(content="test", importance=0.5, access_count=0, entity_ids=())
    m_semantic = Memory(memory_type=MemoryType.SEMANTIC, **base)
    m_working = Memory(memory_type=MemoryType.WORKING, **base)
    assert scorer.score(m_semantic) > scorer.score(m_working)


def test_scorer_boundary_values():
    scorer = ImportanceScorer()
    # 最低配置
    m_low = Memory(content="x", importance=0.0, access_count=0, entity_ids=(),
                   memory_type=MemoryType.WORKING,
                   last_accessed=datetime.now(timezone.utc) - timedelta(days=365))
    low = scorer.score(m_low)
    assert 0.0 <= low <= 1.0

    # 最高配置
    m_high = Memory(content="x", importance=1.0, access_count=100,
                    entity_ids=("a", "b", "c", "d", "e"),
                    memory_type=MemoryType.SEMANTIC,
                    last_accessed=datetime.now(timezone.utc))
    high = scorer.score(m_high)
    assert 0.0 <= high <= 1.0
    assert high > low


def test_recency_factor():
    now = datetime.now(timezone.utc)
    assert _recency_factor(now) > _recency_factor(now - timedelta(days=60))


def test_access_factor():
    assert _access_factor(0) == 0.0
    assert _access_factor(10) > _access_factor(1)
    assert _access_factor(1000) <= 1.0


def test_entity_factor():
    assert _entity_factor(0) == 0.0
    assert _entity_factor(5) == 1.0
    assert _entity_factor(3) > _entity_factor(1)


def test_type_factor():
    assert _type_factor(MemoryType.SEMANTIC) > _type_factor(MemoryType.WORKING)
    assert _type_factor(MemoryType.PROCEDURAL) > _type_factor(MemoryType.EPISODIC)
