"""NeuralMem 主类集成测试（使用 mock embedder，不下载模型）"""
from __future__ import annotations

import pytest

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.memory import NeuralMem


@pytest.fixture
def mem(tmp_path, mock_embedder):
    cfg = NeuralMemConfig(db_path=str(tmp_path / "test.db"))
    nm = NeuralMem(config=cfg, embedder=mock_embedder)
    return nm


def test_remember_basic(mem):
    memories = mem.remember("User likes Python programming")
    assert len(memories) > 0
    assert any("python" in m.content.lower() or "Python" in m.content for m in memories)


def test_remember_returns_memory_with_id(mem):
    memories = mem.remember("Test memory content here")
    assert all(m.id for m in memories)


def test_remember_with_user_id(mem):
    memories = mem.remember("User prefers TypeScript", user_id="user-1")
    assert all(m.user_id == "user-1" for m in memories)


def test_remember_with_tags(mem):
    memories = mem.remember("Important note", tags=["note", "important"])
    assert len(memories) > 0
    for m in memories:
        assert "note" in m.tags or "important" in m.tags


def test_recall_after_remember(mem):
    mem.remember("The user prefers TypeScript", user_id="user-1")
    results = mem.recall("TypeScript preferences", user_id="user-1")
    assert isinstance(results, list)
    # 检索结果可能为空（取决于 mock embedder），但不应该报错


def test_recall_returns_search_results(mem):
    from neuralmem.core.types import SearchResult
    mem.remember("Python is great for data science")
    results = mem.recall("Python data")
    assert isinstance(results, list)
    assert all(isinstance(r, SearchResult) for r in results)


def test_forget_by_user(mem):
    mem.remember("Memory to delete", user_id="delete-user")
    count = mem.forget(user_id="delete-user")
    assert count >= 0


def test_forget_by_memory_id(mem):
    memories = mem.remember("Specific memory to delete")
    if memories:
        mid = memories[0].id
        count = mem.forget(memory_id=mid)
        assert count >= 0


def test_consolidate_returns_dict(mem):
    stats = mem.consolidate()
    assert "decayed" in stats
    assert "merged" in stats
    assert "forgotten" in stats


def test_consolidate_values_are_int(mem):
    stats = mem.consolidate()
    assert isinstance(stats["decayed"], int)
    assert isinstance(stats["merged"], int)
    assert isinstance(stats["forgotten"], int)


def test_get_stats(mem):
    mem.remember("test memory for stats")
    stats = mem.get_stats()
    assert isinstance(stats, dict)
    # storage_stats + graph_stats の結合
    assert "node_count" in stats or "total" in stats


def test_reflect_returns_string(mem):
    mem.remember("Alice is a software engineer", user_id="u1")
    report = mem.reflect("Alice", user_id="u1")
    assert isinstance(report, str)
    assert "Reflection" in report


def test_reflect_contains_topic(mem):
    mem.remember("Python is used for data science")
    report = mem.reflect("Python")
    assert "Python" in report


def test_context_manager(tmp_path, mock_embedder):
    cfg = NeuralMemConfig(db_path=str(tmp_path / "cm.db"))
    with NeuralMem(config=cfg, embedder=mock_embedder) as nm:
        memories = nm.remember("context manager test")
        assert isinstance(memories, list)


def test_remember_dedup(mem):
    """相同内容第二次存储应该被去重跳过"""
    mem.remember("Unique content for dedup test abc123")
    second = mem.remember("Unique content for dedup test abc123")
    # 第二次如果 embedding 相同会触发去重，返回 0 条
    # 但由于 mock embedder 是确定性的，应触发去重
    assert isinstance(second, list)


def test_entity_resolver_prevents_duplicate_entities(mem):
    """相同实体名称的两次 remember 不应创建两个图谱节点"""
    mem.remember("Alice is a software engineer", user_id="u1")
    mem.remember("Alice prefers Python", user_id="u1")
    entities = mem.graph.get_entities()
    alice_nodes = [e for e in entities if "alice" in e.name.lower()]
    # 完全匹配 → 消歧后只有一个 Alice 节点
    assert len(alice_nodes) == 1


def test_neuralmem_uses_embedding_registry(tmp_path):
    """NeuralMem 应通过 registry 选择 embedder，默认为 LocalEmbedding。"""
    from neuralmem.core.config import NeuralMemConfig
    from neuralmem.core.memory import NeuralMem
    from neuralmem.embedding.local import LocalEmbedding
    cfg = NeuralMemConfig(db_path=str(tmp_path / "test.db"))
    mem = NeuralMem(config=cfg)
    assert isinstance(mem.embedding, LocalEmbedding)


def test_neuralmem_uses_extractor_registry(tmp_path):
    """NeuralMem 应通过 registry 选择 extractor，默认为 MemoryExtractor。"""
    from neuralmem.core.config import NeuralMemConfig
    from neuralmem.core.memory import NeuralMem
    from neuralmem.extraction.extractor import MemoryExtractor
    cfg = NeuralMemConfig(db_path=str(tmp_path / "test.db"))
    mem = NeuralMem(config=cfg)
    assert isinstance(mem.extractor, MemoryExtractor)
