"""EntityResolver 两阶段实体消歧测试"""
from __future__ import annotations

import pytest

from neuralmem.core.types import Entity
from neuralmem.extraction.entity_resolver import EntityResolver


def _entity(name: str, eid: str | None = None) -> Entity:
    return Entity(id=eid or name[:8].ljust(8, "x"), name=name, entity_type="person")


# ── 完全匹配 ─────────────────────────────────────────────────

def test_exact_match_merges(mock_embedder):
    resolver = EntityResolver(mock_embedder)
    existing = [_entity("Alice")]
    new_e = _entity("Alice")
    result = resolver.resolve([new_e], existing)
    assert len(result) == 1
    assert result[0].id == existing[0].id   # 返回已有实体


def test_case_insensitive_exact_match(mock_embedder):
    resolver = EntityResolver(mock_embedder)
    existing = [_entity("alice")]
    result = resolver.resolve([_entity("Alice")], existing)
    assert result[0].id == existing[0].id


# ── 空图谱 ──────────────────────────────────────────────────

def test_empty_existing_returns_new(mock_embedder):
    resolver = EntityResolver(mock_embedder)
    new_e = _entity("Alice")
    result = resolver.resolve([new_e], [])
    assert result[0].id == new_e.id


def test_empty_new_entities(mock_embedder):
    resolver = EntityResolver(mock_embedder)
    result = resolver.resolve([], [_entity("Alice")])
    assert result == []


# ── 编辑距离 ─────────────────────────────────────────────────

def test_edit_distance_1_is_candidate(mock_embedder):
    """编辑距离 1 进入候选（最终是否合并取决于 embedding）"""
    resolver = EntityResolver(mock_embedder)
    existing = [_entity("Alice")]
    # "Alise" 距离 "alice" = 1（s/c）
    result = resolver.resolve([_entity("Alise")], existing)
    assert len(result) == 1  # 不崩溃，返回 1 条结果（合并或新建）


def test_edit_distance_3_no_candidate(mock_embedder):
    """编辑距离 > 2 且无子串 → 直接创建新实体（不进入候选）"""
    resolver = EntityResolver(mock_embedder)
    existing = [_entity("Alice")]
    new_e = _entity("Zxqbc")  # 完全不相关
    result = resolver.resolve([new_e], existing)
    assert result[0].id == new_e.id  # 返回新实体本身


# ── Alias 追加 ───────────────────────────────────────────────

def test_alias_not_duplicated_on_exact_match(mock_embedder):
    """完全匹配时不重复添加相同 alias"""
    resolver = EntityResolver(mock_embedder)
    existing = [_entity("Alice")]
    result = resolver.resolve([_entity("Alice")], existing)
    merged = result[0]
    aliases_list = list(merged.aliases)
    assert aliases_list.count("Alice") <= 1


def test_multiple_new_entities_independent(mock_embedder):
    """多个新实体各自独立消歧"""
    resolver = EntityResolver(mock_embedder)
    existing = [_entity("Python")]
    new_entities = [_entity("Python"), _entity("React")]
    result = resolver.resolve(new_entities, existing)
    assert len(result) == 2


# ── 内部工具函数 ─────────────────────────────────────────────

def test_edit_distance_function():
    from neuralmem.extraction.entity_resolver import _edit_distance
    assert _edit_distance("", "") == 0
    assert _edit_distance("a", "a") == 0
    assert _edit_distance("a", "b") == 1
    assert _edit_distance("kitten", "sitting") == 3
    assert _edit_distance("alice", "alise") == 1


def test_cosine_function():
    from neuralmem.extraction.entity_resolver import _cosine
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)  # 零向量
