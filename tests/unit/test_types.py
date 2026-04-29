"""数据模型单元测试"""
from __future__ import annotations
import pytest
from datetime import datetime
from neuralmem.core.types import (
    Memory, Entity, Relation, SearchResult, MemoryType, MemoryScope, SearchQuery
)


def test_memory_defaults():
    m = Memory(content="test")
    assert m.id
    assert m.memory_type == MemoryType.SEMANTIC
    assert m.scope == MemoryScope.USER
    assert m.importance == 0.5
    assert m.access_count == 0
    assert isinstance(m.created_at, datetime)


def test_memory_id_unique():
    m1 = Memory(content="a")
    m2 = Memory(content="b")
    assert m1.id != m2.id


def test_memory_tags_tuple():
    m = Memory(content="test", tags=("a", "b"))
    assert isinstance(m.tags, tuple)


def test_memory_embedding_excluded_from_json():
    m = Memory(content="test", embedding=[1.0, 2.0])
    data = m.model_dump()
    assert "embedding" not in data


def test_entity_defaults():
    e = Entity(name="Alice")
    assert e.id
    assert e.entity_type == "unknown"
    assert isinstance(e.aliases, tuple)


def test_entity_immutable():
    e = Entity(name="Alice")
    with pytest.raises(Exception):
        e.name = "Bob"  # type: ignore[misc]


def test_relation_creation():
    r = Relation(source_id="a", target_id="b", relation_type="uses")
    assert r.weight == 1.0


def test_search_query_defaults():
    q = SearchQuery(query="test")
    assert q.limit == 10
    assert q.min_score == 0.3


def test_memory_type_values():
    assert MemoryType.SEMANTIC == "semantic"
    assert MemoryType.EPISODIC == "episodic"
    assert MemoryType.PROCEDURAL == "procedural"
    assert MemoryType.WORKING == "working"


def test_search_result_creation():
    m = Memory(content="test content")
    r = SearchResult(memory=m, score=0.9, retrieval_method="semantic")
    assert r.score == 0.9
    assert r.retrieval_method == "semantic"
