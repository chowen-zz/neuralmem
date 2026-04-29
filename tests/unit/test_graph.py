"""知识图谱单元测试"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from neuralmem.core.types import Entity, Relation
from neuralmem.graph.knowledge_graph import KnowledgeGraph


def make_graph():
    """创建带 mock storage 的 KnowledgeGraph"""
    storage = MagicMock()
    storage.load_graph_snapshot = MagicMock(return_value=None)
    storage.save_graph_snapshot = MagicMock()
    return KnowledgeGraph(storage)


def test_upsert_and_get_entity():
    kg = make_graph()
    e = Entity(id="e1", name="Alice", entity_type="person")
    kg.upsert_entity(e)
    retrieved = kg.get_entity("e1")
    assert retrieved is not None
    assert retrieved.name == "Alice"


def test_get_entity_not_found():
    kg = make_graph()
    assert kg.get_entity("nonexistent") is None


def test_add_relation():
    kg = make_graph()
    e1 = Entity(id="e1", name="Alice", entity_type="person")
    e2 = Entity(id="e2", name="Python", entity_type="technology")
    kg.upsert_entity(e1)
    kg.upsert_entity(e2)
    r = Relation(source_id="e1", target_id="e2", relation_type="uses")
    kg.add_relation(r)
    neighbors = kg.get_neighbors(["e1"])
    assert any(e.name == "Python" for e in neighbors)


def test_find_entities_by_name():
    kg = make_graph()
    kg.upsert_entity(Entity(id="e1", name="Alice", entity_type="person"))
    kg.upsert_entity(Entity(id="e2", name="Bob", entity_type="person"))
    results = kg.find_entities("Ali")
    assert len(results) == 1
    assert results[0].name == "Alice"


def test_find_entities_case_insensitive():
    kg = make_graph()
    kg.upsert_entity(Entity(id="e1", name="Python", entity_type="technology"))
    results = kg.find_entities("python")
    assert len(results) == 1


def test_get_entities_returns_all():
    kg = make_graph()
    for i in range(3):
        kg.upsert_entity(Entity(id=f"e{i}", name=f"Entity{i}", entity_type="test"))
    entities = kg.get_entities()
    assert len(entities) == 3


def test_link_memory_to_entity():
    kg = make_graph()
    kg.upsert_entity(Entity(id="e1", name="Test"))
    kg.link_memory_to_entity("mem-001", "e1")
    result = kg.traverse_for_memories(["e1"])
    assert any(mid == "mem-001" for mid, _ in result)


def test_get_stats():
    kg = make_graph()
    kg.upsert_entity(Entity(id="e1", name="A"))
    kg.upsert_entity(Entity(id="e2", name="B"))
    kg.add_relation(Relation(source_id="e1", target_id="e2", relation_type="test"))
    stats = kg.get_stats()
    assert stats["node_count"] == 2
    assert stats["edge_count"] == 1


def test_upsert_entity_preserves_first_seen():
    """二次 upsert 时 first_seen 不变"""
    kg = make_graph()
    e = Entity(id="e1", name="Alice", entity_type="person")
    kg.upsert_entity(e)
    first = kg.get_entity("e1")
    # 用新 entity 对象再次 upsert（名字不变）
    e2 = Entity(id="e1", name="Alice Updated", entity_type="person")
    kg.upsert_entity(e2)
    second = kg.get_entity("e1")
    assert second is not None
    assert second.first_seen == first.first_seen


def test_get_neighbors_empty():
    kg = make_graph()
    kg.upsert_entity(Entity(id="e1", name="Lone"))
    neighbors = kg.get_neighbors(["e1"])
    assert neighbors == []
