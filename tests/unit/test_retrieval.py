"""检索策略单元测试"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from neuralmem.retrieval.semantic import SemanticStrategy
from neuralmem.retrieval.keyword import KeywordStrategy
from neuralmem.retrieval.graph import GraphStrategy
from neuralmem.retrieval.reranker import CrossEncoderReranker


def test_semantic_strategy_empty_results(mock_embedder):
    storage = MagicMock()
    storage.vector_search.return_value = []
    strategy = SemanticStrategy(storage, mock_embedder)
    results = strategy.retrieve("test query")
    assert results == []


def test_semantic_strategy_returns_ranked_items(mock_embedder):
    storage = MagicMock()
    storage.vector_search.return_value = [("mem-1", 0.9), ("mem-2", 0.7)]
    strategy = SemanticStrategy(storage, mock_embedder)
    results = strategy.retrieve("test")
    assert len(results) == 2
    assert results[0].id == "mem-1"
    assert results[0].method == "semantic"


def test_keyword_strategy_calls_storage(storage, sample_memory):
    storage.save_memory(sample_memory)
    strategy = KeywordStrategy(storage)
    results = strategy.retrieve("TypeScript frontend")
    assert isinstance(results, list)


def test_keyword_strategy_empty(storage):
    strategy = KeywordStrategy(storage)
    results = strategy.retrieve("completely nonexistent xyz123")
    assert isinstance(results, list)


def test_graph_strategy_no_entities():
    graph = MagicMock()
    graph.find_entities.return_value = []
    strategy = GraphStrategy(graph)
    results = strategy.retrieve("unknown query")
    assert results == []


def test_graph_strategy_returns_ranked_items():
    from neuralmem.core.types import Entity
    graph = MagicMock()
    entity = Entity(id="e1", name="Alice", entity_type="person")
    graph.find_entities.return_value = [entity]
    graph.traverse_for_memories.return_value = [("mem-1", 0.9), ("mem-2", 0.6)]
    strategy = GraphStrategy(graph)
    results = strategy.retrieve("Alice")
    assert len(results) == 2
    assert results[0].id == "mem-1"
    assert results[0].method == "graph"


def test_reranker_stub_preserves_order():
    reranker = CrossEncoderReranker()
    candidates = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
    result = reranker.rerank("test", candidates)
    assert result == candidates


def test_semantic_strategy_handles_exception(mock_embedder):
    storage = MagicMock()
    storage.vector_search.side_effect = Exception("DB error")
    strategy = SemanticStrategy(storage, mock_embedder)
    # 不应该抛出异常，而是返回空列表
    results = strategy.retrieve("test")
    assert results == []


def test_keyword_strategy_handles_exception():
    storage = MagicMock()
    storage.keyword_search.side_effect = Exception("DB error")
    strategy = KeywordStrategy(storage)
    results = strategy.retrieve("test")
    assert results == []
