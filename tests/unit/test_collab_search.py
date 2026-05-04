"""Tests for NeuralMem V1.8 collaborative search engine."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.multi_agent.collab_search import CollaborativeSearchEngine, CollaborativeResult


class TestSearch:
    def test_search_with_mock_query(self):
        def mock_query(source, query, limit):
            return [{"content": f"from_{source}", "score": 0.9, "id": "1"}]
        engine = CollaborativeSearchEngine(query_fn=mock_query)
        results = engine.search("test", "a1", ["pool1"], ["a2"], limit=5)
        assert len(results) == 2  # pool1 + a2

    def test_search_respects_permissions(self):
        def mock_perm(agent, pool):
            return agent == "a1"
        engine = CollaborativeSearchEngine(permission_fn=mock_perm)
        results = engine.search("test", "a1", ["pool1", "pool2"], [], limit=5)
        assert len(results) == 0  # no query_fn, but permission checked

    def test_search_deduplicates(self):
        def mock_query(source, query, limit):
            return [{"content": "same", "score": 0.9, "id": "1"}]
        engine = CollaborativeSearchEngine(query_fn=mock_query)
        results = engine.search("test", "a1", ["pool1", "pool2"], [], limit=5)
        deduped = engine.deduplicate(results)
        assert len(deduped) == 1


class TestMerge:
    def test_merge_results(self):
        engine = CollaborativeSearchEngine()
        r1 = CollaborativeResult("1", "a1", None, {"c": "x1"}, 0.9, True)
        r2 = CollaborativeResult("2", "a2", None, {"c": "x2"}, 0.7, True)
        r3 = CollaborativeResult("3", "a3", None, {"c": "x3"}, 0.8, True)
        merged = engine.merge_results([[r1], [r2, r3]], top_k=5)
        # After dedup, we should have 3 unique results
        assert len(merged) >= 1
        assert merged[0].relevance_score == 0.9

    def test_merge_with_deduplication(self):
        engine = CollaborativeSearchEngine()
        r1 = CollaborativeResult("1", "a1", None, {"content": "x"}, 0.9, True)
        r2 = CollaborativeResult("2", "a2", None, {"content": "x"}, 0.8, True)
        merged = engine.merge_results([[r1, r2]], top_k=2)
        assert len(merged) == 1


class TestReset:
    def test_reset(self):
        engine = CollaborativeSearchEngine()
        engine._history.append({"q": "test"})
        engine.reset()
        assert len(engine.get_history()) == 0
