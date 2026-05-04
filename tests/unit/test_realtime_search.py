"""Tests for NeuralMem V1.8 real-time search engine."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.streaming.realtime_search import RealtimeSearchEngine, WindowAggregation


class TestIndexAndSearch:
    def test_index_document(self):
        engine = RealtimeSearchEngine()
        engine.index_document({"content": "hello world", "user": "u1"})
        assert engine.get_stats()["total_docs"] == 1

    def test_search_basic(self):
        engine = RealtimeSearchEngine()
        engine.index_document({"content": "hello world"})
        results = engine.search("hello")
        assert len(results) == 1

    def test_search_no_match(self):
        engine = RealtimeSearchEngine()
        engine.index_document({"content": "hello"})
        results = engine.search("xyz")
        assert len(results) == 0

    def test_search_limit(self):
        engine = RealtimeSearchEngine()
        for i in range(20):
            engine.index_document({"content": f"doc {i}"})
        results = engine.search("doc", limit=5)
        assert len(results) == 5


class TestWindowAggregation:
    def test_aggregate_window(self):
        engine = RealtimeSearchEngine(window_size_sec=60.0)
        engine.index_document({"content": "a", "score": 10})
        engine.index_document({"content": "b", "score": 20})
        agg = engine.aggregate_window("score")
        assert agg.count == 2
        assert agg.values == [10, 20]


class TestSlideWindow:
    def test_slide_removes_old(self):
        engine = RealtimeSearchEngine(window_size_sec=0.01, slide_interval_sec=0.001)
        engine.index_document({"content": "old"})
        import time
        time.sleep(0.02)
        engine.slide_window()
        results = engine.search("old")
        assert len(results) == 0

    def test_slide_keeps_recent(self):
        engine = RealtimeSearchEngine(window_size_sec=60.0, slide_interval_sec=0.001)
        engine.index_document({"content": "recent"})
        engine.slide_window()
        results = engine.search("recent")
        assert len(results) == 1


class TestStats:
    def test_get_stats(self):
        engine = RealtimeSearchEngine()
        engine.index_document({"content": "test"})
        stats = engine.get_stats()
        assert stats["total_docs"] == 1
        assert stats["index_terms"] > 0

    def test_reset(self):
        engine = RealtimeSearchEngine()
        engine.index_document({"content": "test"})
        engine.reset()
        assert engine.get_stats()["total_docs"] == 0
