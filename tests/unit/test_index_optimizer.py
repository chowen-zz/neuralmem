"""Tests for NeuralMem V1.8 index optimizer."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.storage.index_optimizer import IndexOptimizer, IndexRecommendation, QueryLogEntry


class TestQueryLogging:
    def test_record_query(self):
        opt = IndexOptimizer()
        opt.record_query({"user_id": "u1", "type": "note"}, ["created_at"], 45.0)
        assert len(opt._query_logs) == 1

    def test_analyze_filter_patterns(self):
        opt = IndexOptimizer()
        for _ in range(20):
            opt.record_query({"user_id": "u1", "type": "note"}, ["created_at"], 45.0)
        for _ in range(10):
            opt.record_query({"tag": "work"}, [], 30.0)
        freq = opt.analyze_filter_patterns()
        assert freq["user_id"] == 20
        assert freq["type"] == 20


class TestRecommendations:
    def test_recommend_composite_index(self):
        opt = IndexOptimizer()
        for _ in range(20):
            opt.record_query({"user_id": "u1", "type": "note"}, ["created_at"], 45.0)
        recs = opt.recommend_indexes(min_frequency=10)
        assert len(recs) == 1
        assert "user_id" in recs[0].columns
        assert recs[0].index_type == "composite"

    def test_recommend_empty_when_low_frequency(self):
        opt = IndexOptimizer()
        opt.record_query({"user_id": "u1"}, [], 45.0)
        recs = opt.recommend_indexes(min_frequency=10)
        assert len(recs) == 0

    def test_estimate_speedup(self):
        opt = IndexOptimizer()
        rec = IndexRecommendation(columns=["a", "b", "c"])
        speedup = opt.estimate_speedup(rec)
        assert speedup == 4.5


class TestQueryStats:
    def test_get_query_stats(self):
        opt = IndexOptimizer()
        opt.record_query({"a": "1"}, [], 10.0)
        opt.record_query({"b": "2"}, [], 20.0)
        stats = opt.get_query_stats()
        assert stats["total_queries"] == 2
        assert stats["avg_latency_ms"] == 15.0

    def test_empty_stats(self):
        opt = IndexOptimizer()
        stats = opt.get_query_stats()
        assert stats["total_queries"] == 0

    def test_reset(self):
        opt = IndexOptimizer()
        opt.record_query({"a": "1"}, [], 10.0)
        opt.reset()
        assert len(opt._query_logs) == 0
