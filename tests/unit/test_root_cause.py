"""Tests for NeuralMem V1.8 root cause analyzer."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.diagnosis.root_cause import RootCauseAnalyzer, CausalChain, AttributionResult


class TestAnalyze:
    def test_analyze_high_latency_cache_miss(self):
        analyzer = RootCauseAnalyzer()
        chain = analyzer.analyze("high_latency", {"cache_hit_rate": 0.1})
        assert chain.root_cause == "cache_miss"
        assert chain.confidence == 0.9
        assert "cache" in chain.recommended_fix.lower()

    def test_analyze_high_latency_index(self):
        analyzer = RootCauseAnalyzer()
        chain = analyzer.analyze("high_latency", {"index_size_ratio": 3.0})
        assert chain.root_cause == "index_fragmentation"
        assert chain.confidence == 0.8

    def test_analyze_recall_failure(self):
        analyzer = RootCauseAnalyzer()
        chain = analyzer.analyze("recall_failure", {"vector_dim_mismatch": True})
        assert chain.root_cause == "embedding_mismatch"
        assert chain.confidence == 0.95

    def test_analyze_unknown_symptom(self):
        analyzer = RootCauseAnalyzer()
        chain = analyzer.analyze("unknown", {})
        assert chain.root_cause == "unknown"


class TestAttribution:
    def test_multi_factor_attribution(self):
        analyzer = RootCauseAnalyzer()
        results = analyzer.multi_factor_attribution("slow", {
            "cpu": 80,
            "memory": 20,
            "io": 50,
        })
        assert len(results) == 3
        assert results[0].factor == "cpu"
        assert results[0].impact_score == round(80 / 150, 2)

    def test_empty_factors(self):
        analyzer = RootCauseAnalyzer()
        results = analyzer.multi_factor_attribution("slow", {})
        assert len(results) == 0


class TestHistory:
    def test_history_tracking(self):
        analyzer = RootCauseAnalyzer()
        analyzer.analyze("high_latency", {"cache_hit_rate": 0.1})
        assert len(analyzer.get_history()) == 1

    def test_reset(self):
        analyzer = RootCauseAnalyzer()
        analyzer.analyze("high_latency", {})
        analyzer.reset()
        assert len(analyzer.get_history()) == 0
