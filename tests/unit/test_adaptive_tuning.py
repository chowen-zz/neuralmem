"""Tests for NeuralMem V1.8 adaptive tuning engine."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.intelligence.adaptive_tuning import (
    AdaptiveTuningEngine, ParameterSet, QueryPattern, TuningResult
)


class TestQueryPatternClassification:
    def test_classify_batch(self):
        engine = AdaptiveTuningEngine()
        for _ in range(50):
            engine.record_query("batch", 100.0, 10)
        assert engine.classify_pattern() == QueryPattern.BATCH

    def test_classify_oltp(self):
        engine = AdaptiveTuningEngine()
        for _ in range(50):
            engine.record_query("recall", 5.0, 5)
        assert engine.classify_pattern() == QueryPattern.OLTP

    def test_classify_olap(self):
        engine = AdaptiveTuningEngine()
        for _ in range(50):
            engine.record_query("search", 200.0, 150)
        assert engine.classify_pattern() == QueryPattern.OLAP

    def test_classify_empty(self):
        engine = AdaptiveTuningEngine()
        assert engine.classify_pattern() == QueryPattern.RANDOM


class TestParameterRecommendation:
    def test_sequential_params(self):
        engine = AdaptiveTuningEngine()
        params = engine.recommend_params(QueryPattern.SEQUENTIAL)
        assert params.cache_strategy == "predictive"
        assert params.prefetch_depth == 5

    def test_oltp_params(self):
        engine = AdaptiveTuningEngine()
        params = engine.recommend_params(QueryPattern.OLTP)
        assert params.vector_dim == 256

    def test_olap_params(self):
        engine = AdaptiveTuningEngine()
        params = engine.recommend_params(QueryPattern.OLAP)
        assert params.vector_dim == 512
        assert params.rrf_k == 100.0


class TestTuningApplication:
    def test_apply_tuning(self):
        engine = AdaptiveTuningEngine()
        for _ in range(50):
            engine.record_query("batch", 100.0, 10)
        result = engine.apply_tuning()
        assert isinstance(result, TuningResult)
        assert result.pattern == QueryPattern.BATCH
        assert result.confidence == 0.85

    def test_callback_called(self):
        cb = MagicMock()
        engine = AdaptiveTuningEngine(param_callback=cb)
        for _ in range(50):
            engine.record_query("batch", 100.0, 10)
        engine.apply_tuning()
        cb.assert_called_once()

    def test_history_tracking(self):
        engine = AdaptiveTuningEngine()
        for _ in range(50):
            engine.record_query("batch", 100.0, 10)
        engine.apply_tuning()
        assert len(engine.get_tuning_history()) == 1

    def test_get_current_params(self):
        engine = AdaptiveTuningEngine()
        params = engine.get_current_params()
        assert isinstance(params, ParameterSet)
        assert params.rrf_k == 60.0

    def test_reset(self):
        engine = AdaptiveTuningEngine()
        engine.record_query("test", 10.0, 5)
        engine.apply_tuning()
        engine.reset()
        assert len(engine.get_tuning_history()) == 0
        assert len(engine._query_log) == 0
