"""Tests for AdaptiveRetriever — strategy weight adaptation."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neuralmem.core.types import Memory, SearchQuery, SearchResult
from neuralmem.retrieval.adaptive import AdaptiveRetriever
from neuralmem.retrieval.fusion import RankedItem

# --- Helpers ---

def _make_engine(results=None):
    """Create a mock retrieval engine."""
    engine = MagicMock()
    engine.search.return_value = results or []
    return engine


def _make_result(mem_id: str, score: float, method: str = "semantic"):
    """Create a SearchResult with minimal data."""
    mem = Memory(content=f"content-{mem_id}", id=mem_id)
    return SearchResult(
        memory=mem, score=score, retrieval_method=method
    )


# --- Tests ---

class TestAdaptiveRetriever:
    def test_initial_weights_are_equal(self):
        engine = _make_engine()
        ar = AdaptiveRetriever(engine)
        weights = ar.get_weights()
        assert all(w == 1.0 for w in weights.values())

    def test_initial_success_rates_are_half(self):
        engine = _make_engine()
        ar = AdaptiveRetriever(engine)
        rates = ar.get_success_rates()
        assert all(r == 0.5 for r in rates.values())

    def test_strategies_defined(self):
        engine = _make_engine()
        ar = AdaptiveRetriever(engine)
        assert set(ar.STRATEGIES) == {
            "semantic", "keyword", "graph", "temporal"
        }

    def test_feedback_positive_updates_ema(self):
        engine = _make_engine()
        ar = AdaptiveRetriever(engine, alpha=0.5)
        ar.feedback("semantic", was_useful=True)
        rates = ar.get_success_rates()
        # EMA = 0.5 * 1.0 + 0.5 * 0.5 = 0.75
        assert rates["semantic"] == pytest.approx(0.75)

    def test_feedback_negative_updates_ema(self):
        engine = _make_engine()
        ar = AdaptiveRetriever(engine, alpha=0.5)
        ar.feedback("semantic", was_useful=False)
        rates = ar.get_success_rates()
        # EMA = 0.5 * 0.0 + 0.5 * 0.5 = 0.25
        assert rates["semantic"] == pytest.approx(0.25)

    def test_feedback_unknown_strategy_ignored(self):
        engine = _make_engine()
        ar = AdaptiveRetriever(engine)
        ar.feedback("nonexistent", was_useful=True)
        # Should not crash or affect other strategies
        assert all(w == 1.0 for w in ar.get_weights().values())

    def test_counters_increment(self):
        engine = _make_engine()
        ar = AdaptiveRetriever(engine)
        ar.feedback("semantic", was_useful=True)
        ar.feedback("semantic", was_useful=True)
        ar.feedback("semantic", was_useful=False)
        counters = ar.get_counters()
        assert counters["semantic"]["success"] == 2
        assert counters["semantic"]["failure"] == 1

    def test_weights_adjust_after_feedback(self):
        engine = _make_engine()
        ar = AdaptiveRetriever(engine, alpha=0.3)
        # Many positive for semantic, many negative for keyword
        for _ in range(20):
            ar.feedback("semantic", was_useful=True)
            ar.feedback("keyword", was_useful=False)

        weights = ar.get_weights()
        assert weights["semantic"] > weights["keyword"]

    def test_search_delegates_to_engine(self):
        results = [_make_result("m1", 0.8)]
        engine = _make_engine(results)
        ar = AdaptiveRetriever(engine)
        query = SearchQuery(query="test")
        out = ar.search(query)
        engine.search.assert_called_once_with(query)
        assert len(out) == 1

    def test_search_records_strategy_results(self):
        results = [
            _make_result("m1", 0.8, "semantic"),
            _make_result("m2", 0.6, "keyword"),
        ]
        engine = _make_engine(results)
        ar = AdaptiveRetriever(engine)
        ar.search(SearchQuery(query="test"))
        assert "semantic" in ar._last_strategy_results
        assert "keyword" in ar._last_strategy_results

    def test_weighted_rrf_fusion_empty(self):
        engine = _make_engine()
        ar = AdaptiveRetriever(engine)
        result = ar.weighted_rrf_fusion({})
        assert result == []

    def test_weighted_rrf_fusion_basic(self):
        engine = _make_engine()
        ar = AdaptiveRetriever(engine)
        strategy_results = {
            "semantic": [
                RankedItem(id="m1", score=0.9, method="semantic"),
                RankedItem(id="m2", score=0.7, method="semantic"),
            ],
            "keyword": [
                RankedItem(id="m1", score=0.8, method="keyword"),
            ],
        }
        result = ar.weighted_rrf_fusion(strategy_results)
        assert len(result) == 2
        # m1 appears in both → should be ranked first
        assert result[0][0] == "m1"

    def test_weighted_rrf_respects_weights(self):
        engine = _make_engine()
        ar = AdaptiveRetriever(engine)
        # Give semantic very high weight
        ar._weights["semantic"] = 10.0
        ar._weights["keyword"] = 0.1

        strategy_results = {
            "semantic": [
                RankedItem(id="m_sem", score=0.5, method="semantic"),
            ],
            "keyword": [
                RankedItem(id="m_key", score=0.5, method="keyword"),
            ],
        }
        result = ar.weighted_rrf_fusion(strategy_results)
        # m_sem should rank higher due to higher strategy weight
        assert result[0][0] == "m_sem"

    def test_reset(self):
        engine = _make_engine()
        ar = AdaptiveRetriever(engine, alpha=0.5)
        ar.feedback("semantic", was_useful=True)
        ar.reset()
        weights = ar.get_weights()
        assert all(w == 1.0 for w in weights.values())
        rates = ar.get_success_rates()
        assert all(r == 0.5 for r in rates.values())
        counters = ar.get_counters()
        for s in ar.STRATEGIES:
            assert counters[s]["success"] == 0
            assert counters[s]["failure"] == 0

    def test_weighted_rrf_normalizes_to_01(self):
        engine = _make_engine()
        ar = AdaptiveRetriever(engine)
        strategy_results = {
            "semantic": [
                RankedItem(id="m1", score=0.9, method="semantic"),
                RankedItem(id="m2", score=0.7, method="semantic"),
                RankedItem(id="m3", score=0.5, method="semantic"),
            ],
        }
        result = ar.weighted_rrf_fusion(strategy_results)
        scores = [s for _, s in result]
        assert max(scores) == pytest.approx(1.0)
        assert min(scores) >= 0.0

    def test_feedback_multiple_strategies(self):
        engine = _make_engine()
        ar = AdaptiveRetriever(engine, alpha=0.2)
        ar.feedback("semantic", was_useful=True)
        ar.feedback("keyword", was_useful=False)
        ar.feedback("graph", was_useful=True)
        ar.feedback("temporal", was_useful=True)
        rates = ar.get_success_rates()
        assert rates["semantic"] > 0.5
        assert rates["keyword"] < 0.5
        assert rates["graph"] > 0.5
        assert rates["temporal"] > 0.5
