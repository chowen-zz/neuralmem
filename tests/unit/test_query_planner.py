"""Tests for QueryPlanner."""
from __future__ import annotations

from neuralmem.perf.query_planner import (
    QueryPlanner,
    QueryProfile,
    StrategyWeights,
)


class TestQueryPlanner:
    def test_analyze_short_query(self):
        qp = QueryPlanner()
        profile = qp.analyze("what is Python")
        assert profile.size_category == "short"
        assert profile.length == 14
        assert profile.is_factual is True

    def test_analyze_medium_query(self):
        qp = QueryPlanner()
        query = "a" * 100
        profile = qp.analyze(query)
        assert profile.size_category == "medium"

    def test_analyze_long_query(self):
        qp = QueryPlanner()
        query = "a" * 250
        profile = qp.analyze(query)
        assert profile.size_category == "long"

    def test_analyze_temporal(self):
        qp = QueryPlanner()
        profile = qp.analyze(
            "what happened yesterday afternoon"
        )
        assert profile.is_temporal is True

    def test_analyze_entities(self):
        qp = QueryPlanner()
        profile = qp.analyze(
            "tell me about John Smith and Alice Johnson"
        )
        assert profile.has_entities is True
        assert profile.entity_count >= 2

    def test_analyze_factual(self):
        qp = QueryPlanner()
        profile = qp.analyze("where is the library located")
        assert profile.is_factual is True

    def test_select_short_factual(self):
        qp = QueryPlanner()
        profile = qp.analyze("what is AI")
        weights = qp.select_strategies(profile)
        assert "semantic" in weights
        assert "keyword" in weights
        assert weights["semantic"] > weights["keyword"]

    def test_select_long_narrative(self):
        qp = QueryPlanner()
        query = "describe the history of computing " * 10
        profile = qp.analyze(query)
        weights = qp.select_strategies(profile)
        assert "semantic" in weights
        assert "temporal" in weights
        assert "graph" in weights

    def test_select_entity_heavy(self):
        qp = QueryPlanner()
        profile = qp.analyze(
            "relationship between Microsoft and Google"
        )
        weights = qp.select_strategies(profile)
        assert "graph" in weights

    def test_select_temporal(self):
        qp = QueryPlanner()
        # Use a non-factual temporal query to avoid
        # the short-factual rule taking priority
        profile = qp.analyze(
            "events from last monday morning"
        )
        weights = qp.select_strategies(profile)
        assert "temporal" in weights
        assert weights["temporal"] > 0

    def test_select_default(self):
        qp = QueryPlanner()
        profile = qp.analyze("tell me stuff")
        weights = qp.select_strategies(profile)
        assert "semantic" in weights

    def test_explain_returns_string(self):
        qp = QueryPlanner()
        explanation = qp.explain("what is Python")
        assert isinstance(explanation, str)
        assert "what is Python" in explanation
        assert "short" in explanation

    def test_explain_contains_strategies(self):
        qp = QueryPlanner()
        explanation = qp.explain(
            "what happened yesterday to John Smith"
        )
        assert "Strategy weights:" in explanation

    def test_custom_default_weights(self):
        qp = QueryPlanner(
            default_weights={
                "semantic": 0.5,
                "keyword": 0.2,
                "graph": 0.2,
                "temporal": 0.1,
            }
        )
        w = qp.default_weights
        assert w.semantic == 0.5
        assert w.keyword == 0.2

    def test_strategy_weights_to_dict(self):
        w = StrategyWeights(
            semantic=0.4,
            keyword=0.3,
            graph=0.0,
            temporal=0.3,
        )
        d = w.to_dict()
        assert "graph" not in d
        assert d["semantic"] == 0.4

    def test_query_profile_dataclass(self):
        p = QueryProfile(
            query="test",
            length=4,
            size_category="short",
        )
        assert p.query == "test"
        assert p.has_entities is False

    def test_analyze_boundary_50(self):
        qp = QueryPlanner()
        profile = qp.analyze("a" * 50)
        assert profile.size_category == "medium"

    def test_analyze_boundary_200(self):
        qp = QueryPlanner()
        profile = qp.analyze("a" * 200)
        assert profile.size_category == "medium"

    def test_analyze_boundary_201(self):
        qp = QueryPlanner()
        profile = qp.analyze("a" * 201)
        assert profile.size_category == "long"
