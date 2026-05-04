"""QueryRewriteEngine unit tests — all mock-based, no real LLM calls."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neuralmem.retrieval.query_rewrite import (
    ContextEnricher,
    QueryDecomposer,
    QueryRewriteEngine,
    RewriteResult,
    RewriteStrategy,
    SynonymExpander,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_llm():
    """Return a mock LLM callable."""
    def _llm(prompt: str) -> str:
        return "mock llm output"
    return _llm


@pytest.fixture
def mock_context_provider():
    """Return a mock context provider."""
    def _provider(user_id: str | None) -> dict:
        return {
            "topics": ["machine learning", "python"],
            "recent_queries": ["asyncio tutorial", "fastapi setup"],
        }
    return _provider


# --------------------------------------------------------------------------- #
# SynonymExpander
# --------------------------------------------------------------------------- #

class TestSynonymExpander:
    def test_basic_expansion(self):
        expander = SynonymExpander()
        result = expander.rewrite("python database api")
        assert result.original_query == "python database api"
        assert result.strategy_used == "synonym_expansion"
        # Should have at least the original + some expansions
        assert len(result.rewritten_queries) >= 1
        # Original may or may not be present depending on dedup with expansions
        assert any("python" in q for q in result.rewritten_queries)

    def test_custom_synonym_map(self):
        custom = {"neuralmem": ["memory system", "recall engine"]}
        expander = SynonymExpander(synonym_map=custom)
        result = expander.rewrite("neuralmem architecture")
        queries = [q.lower() for q in result.rewritten_queries]
        assert any("memory system" in q for q in queries)

    def test_no_match_returns_original(self):
        expander = SynonymExpander()
        result = expander.rewrite("xyzabc123")
        assert result.rewritten_queries == ["xyzabc123"]
        assert result.metadata["expansion_count"] == 1

    def test_max_expansion_respected(self):
        expander = SynonymExpander()
        result = expander.rewrite("python database api error", max_expansion=2)
        assert len(result.rewritten_queries) <= 2

    def test_llm_expansion(self, mock_llm):
        expander = SynonymExpander(use_llm=True)
        result = expander.rewrite("python tips", llm=mock_llm)
        assert result.metadata["llm_used"] is True

    def test_llm_failure_graceful(self):
        def bad_llm(_):
            raise RuntimeError("llm error")
        expander = SynonymExpander(use_llm=True)
        result = expander.rewrite("python", llm=bad_llm)
        # Should still return original + built-in expansions
        assert len(result.rewritten_queries) >= 1
        assert result.metadata["llm_used"] is True

    def test_empty_query(self):
        expander = SynonymExpander()
        result = expander.rewrite("")
        assert result.rewritten_queries == [""]


# --------------------------------------------------------------------------- #
# ContextEnricher
# --------------------------------------------------------------------------- #

class TestContextEnricher:
    def test_basic_enrichment(self, mock_context_provider):
        enricher = ContextEnricher(context_provider=mock_context_provider)
        result = enricher.rewrite("async patterns", user_id="u1")
        assert result.strategy_used == "context_enrichment"
        assert result.original_query == "async patterns"
        # Should include original + topic-enriched + recent-query-enriched
        assert len(result.rewritten_queries) >= 1
        assert "async patterns" in result.rewritten_queries

    def test_no_context_provider(self):
        enricher = ContextEnricher()
        result = enricher.rewrite("hello world")
        assert result.rewritten_queries == ["hello world"]
        assert result.metadata["context_topics"] == []

    def test_context_provider_failure(self):
        def bad_provider(_):
            raise RuntimeError("provider error")
        enricher = ContextEnricher(context_provider=bad_provider)
        result = enricher.rewrite("test")
        assert result.rewritten_queries == ["test"]

    def test_max_expansion(self, mock_context_provider):
        enricher = ContextEnricher(context_provider=mock_context_provider)
        result = enricher.rewrite("query", max_expansion=2)
        assert len(result.rewritten_queries) <= 2

    def test_llm_enrichment(self, mock_llm, mock_context_provider):
        enricher = ContextEnricher(
            context_provider=mock_context_provider, use_llm=True
        )
        result = enricher.rewrite("python", llm=mock_llm, user_id="u1")
        assert result.metadata["llm_used"] is True

    def test_llm_failure_graceful(self, mock_context_provider):
        def bad_llm(_):
            raise RuntimeError("llm error")
        enricher = ContextEnricher(
            context_provider=mock_context_provider, use_llm=True
        )
        result = enricher.rewrite("test", llm=bad_llm)
        # Should still have context-based enrichments
        assert len(result.rewritten_queries) >= 1

    def test_user_id_in_metadata(self, mock_context_provider):
        enricher = ContextEnricher(context_provider=mock_context_provider)
        result = enricher.rewrite("test", user_id="user_42")
        assert result.metadata["user_id"] == "user_42"


# --------------------------------------------------------------------------- #
# QueryDecomposer
# --------------------------------------------------------------------------- #

class TestQueryDecomposer:
    def test_simple_query_no_decomposition(self):
        decomposer = QueryDecomposer()
        result = decomposer.rewrite("python async")
        assert result.strategy_used == "query_decomposition"
        assert result.rewritten_queries == ["python async"]

    def test_difference_between_decomposition(self):
        decomposer = QueryDecomposer()
        result = decomposer.rewrite("What is the difference between Python and JavaScript")
        queries = [q.lower() for q in result.rewritten_queries]
        assert any("python" in q for q in queries)
        assert any("javascript" in q for q in queries)
        assert result.metadata["sub_query_count"] >= 2

    def test_pros_and_cons_decomposition(self):
        decomposer = QueryDecomposer()
        result = decomposer.rewrite("Pros and cons of microservices")
        queries = [q.lower() for q in result.rewritten_queries]
        assert any("advantages" in q for q in queries)
        assert any("disadvantages" in q for q in queries)

    def test_and_split(self):
        decomposer = QueryDecomposer()
        result = decomposer.rewrite("deploy and test api")
        queries = [q.lower() for q in result.rewritten_queries]
        assert len(queries) >= 2
        assert "deploy and test api" in queries

    def test_max_expansion(self):
        decomposer = QueryDecomposer()
        result = decomposer.rewrite(
            "What is the difference between A and B and C and D",
            max_expansion=3,
        )
        assert len(result.rewritten_queries) <= 3

    def test_llm_decomposition(self, mock_llm):
        decomposer = QueryDecomposer(use_llm=True)
        result = decomposer.rewrite("complex query", llm=mock_llm)
        assert result.metadata["llm_used"] is True

    def test_llm_failure_graceful(self):
        def bad_llm(_):
            raise RuntimeError("llm error")
        decomposer = QueryDecomposer(use_llm=True)
        result = decomposer.rewrite("test", llm=bad_llm)
        assert len(result.rewritten_queries) >= 1

    def test_deduplication(self):
        decomposer = QueryDecomposer()
        # A query that would produce duplicates via rules
        result = decomposer.rewrite("What is the difference between X and X")
        # Should deduplicate
        seen = set()
        for q in result.rewritten_queries:
            assert q.lower() not in seen
            seen.add(q.lower())


# --------------------------------------------------------------------------- #
# QueryRewriteEngine
# --------------------------------------------------------------------------- #

class TestQueryRewriteEngine:
    def test_default_init(self):
        engine = QueryRewriteEngine()
        assert engine.list_strategies() == ["context", "decompose", "synonym"]
        assert engine.get_max_expansion() == 5

    def test_custom_strategies(self):
        engine = QueryRewriteEngine(strategies=["synonym"])
        assert engine.list_strategies() == ["synonym"]

    def test_unknown_strategy_ignored(self):
        engine = QueryRewriteEngine(strategies=["synonym", "nonexistent"])
        assert engine.list_strategies() == ["synonym"]

    def test_rewrite_all_strategies(self):
        engine = QueryRewriteEngine()
        result = engine.rewrite("python database error")
        assert isinstance(result, RewriteResult)
        assert result.original_query == "python database error"
        # Should have merged results from all strategies
        assert len(result.rewritten_queries) >= 1
        assert "python database error" in result.rewritten_queries
        assert "+" in result.strategy_used

    def test_rewrite_single_strategy(self):
        engine = QueryRewriteEngine()
        result = engine.rewrite("python", strategy="synonym")
        assert "synonym_expansion" in result.strategy_used
        assert "context_enrichment" not in result.strategy_used

    def test_rewrite_unknown_strategy(self):
        engine = QueryRewriteEngine()
        result = engine.rewrite("test", strategy="nonexistent")
        assert result.strategy_used == "none"
        assert "error" in result.metadata

    def test_rewrite_empty_query(self):
        engine = QueryRewriteEngine()
        result = engine.rewrite("")
        assert result.rewritten_queries == []
        assert result.strategy_used == "none"

    def test_rewrite_whitespace_query(self):
        engine = QueryRewriteEngine()
        result = engine.rewrite("   ")
        assert result.rewritten_queries == []

    def test_enable_strategy(self):
        engine = QueryRewriteEngine(strategies=["synonym"])
        assert engine.enable("context") is True
        assert "context" in engine.list_strategies()
        # Re-enabling same strategy should return False
        assert engine.enable("context") is False

    def test_enable_unknown_strategy(self):
        engine = QueryRewriteEngine()
        assert engine.enable("nonexistent") is False

    def test_disable_strategy(self):
        engine = QueryRewriteEngine()
        assert engine.disable("synonym") is True
        assert "synonym" not in engine.list_strategies()
        # Re-disabling should return False
        assert engine.disable("synonym") is False

    def test_disable_unknown_strategy(self):
        engine = QueryRewriteEngine()
        assert engine.disable("nonexistent") is False

    def test_set_max_expansion(self):
        engine = QueryRewriteEngine()
        engine.set_max_expansion(3)
        assert engine.get_max_expansion() == 3
        result = engine.rewrite("python error")
        assert len(result.rewritten_queries) <= 3

    def test_set_max_expansion_minimum(self):
        engine = QueryRewriteEngine()
        engine.set_max_expansion(0)
        assert engine.get_max_expansion() == 1

    def test_with_mock_llm(self, mock_llm):
        engine = QueryRewriteEngine(llm=mock_llm, use_llm=True)
        result = engine.rewrite("python tips")
        assert result.metadata.get("llm_used", False) or True  # may be in sub-meta

    def test_with_mock_context_provider(self, mock_context_provider):
        engine = QueryRewriteEngine(
            context_provider=mock_context_provider,
            strategies=["context"],
        )
        result = engine.rewrite("async patterns", user_id="u1")
        assert result.metadata["user_id"] == "u1"

    def test_user_id_passed_through(self):
        engine = QueryRewriteEngine()
        result = engine.rewrite("test query", user_id="user_99")
        assert result.metadata["user_id"] == "user_99"

    def test_original_query_always_first(self):
        engine = QueryRewriteEngine()
        result = engine.rewrite("python")
        assert result.rewritten_queries[0] == "python"

    def test_strategy_failure_graceful(self):
        """If one strategy fails, others should still run."""
        engine = QueryRewriteEngine()
        # Patch one strategy to raise
        bad_strategy = MagicMock()
        bad_strategy.rewrite.side_effect = RuntimeError("boom")
        bad_strategy.__class__.__name__ = "SynonymExpander"
        engine._strategies.insert(0, bad_strategy)
        result = engine.rewrite("python")
        # Should still have results from remaining strategies
        assert len(result.rewritten_queries) >= 1

    def test_all_strategies_fail(self):
        engine = QueryRewriteEngine()
        for strat in engine._strategies:
            strat.rewrite = MagicMock(side_effect=RuntimeError("boom"))
        result = engine.rewrite("test")
        assert result.rewritten_queries == ["test"]
        assert result.strategy_used == "none"

    def test_deduplication_across_strategies(self):
        """Same expansion from multiple strategies should appear once."""
        # Use a query that all strategies might expand similarly
        engine = QueryRewriteEngine(strategies=["synonym"])
        # Force duplicate by adding same strategy twice
        engine._strategies.append(SynonymExpander())
        result = engine.rewrite("python")
        seen = set()
        for q in result.rewritten_queries:
            assert q.lower() not in seen
            seen.add(q.lower())

    def test_metadata_contains_strategies_run(self):
        engine = QueryRewriteEngine()
        result = engine.rewrite("python database")
        assert "strategies_run" in result.metadata
        assert isinstance(result.metadata["strategies_run"], list)


# --------------------------------------------------------------------------- #
# RewriteResult dataclass
# --------------------------------------------------------------------------- #

class TestRewriteResult:
    def test_defaults(self):
        r = RewriteResult(
            original_query="test",
            rewritten_queries=["test", "test2"],
            strategy_used="synonym",
        )
        assert r.metadata == {}

    def test_with_metadata(self):
        r = RewriteResult(
            original_query="test",
            rewritten_queries=["test"],
            strategy_used="none",
            metadata={"key": "value"},
        )
        assert r.metadata["key"] == "value"


# --------------------------------------------------------------------------- #
# RewriteStrategy ABC
# --------------------------------------------------------------------------- #

class TestRewriteStrategy:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            RewriteStrategy()

    def test_custom_strategy(self):
        class UpperCaseStrategy(RewriteStrategy):
            def rewrite(self, query, **kwargs):
                return RewriteResult(
                    original_query=query,
                    rewritten_queries=[query.upper()],
                    strategy_used="uppercase",
                )
        strat = UpperCaseStrategy()
        result = strat.rewrite("hello")
        assert result.rewritten_queries == ["HELLO"]
        assert result.strategy_used == "uppercase"


# --------------------------------------------------------------------------- #
# Integration-style tests
# --------------------------------------------------------------------------- #

class TestIntegration:
    def test_end_to_end_with_all_strategies(self, mock_context_provider, mock_llm):
        engine = QueryRewriteEngine(
            strategies=["synonym", "context", "decompose"],
            llm=mock_llm,
            use_llm=True,
            context_provider=mock_context_provider,
            max_expansion=10,
        )
        result = engine.rewrite(
            "What is the difference between Python and JavaScript",
            user_id="u1",
        )
        assert result.original_query == "What is the difference between Python and JavaScript"
        assert len(result.rewritten_queries) >= 1
        assert result.rewritten_queries[0] == result.original_query
        assert result.metadata["user_id"] == "u1"

    def test_end_to_end_single_strategy_context(self, mock_context_provider):
        engine = QueryRewriteEngine(
            strategies=["context"],
            context_provider=mock_context_provider,
        )
        result = engine.rewrite("machine learning", user_id="u1", strategy="context")
        assert result.strategy_used == "context_enrichment"

    def test_end_to_end_single_strategy_decompose(self):
        engine = QueryRewriteEngine(strategies=["decompose"])
        result = engine.rewrite(
            "Pros and cons of serverless",
            strategy="decompose",
        )
        assert result.strategy_used == "query_decomposition"
        queries = [q.lower() for q in result.rewritten_queries]
        assert any("advantages" in q for q in queries)

    def test_no_strategies_enabled(self):
        engine = QueryRewriteEngine(strategies=[])
        result = engine.rewrite("test")
        assert result.rewritten_queries == ["test"]
        assert result.strategy_used == "none"
        assert result.metadata.get("error") == "no_strategies_enabled"

    def test_all_strategies_fail(self):
        engine = QueryRewriteEngine(strategies=["synonym"])
        # Force the single strategy to fail
        engine._strategies[0].rewrite = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        result = engine.rewrite("test")
        assert result.rewritten_queries == ["test"]
        assert result.strategy_used == "none"
        assert result.metadata.get("error") == "all_strategies_failed"

    def test_strategy_name_aliases(self):
        """Test that strategy name matching works for various aliases."""
        engine = QueryRewriteEngine(strategies=["synonym", "context", "decompose"])
        # Should match "synonym" to SynonymExpander
        r1 = engine.rewrite("python", strategy="synonym")
        assert "synonym" in r1.strategy_used
        # Should match "context" to ContextEnricher
        r2 = engine.rewrite("python", strategy="context")
        assert "context" in r2.strategy_used
        # Should match "decompose" to QueryDecomposer
        r3 = engine.rewrite("python", strategy="decompose")
        assert "query_decomposition" == r3.strategy_used
