"""Tests for ReasoningChain — multi-step memory retrieval."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from neuralmem.core.types import Memory, SearchResult
from neuralmem.retrieval.reasoning import ReasoningChain, ReasoningResult

# --- Helpers ---

def _make_memory(
    mem_id: str = "mem-1",
    content: str = "test content",
    entity_ids: tuple[str, ...] = (),
    importance: float = 0.5,
    created_at: datetime | None = None,
    is_active: bool = True,
) -> Memory:
    """Create a Memory with specified attributes."""
    return Memory(
        id=mem_id,
        content=content,
        entity_ids=entity_ids,
        importance=importance,
        created_at=created_at or datetime.now(timezone.utc),
        is_active=is_active,
    )


def _make_search_result(
    mem_id: str = "mem-1",
    score: float = 0.8,
    method: str = "semantic",
    content: str = "test",
    entity_ids: tuple[str, ...] = (),
) -> SearchResult:
    """Create a SearchResult."""
    mem = _make_memory(mem_id, content=content, entity_ids=entity_ids)
    return SearchResult(
        memory=mem, score=score, retrieval_method=method
    )


# --- Tests ---

class TestReasoningChain:
    def test_basic_reasoning(self):
        mem = MagicMock()
        results = [_make_search_result("m1", 0.8)]
        mem.recall.return_value = results
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem)
        result = chain.reason("test query")

        assert isinstance(result, ReasoningResult)
        assert len(result.results) > 0
        assert result.total_steps == 4

    def test_returns_reasoning_trace(self):
        mem = MagicMock()
        mem.recall.return_value = []
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem)
        result = chain.reason("test query")

        assert len(result.reasoning_trace) == 4
        step_names = [s.name for s in result.reasoning_trace]
        assert "initial_recall" in step_names
        assert "entity_expansion" in step_names
        assert "dedup_and_rank" in step_names
        assert "confidence_scoring" in step_names

    def test_confidence_scores_populated(self):
        mem = MagicMock()
        results = [_make_search_result("m1", 0.8)]
        mem.recall.return_value = results
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem)
        result = chain.reason("test query")

        assert "m1" in result.confidence_scores
        assert 0.0 <= result.confidence_scores["m1"] <= 1.0

    def test_execution_time_recorded(self):
        mem = MagicMock()
        mem.recall.return_value = []
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem)
        result = chain.reason("test query")

        assert result.execution_time_ms >= 0

    def test_dedup_removes_duplicates(self):
        mem = MagicMock()
        # Two results with same ID but different scores
        r1 = _make_search_result("m1", 0.5)
        r2 = _make_search_result("m1", 0.9)
        mem.recall.side_effect = [[r1], [r2]]
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem)
        result = chain.reason("test")

        # Should dedup — only one m1 in results
        ids = [r.memory.id for r in result.results]
        assert ids.count("m1") == 1

    def test_entity_expansion_calls_graph(self):
        mem = MagicMock()
        results = [_make_search_result("m1", 0.8, entity_ids=("e1",))]
        mem.recall.side_effect = [results, []]
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem)
        chain.reason("test")

        mem.graph.get_neighbors.assert_called_once_with(
            ["e1"], depth=2
        )

    def test_entity_expansion_no_entities(self):
        mem = MagicMock()
        results = [_make_search_result("m1", 0.8, entity_ids=())]
        mem.recall.return_value = results
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem)
        chain.reason("test")

        # No entities → get_neighbors should not be called
        mem.graph.get_neighbors.assert_not_called()

    def test_entity_expansion_graph_error_handled(self):
        mem = MagicMock()
        results = [_make_search_result("m1", 0.8, entity_ids=("e1",))]
        mem.recall.return_value = results
        mem.graph = MagicMock()
        mem.graph.get_neighbors.side_effect = RuntimeError("graph error")

        chain = ReasoningChain(mem)
        result = chain.reason("test")

        # Should not crash
        assert isinstance(result, ReasoningResult)

    def test_no_graph_still_works(self):
        mem = MagicMock()
        mem.recall.return_value = [_make_search_result("m1", 0.8)]
        # No graph attribute
        del mem.graph

        chain = ReasoningChain(mem, graph=None)
        result = chain.reason("test")

        assert isinstance(result, ReasoningResult)

    def test_initial_recall_step_count(self):
        mem = MagicMock()
        r1 = _make_search_result("m1", 0.8)
        r2 = _make_search_result("m2", 0.6)
        mem.recall.return_value = [r1, r2]
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem)
        result = chain.reason("test")

        initial_step = result.reasoning_trace[0]
        assert initial_step.name == "initial_recall"
        assert initial_step.output_count == 2

    def test_higher_confidence_for_initial_hits(self):
        mem = MagicMock()
        initial = [_make_search_result("m1", 0.8)]
        expanded = [_make_search_result("m2", 0.8)]
        mem.recall.side_effect = [initial, expanded]
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem)
        result = chain.reason("test")

        # m1 was in initial results, m2 was not
        conf_m1 = result.confidence_scores.get("m1", 0)
        conf_m2 = result.confidence_scores.get("m2", 0)
        assert conf_m1 >= conf_m2

    def test_importance_boosts_confidence(self):
        mem = MagicMock()
        high_imp = _make_memory("m1", importance=0.9)
        low_imp = _make_memory("m2", importance=0.1)
        mem.recall.return_value = [
            SearchResult(
                memory=high_imp, score=0.8, retrieval_method="test"
            ),
            SearchResult(
                memory=low_imp, score=0.8, retrieval_method="test"
            ),
        ]
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem)
        result = chain.reason("test")

        assert result.confidence_scores["m1"] > result.confidence_scores["m2"]

    def test_entity_richness_boosts_confidence(self):
        mem = MagicMock()
        rich = _make_memory("m1", entity_ids=("e1", "e2", "e3", "e4", "e5"))
        poor = _make_memory("m2", entity_ids=())
        mem.recall.return_value = [
            SearchResult(memory=rich, score=0.8, retrieval_method="test"),
            SearchResult(memory=poor, score=0.8, retrieval_method="test"),
        ]
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem)
        result = chain.reason("test")

        assert result.confidence_scores["m1"] > result.confidence_scores["m2"]

    def test_results_sorted_by_adjusted_score(self):
        mem = MagicMock()
        # m1 has lower raw score but higher confidence
        m1 = _make_memory("m1", importance=0.9, entity_ids=("e1", "e2"))
        m2 = _make_memory("m2", importance=0.1, entity_ids=())
        mem.recall.return_value = [
            SearchResult(memory=m1, score=0.6, retrieval_method="semantic"),
            SearchResult(memory=m2, score=0.9, retrieval_method="keyword"),
        ]
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem)
        result = chain.reason("test", min_score=0.0)

        # Results should be sorted by adjusted score (raw * confidence)
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_min_score_filters_results(self):
        mem = MagicMock()
        mem.recall.return_value = [_make_search_result("m1", 0.05)]
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem)
        result = chain.reason("test", min_score=0.9)

        # Very low score result should be filtered out
        assert len(result.results) == 0

    def test_explanation_includes_confidence(self):
        mem = MagicMock()
        mem.recall.return_value = [_make_search_result("m1", 0.8)]
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem)
        result = chain.reason("test")

        if result.results:
            assert "confidence" in result.results[0].explanation

    def test_custom_depth(self):
        mem = MagicMock()
        results = [_make_search_result("m1", 0.8, entity_ids=("e1",))]
        mem.recall.side_effect = [results, []]
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem, max_expansion_depth=3)
        chain.reason("test")

        mem.graph.get_neighbors.assert_called_once_with(
            ["e1"], depth=3
        )

    def test_reasoning_step_details(self):
        mem = MagicMock()
        mem.recall.return_value = []
        mem.graph = MagicMock()
        mem.graph.get_neighbors.return_value = []

        chain = ReasoningChain(mem)
        result = chain.reason("my query")

        initial_step = result.reasoning_trace[0]
        assert initial_step.details["query"] == "my query"
