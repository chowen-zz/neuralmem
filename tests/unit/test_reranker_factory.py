"""Tests for RerankerFactory and all reranker backends (mock-based)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuralmem.retrieval.reranker_factory import RerankerFactory

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_registry():
    """Ensure each test starts with a clean registry."""
    RerankerFactory.clear()
    yield
    RerankerFactory.clear()


class DummyReranker:
    """Minimal reranker for factory tests."""

    def __init__(self, api_key: str = "", **kwargs):
        self.api_key = api_key
        self.kwargs = kwargs

    def rerank(self, query, documents, top_k=5):
        return [(i, 0.9 - i * 0.1) for i in range(min(top_k, len(documents)))]


# ── RerankerFactory tests ───────────────────────────────────────────────


class TestRerankerFactory:

    def test_register_and_create(self):
        RerankerFactory.register("dummy", DummyReranker)
        r = RerankerFactory.create("dummy", api_key="test-key")
        assert isinstance(r, DummyReranker)
        assert r.api_key == "test-key"

    def test_create_is_case_insensitive(self):
        RerankerFactory.register("Dummy", DummyReranker)
        r = RerankerFactory.create("DUMMY")
        assert isinstance(r, DummyReranker)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown reranker"):
            RerankerFactory.create("nonexistent")

    def test_list_available_empty(self):
        assert RerankerFactory.list_available() == []

    def test_list_available_returns_sorted(self):
        RerankerFactory.register("zeta", DummyReranker)
        RerankerFactory.register("alpha", DummyReranker)
        assert RerankerFactory.list_available() == ["alpha", "zeta"]

    def test_register_overwrites(self):
        RerankerFactory.register("x", DummyReranker)
        RerankerFactory.register("x", DummyReranker)
        assert RerankerFactory.list_available() == ["x"]

    def test_clear_removes_all(self):
        RerankerFactory.register("a", DummyReranker)
        RerankerFactory.register("b", DummyReranker)
        RerankerFactory.clear()
        assert RerankerFactory.list_available() == []

    def test_create_passes_kwargs(self):
        RerankerFactory.register("d", DummyReranker)
        r = RerankerFactory.create("d", api_key="k", extra="val")
        assert r.kwargs.get("extra") == "val"

    def test_dummy_rerank_returns_tuples(self):
        r = DummyReranker()
        result = r.rerank("query", ["a", "b", "c"], top_k=2)
        assert len(result) == 2
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

    def test_create_error_message_lists_available(self):
        RerankerFactory.register("a", DummyReranker)
        with pytest.raises(ValueError, match="a"):
            RerankerFactory.create("missing")


# ── CohereReranker tests ───────────────────────────────────────────────


class TestCohereReranker:

    def test_import_error_when_cohere_missing(self):
        with patch.dict("sys.modules", {"cohere": None}):
            from neuralmem.retrieval.cohere_reranker import CohereReranker

            with pytest.raises(ImportError, match="cohere is required"):
                CohereReranker(api_key="test")

    def test_rerank_empty_documents(self):
        mock_cohere = MagicMock()
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from neuralmem.retrieval.cohere_reranker import CohereReranker

            r = CohereReranker(api_key="k")
            result = r.rerank("query", [], top_k=5)
            assert result == []

    def test_rerank_returns_correct_format(self):
        mock_cohere = MagicMock()
        mock_result_item = MagicMock()
        mock_result_item.index = 1
        mock_result_item.relevance_score = 0.95
        mock_result_item2 = MagicMock()
        mock_result_item2.index = 0
        mock_result_item2.relevance_score = 0.5

        mock_response = MagicMock()
        mock_response.results = [mock_result_item, mock_result_item2]
        mock_cohere.Client.return_value.rerank.return_value = (
            mock_response
        )

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from neuralmem.retrieval.cohere_reranker import CohereReranker

            r = CohereReranker(api_key="k")
            result = r.rerank("query", ["doc0", "doc1"], top_k=2)

        assert result == [(1, 0.95), (0, 0.5)]

    def test_rerank_respects_top_k(self):
        mock_cohere = MagicMock()
        items = []
        for i in range(3):
            item = MagicMock()
            item.index = i
            item.relevance_score = 0.9 - i * 0.1
            items.append(item)

        mock_response = MagicMock()
        mock_response.results = items[:2]
        mock_cohere.Client.return_value.rerank.return_value = (
            mock_response
        )

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from neuralmem.retrieval.cohere_reranker import CohereReranker

            r = CohereReranker(api_key="k", model="rerank-v3")
            result = r.rerank("q", ["a", "b", "c"], top_k=2)

        assert len(result) == 2


# ── HuggingFaceReranker tests ──────────────────────────────────────────


class TestHuggingFaceReranker:

    def test_import_error_when_hf_missing(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            from neuralmem.retrieval.huggingface_reranker import (
                HuggingFaceReranker,
            )

            with pytest.raises(ImportError, match="huggingface_hub"):
                HuggingFaceReranker(api_key="hf_test")

    def test_rerank_empty_documents(self):
        mock_hf = MagicMock()
        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            from neuralmem.retrieval.huggingface_reranker import (
                HuggingFaceReranker,
            )

            r = HuggingFaceReranker(api_key="hf_test")
            result = r.rerank("query", [], top_k=5)
            assert result == []

    def test_rerank_via_feature_extraction(self):
        mock_hf = MagicMock()
        # Return embeddings: query=[1,0], doc0=[1,0], doc1=[0,1]
        mock_client = MagicMock()
        mock_client.feature_extraction.side_effect = [
            [[1.0, 0.0]],  # query
            [[1.0, 0.0]],  # doc0 (same direction -> high score)
            [[0.0, 1.0]],  # doc1 (orthogonal -> 0)
        ]
        mock_hf.InferenceClient.return_value = mock_client

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            from neuralmem.retrieval.huggingface_reranker import (
                HuggingFaceReranker,
            )

            r = HuggingFaceReranker(api_key="hf_test")
            result = r.rerank("query", ["doc0", "doc1"], top_k=2)

        assert len(result) == 2
        # doc0 (index 0) should rank higher
        assert result[0][0] == 0
        assert result[0][1] > result[1][1]


# ── LLMReranker tests ──────────────────────────────────────────────────


class TestLLMReranker:

    def test_init_defaults(self):
        from neuralmem.retrieval.llm_reranker import LLMReranker

        r = LLMReranker()
        assert r._provider == "ollama"
        assert r._model == "llama3"

    def test_init_openai_default_model(self):
        from neuralmem.retrieval.llm_reranker import LLMReranker

        r = LLMReranker(provider="openai", api_key="sk-test")
        assert r._model == "gpt-4o-mini"

    def test_init_custom_model(self):
        from neuralmem.retrieval.llm_reranker import LLMReranker

        r = LLMReranker(provider="openai", model="gpt-4")
        assert r._model == "gpt-4"

    def test_parse_score_valid(self):
        from neuralmem.retrieval.llm_reranker import LLMReranker

        assert LLMReranker._parse_score("0.85") == pytest.approx(0.85)
        assert LLMReranker._parse_score("1") == pytest.approx(1.0)
        assert LLMReranker._parse_score("0") == pytest.approx(0.0)

    def test_parse_score_clamped(self):
        from neuralmem.retrieval.llm_reranker import LLMReranker

        assert LLMReranker._parse_score("1.5") == 1.0
        # "-0.3" matches "0.3" via regex, then clamped normally
        assert LLMReranker._parse_score("-0.3") == 0.3

    def test_parse_score_no_number(self):
        from neuralmem.retrieval.llm_reranker import LLMReranker

        assert LLMReranker._parse_score("no number here") == 0.0

    def test_parse_score_embedded(self):
        from neuralmem.retrieval.llm_reranker import LLMReranker

        assert LLMReranker._parse_score("The score is 0.72 out of 1") == pytest.approx(0.72)

    def test_rerank_empty_documents(self):
        from neuralmem.retrieval.llm_reranker import LLMReranker

        r = LLMReranker()
        assert r.rerank("query", [], top_k=5) == []

    def test_rerank_with_mocked_ollama(self):
        from neuralmem.retrieval.llm_reranker import LLMReranker

        r = LLMReranker(provider="ollama")

        with patch.object(r, "_score_pair", side_effect=[0.9, 0.3, 0.7]):
            result = r.rerank("q", ["a", "b", "c"], top_k=3)

        assert result == [(0, 0.9), (2, 0.7), (1, 0.3)]

    def test_rerank_top_k_limits_results(self):
        from neuralmem.retrieval.llm_reranker import LLMReranker

        r = LLMReranker()
        with patch.object(r, "_score_pair", side_effect=[0.5, 0.9, 0.1]):
            result = r.rerank("q", ["a", "b", "c"], top_k=1)

        assert len(result) == 1
        assert result[0] == (1, 0.9)

    def test_score_pair_returns_zero_on_error(self):
        from neuralmem.retrieval.llm_reranker import LLMReranker

        r = LLMReranker(provider="ollama")
        with patch.object(r, "_call_llm", side_effect=RuntimeError("fail")):
            score = r._score_pair("query", "doc")

        assert score == 0.0

    def test_unsupported_provider_raises(self):
        from neuralmem.retrieval.llm_reranker import LLMReranker

        r = LLMReranker(provider="unknown")
        with pytest.raises(ValueError, match="Unsupported provider"):
            r._call_llm("test")
