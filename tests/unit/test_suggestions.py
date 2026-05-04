"""Tests for SuggestionEngine."""
from __future__ import annotations

import pytest

from neuralmem.assistant.suggestions import (
    Suggestion,
    SuggestionEngine,
    SuggestionType,
)


# ------------------------------------------------------------------
# Mocks
# ------------------------------------------------------------------

class MockLLM:
    """Deterministic mock LLM for suggestion tests."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self._calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self._calls.append(prompt)
        for key, val in self._responses.items():
            if key in prompt:
                return val
        return f"MOCK: {prompt[:40]}..."


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    return MockLLM(responses={
        "autocomplete": " is a powerful programming language.",
        "rewrite": (
            "1. The code quality is poor.\n"
            "2. The implementation needs improvement.\n"
            "3. There are issues with the current approach."
        ),
        "expand": (
            "1. Add more examples and edge cases.\n"
            "2. Include performance benchmarks and comparisons.\n"
            "3. Provide a step-by-step tutorial."
        ),
        "summarize": (
            "1. Brief overview of key points.\n"
            "2. Main takeaway in one sentence.\n"
            "3. Executive summary for stakeholders."
        ),
        "style_transfer": "Hark! The code doth require refinement, good sir.",
    })


@pytest.fixture
def engine(mock_llm):
    return SuggestionEngine(
        llm_caller=mock_llm,
        max_suggestions=3,
        min_confidence=0.3,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestSuggestionEngineAutocomplete:
    def test_autocomplete_basic(self, engine):
        suggestions = engine.autocomplete("Python")
        assert len(suggestions) > 0
        assert all(s.suggestion_type == SuggestionType.AUTOCOMPLETE for s in suggestions)
        assert all(s.confidence >= 0.3 for s in suggestions)

    def test_autocomplete_with_context(self, engine, mock_llm):
        suggestions = engine.autocomplete("Python", context="backend development")
        assert len(suggestions) > 0
        # Context should appear in prompt
        assert any("backend development" in call for call in mock_llm._calls)

    def test_autocomplete_no_llm_raises(self):
        engine = SuggestionEngine(llm_caller=None)
        with pytest.raises(RuntimeError, match="No LLM caller"):
            engine.autocomplete("hello")


class TestSuggestionEngineRewrite:
    def test_rewrite_suggestions_basic(self, engine):
        suggestions = engine.rewrite_suggestions("The code is bad.")
        assert len(suggestions) > 0
        assert all(s.suggestion_type == SuggestionType.REWRITE for s in suggestions)

    def test_rewrite_suggestions_count(self, engine):
        suggestions = engine.rewrite_suggestions("Text.", count=2)
        assert len(suggestions) <= 3  # max_suggestions cap

    def test_rewrite_suggestions_with_context(self, engine, mock_llm):
        engine.rewrite_suggestions("Text.", context="technical review")
        assert any("technical review" in call for call in mock_llm._calls)


class TestSuggestionEngineExpand:
    def test_expand_suggestions_basic(self, engine):
        suggestions = engine.expand_suggestions("Short text.")
        assert len(suggestions) > 0
        assert all(s.suggestion_type == SuggestionType.EXPAND for s in suggestions)

    def test_expand_suggestions_count(self, engine):
        suggestions = engine.expand_suggestions("Text.", count=1)
        assert len(suggestions) <= 3


class TestSuggestionEngineSummarize:
    def test_summarize_suggestions_basic(self, engine):
        suggestions = engine.summarize_suggestions("Long text here...")
        assert len(suggestions) > 0
        assert all(s.suggestion_type == SuggestionType.SUMMARIZE for s in suggestions)

    def test_summarize_suggestions_count(self, engine):
        suggestions = engine.summarize_suggestions("Text.", count=2)
        assert len(suggestions) <= 3


class TestSuggestionEngineStyleTransfer:
    def test_style_transfer_basic(self, engine):
        suggestions = engine.style_transfer("Hello world.", target_style="Shakespeare")
        assert len(suggestions) > 0
        assert all(s.suggestion_type == SuggestionType.STYLE_TRANSFER for s in suggestions)

    def test_style_transfer_with_context(self, engine, mock_llm):
        engine.style_transfer("Hello.", target_style="poetic", context="greeting card")
        assert any("poetic" in call for call in mock_llm._calls)
        assert any("greeting card" in call for call in mock_llm._calls)


class TestSuggestionParsing:
    def test_parse_numbered_suggestions(self):
        raw = (
            "1. First suggestion here.\n"
            "2. Second suggestion here.\n"
            "3. Third suggestion here."
        )
        suggestions = SuggestionEngine._parse_suggestions(raw, SuggestionType.REWRITE)
        assert len(suggestions) == 3
        assert suggestions[0].text == "First suggestion here."
        assert suggestions[1].text == "Second suggestion here."

    def test_parse_bullet_suggestions(self):
        raw = (
            "- First bullet\n"
            "- Second bullet\n"
            "- Third bullet"
        )
        suggestions = SuggestionEngine._parse_suggestions(raw, SuggestionType.EXPAND)
        assert len(suggestions) == 3
        assert all(s.text.startswith("First") or s.text.startswith("Second") or s.text.startswith("Third") for s in suggestions)

    def test_parse_single_suggestion(self):
        raw = "Just one suggestion."
        suggestions = SuggestionEngine._parse_suggestions(raw, SuggestionType.AUTOCOMPLETE)
        assert len(suggestions) == 1
        assert suggestions[0].text == "Just one suggestion."

    def test_parse_multiline_suggestion(self):
        raw = (
            "1. First line\n"
            "   continuation of first\n"
            "2. Second line"
        )
        suggestions = SuggestionEngine._parse_suggestions(raw, SuggestionType.REWRITE)
        assert len(suggestions) == 2
        assert "continuation" in suggestions[0].text

    def test_parse_empty(self):
        suggestions = SuggestionEngine._parse_suggestions("", SuggestionType.SUMMARIZE)
        assert suggestions == []

    def test_parse_parenthesis_numbers(self):
        raw = "1) First\n2) Second"
        suggestions = SuggestionEngine._parse_suggestions(raw, SuggestionType.REWRITE)
        assert len(suggestions) == 2


class TestSuggestionFiltering:
    def test_filter_by_confidence(self):
        engine = SuggestionEngine(min_confidence=0.8)
        suggestions = [
            Suggestion(SuggestionType.REWRITE, "good", confidence=0.9),
            Suggestion(SuggestionType.REWRITE, "low", confidence=0.2),
        ]
        filtered = engine._filter_and_rank(suggestions)
        assert len(filtered) == 1
        assert filtered[0].text == "good"

    def test_deduplicate(self):
        engine = SuggestionEngine()
        suggestions = [
            Suggestion(SuggestionType.REWRITE, "same text", confidence=0.7),
            Suggestion(SuggestionType.REWRITE, "Same Text", confidence=0.7),
            Suggestion(SuggestionType.REWRITE, "different", confidence=0.7),
        ]
        filtered = engine._filter_and_rank(suggestions)
        assert len(filtered) == 2

    def test_max_suggestions_cap(self):
        engine = SuggestionEngine(max_suggestions=2)
        suggestions = [
            Suggestion(SuggestionType.REWRITE, f"s{i}", confidence=0.7)
            for i in range(5)
        ]
        filtered = engine._filter_and_rank(suggestions)
        assert len(filtered) == 2


class TestSuggestionDataclass:
    def test_suggestion_creation(self):
        s = Suggestion(
            suggestion_type=SuggestionType.AUTOCOMPLETE,
            text="test",
            confidence=0.8,
            description="desc",
            metadata={"key": "val"},
        )
        assert s.suggestion_type == SuggestionType.AUTOCOMPLETE
        assert s.text == "test"
        assert s.confidence == 0.8
        assert s.description == "desc"
        assert s.metadata == {"key": "val"}

    def test_suggestion_defaults(self):
        s = Suggestion(suggestion_type=SuggestionType.REWRITE, text="test")
        assert s.confidence == 0.5
        assert s.description == ""
        assert s.metadata == {}
