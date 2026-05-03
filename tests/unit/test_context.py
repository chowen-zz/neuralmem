"""Tests for the Context Composer module (16 tests)."""

from __future__ import annotations

import pytest

from neuralmem.context import ComposedContext, ContextComposer, ContextSource

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def composer() -> ContextComposer:
    """Default composer with 100-token budget."""
    return ContextComposer(token_budget=100, tokenizer="chars")


@pytest.fixture
def large_composer() -> ContextComposer:
    """Composer with a generous budget."""
    return ContextComposer(token_budget=4000, tokenizer="chars")


# ------------------------------------------------------------------ #
# add_source / get_sources
# ------------------------------------------------------------------ #


class TestAddSource:
    def test_add_source(self, composer: ContextComposer):
        """Sources accumulate correctly."""
        composer.add_source(ContextSource.memory, "Memory content A")
        composer.add_source(ContextSource.memory, "Memory content B")
        composer.add_source(ContextSource.web, "Web content")

        sources = composer.get_sources()
        assert ContextSource.memory in sources
        assert len(sources[ContextSource.memory]) == 2
        assert sources[ContextSource.memory][0] == "Memory content A"
        assert sources[ContextSource.memory][1] == "Memory content B"
        assert ContextSource.web in sources
        assert len(sources[ContextSource.web]) == 1


class TestGetSources:
    def test_get_sources(self, composer: ContextComposer):
        """get_sources returns correct structure."""
        composer.add_source(ContextSource.repo, "repo data")
        composer.add_source(ContextSource.custom, "custom data")
        sources = composer.get_sources()
        assert isinstance(sources, dict)
        assert ContextSource.repo in sources
        assert ContextSource.custom in sources
        assert sources[ContextSource.repo] == ["repo data"]
        assert sources[ContextSource.custom] == ["custom data"]

    def test_get_sources_empty(self, composer: ContextComposer):
        """get_sources on empty composer returns empty dict."""
        assert composer.get_sources() == {}


# ------------------------------------------------------------------ #
# compose
# ------------------------------------------------------------------ #


class TestCompose:
    def test_compose_basic(self, composer: ContextComposer):
        """compose returns ComposedContext with all fields."""
        composer.add_source(ContextSource.memory, "Hello world.")
        result = composer.compose(query="test query")

        assert isinstance(result, ComposedContext)
        assert result.query == "test query"
        assert isinstance(result.sources, dict)
        assert isinstance(result.composed, str)
        assert isinstance(result.token_count, int)
        assert isinstance(result.confidence, float)
        assert isinstance(result.metadata, dict)

    def test_compose_with_query(self, composer: ContextComposer):
        """Query is stored in result."""
        composer.add_source(ContextSource.memory, "data")
        result = composer.compose(query="What is AI?")
        assert result.query == "What is AI?"

    def test_empty_compose(self, composer: ContextComposer):
        """No sources returns empty ComposedContext."""
        result = composer.compose(query="empty")
        assert result.composed == ""
        assert result.token_count == 0
        assert result.confidence == 0.0
        assert result.sources == {}

    def test_compose_within_budget(self):
        """Composed text fits within token_budget."""
        budget = 200
        comp = ContextComposer(token_budget=budget, tokenizer="words")
        comp.add_source(ContextSource.memory, "word " * 500)
        result = comp.compose()
        assert result.token_count <= budget

    def test_compose_truncates(self):
        """Excess content is truncated."""
        comp = ContextComposer(token_budget=50, tokenizer="chars")
        long_text = "This is a sentence. " * 200  # ~4000 chars -> ~1000 tokens
        comp.add_source(ContextSource.memory, long_text)
        result = comp.compose()
        assert len(result.composed) < len(long_text)
        assert result.token_count <= 50

    def test_token_budget_respected(self):
        """Output never exceeds budget significantly."""
        for budget in [50, 100, 500, 1000]:
            comp = ContextComposer(token_budget=budget, tokenizer="chars")
            comp.add_source(ContextSource.memory, "x " * 5000)
            comp.add_source(ContextSource.web, "y " * 5000)
            comp.add_source(ContextSource.repo, "z " * 5000)
            result = comp.compose()
            # Allow small overshoot (1 token) from rounding
            assert result.token_count <= budget + 1, (
                f"budget={budget}, got {result.token_count}"
            )


# ------------------------------------------------------------------ #
# Priority & confidence
# ------------------------------------------------------------------ #


class TestPriority:
    def test_priority_order(self):
        """Memory content preserved over web content when budget is tight."""
        comp = ContextComposer(token_budget=30, tokenizer="chars")
        memory_text = "Important memory fact. " * 5
        web_text = "Random web article. " * 5
        comp.add_source(ContextSource.memory, memory_text)
        comp.add_source(ContextSource.web, web_text)
        result = comp.compose()

        # Memory content should appear in the composed output
        assert "Important memory fact" in result.composed
        # With a tight budget, web may be truncated more aggressively

    def test_confidence_affects_budget(self):
        """Higher confidence sources get more budget."""
        comp = ContextComposer(token_budget=100, tokenizer="words")
        comp.add_source(ContextSource.memory, "high conf. " * 100, confidence=1.0)
        comp.add_source(ContextSource.web, "low conf. " * 100, confidence=0.1)
        result = comp.compose()

        # Memory (high confidence) should retain more content than web (low)
        mem_text = result.sources.get(ContextSource.memory, [""])[0]
        web_text = result.sources.get(ContextSource.web, [""])[0]
        assert comp.estimate_tokens(mem_text) > comp.estimate_tokens(web_text)


# ------------------------------------------------------------------ #
# Token estimation
# ------------------------------------------------------------------ #


class TestEstimateTokens:
    def test_estimate_tokens_chars(self):
        """Char-based estimation: len(text) // 4."""
        comp = ContextComposer(tokenizer="chars")
        assert comp.estimate_tokens("") == 0
        assert comp.estimate_tokens("abcd") == 1
        assert comp.estimate_tokens("abcdefgh") == 2
        assert comp.estimate_tokens("x" * 100) == 25

    def test_estimate_tokens_words(self):
        """Word-based estimation: word count."""
        comp = ContextComposer(tokenizer="words")
        assert comp.estimate_tokens("") == 0
        assert comp.estimate_tokens("hello") == 1
        assert comp.estimate_tokens("hello world") == 2
        assert comp.estimate_tokens("one two three four five") == 5

    def test_custom_tokenizer(self):
        """Custom tokenizer config works correctly."""
        comp_chars = ContextComposer(tokenizer="chars")
        comp_words = ContextComposer(tokenizer="words")
        text = "hello world test"  # 16 chars, 3 words
        assert comp_chars.estimate_tokens(text) == 16 // 4  # 4
        assert comp_words.estimate_tokens(text) == 3


# ------------------------------------------------------------------ #
# clear
# ------------------------------------------------------------------ #


class TestClear:
    def test_clear(self, composer: ContextComposer):
        """clear() removes all sources."""
        composer.add_source(ContextSource.memory, "data1")
        composer.add_source(ContextSource.web, "data2")
        assert len(composer.get_sources()) == 2
        composer.clear()
        assert composer.get_sources() == {}


# ------------------------------------------------------------------ #
# Multiple sources & truncation behaviour
# ------------------------------------------------------------------ #


class TestMultipleSources:
    def test_multiple_sources_same_type(self):
        """Multiple entries for the same source type are joined."""
        comp = ContextComposer(token_budget=1000, tokenizer="words")
        comp.add_source(ContextSource.memory, "First memory.")
        comp.add_source(ContextSource.memory, "Second memory.")
        comp.add_source(ContextSource.memory, "Third memory.")
        result = comp.compose()
        # All three should be present in the composed output
        assert "First memory" in result.composed
        assert "Second memory" in result.composed
        assert "Third memory" in result.composed

    def test_sentence_boundary_truncation(self):
        """Truncation doesn't cut mid-sentence when possible."""
        comp = ContextComposer(token_budget=30, tokenizer="chars")
        # ~25 chars per sentence -> ~6 tokens each
        text = "First sentence is here. Second sentence is here. Third sentence is here."
        comp.add_source(ContextSource.memory, text)
        result = comp.compose()

        # The composed output should not end mid-word when possible
        # Check it ends at a sentence boundary or is the full text
        composed_body = result.composed
        # Find the memory section content (after the [memory] tag)
        if "[memory]" in composed_body:
            parts = composed_body.split("\n", 2)
            content = parts[-1] if len(parts) >= 2 else composed_body
        else:
            content = composed_body
        content = content.strip()
        if content and len(content) < len(text):
            # If truncated, should end at a sentence boundary
            assert content[-1] in ".!?", (
                f"Truncation didn't end at sentence boundary: ...{content[-20:]}"
            )
