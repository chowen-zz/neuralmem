"""Comprehensive tests for WritingAssistant."""
from __future__ import annotations

import pytest

from neuralmem.assistant.assistant import LLMCaller, WriteResult, WritingAssistant
from neuralmem.assistant.context import ContextConfig, ContextInjector
from neuralmem.assistant.suggestions import SuggestionEngine
from neuralmem.assistant.templates import TemplateManager, WritingTemplate
from neuralmem.core.types import Memory, MemoryType, SearchResult


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

class MockLLM(LLMCaller):
    """Deterministic mock LLM for tests."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self._calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self._calls.append(prompt)
        # Return a keyed response if available, else echo
        for key, val in self._responses.items():
            if key in prompt:
                return val
        return f"MOCK_RESPONSE: {prompt[:50]}..."


class MockRetriever:
    """Deterministic mock memory retriever."""

    def __init__(self, memories: dict[str, Memory]) -> None:
        self._memories = memories

    def vector_search(self, vector, user_id=None, memory_types=None, limit=10):
        return [(m.id, 0.9) for m in self._memories.values() if m.user_id == user_id][:limit]

    def keyword_search(self, query, user_id=None, memory_types=None, limit=10):
        return [(m.id, 0.8) for m in self._memories.values()
                if m.user_id == user_id and query.lower() in m.content.lower()][:limit]

    def get_memory(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)


class MockEmbedder:
    def encode_one(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def sample_memories():
    return {
        "mem-1": Memory(
            id="mem-1",
            content="User prefers TypeScript for frontend",
            memory_type=MemoryType.PREFERENCE,
            user_id="u1",
            embedding=[0.1, 0.2, 0.3, 0.4],
        ),
        "mem-2": Memory(
            id="mem-2",
            content="User deploys to AWS",
            memory_type=MemoryType.FACT,
            user_id="u1",
            embedding=[0.2, 0.3, 0.4, 0.5],
        ),
        "mem-3": Memory(
            id="mem-3",
            content="User likes Python",
            memory_type=MemoryType.PREFERENCE,
            user_id="u2",
            embedding=[0.5, 0.4, 0.3, 0.2],
        ),
    }


@pytest.fixture
def context_injector(sample_memories):
    retriever = MockRetriever(sample_memories)
    embedder = MockEmbedder()
    config = ContextConfig(max_memories=3, min_score=0.0)
    return ContextInjector(retriever=retriever, embedder=embedder, config=config)


@pytest.fixture
def mock_llm():
    return MockLLM(responses={
        "write": "Generated blog post about TypeScript.",
        "rewrite": "Rewritten text with improved clarity.",
        "expand": "Expanded version with more detail and examples.",
        "summarize": "Short summary.",
    })


@pytest.fixture
def assistant(context_injector, mock_llm):
    return WritingAssistant(
        context_injector=context_injector,
        suggestion_engine=SuggestionEngine(llm_caller=mock_llm),
        template_manager=TemplateManager(),
        llm_caller=mock_llm,
        max_context_memories=3,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestWritingAssistantWrite:
    def test_write_basic(self, assistant, mock_llm):
        result = assistant.write("write a blog post about TypeScript", user_id="u1")
        assert isinstance(result, WriteResult)
        assert result.operation == "write"
        assert result.text
        assert result.context_memories_used
        # Should have retrieved u1 memories
        assert all(m.user_id == "u1" for m in result.context_memories_used)

    def test_write_with_template(self, assistant):
        result = assistant.write(
            "announce the new feature",
            user_id="u1",
            template_name="blog_post",
            tone="excited",
            audience="developers",
        )
        assert result.template_used == "blog_post"
        assert result.text

    def test_write_no_user_id(self, assistant):
        result = assistant.write("hello world")
        assert result.operation == "write"
        assert result.context_memories_used == []

    def test_write_no_llm_raises(self):
        assistant = WritingAssistant(llm_caller=None)
        with pytest.raises(RuntimeError, match="No LLM caller"):
            assistant.write("hello")

    def test_write_with_style(self, assistant, mock_llm):
        result = assistant.write("greeting", user_id="u1", style="formal")
        assert result.operation == "write"
        # Style hint should appear in prompt passed to LLM
        assert any("formal" in call for call in mock_llm._calls)


class TestWritingAssistantRewrite:
    def test_rewrite_basic(self, assistant):
        result = assistant.rewrite("The code is bad.", user_id="u1")
        assert result.operation == "rewrite"
        assert result.text
        assert result.context_memories_used

    def test_rewrite_with_instruction(self, assistant, mock_llm):
        result = assistant.rewrite(
            "The code is bad.",
            instruction="make it more positive",
            user_id="u1",
        )
        assert result.operation == "rewrite"
        # Instruction should be in prompt
        assert any("make it more positive" in call for call in mock_llm._calls)

    def test_rewrite_with_style(self, assistant, mock_llm):
        result = assistant.rewrite("Hello.", style="poetic", user_id="u1")
        assert any("poetic" in call for call in mock_llm._calls)

    def test_rewrite_no_user_id(self, assistant):
        result = assistant.rewrite("Hello.")
        assert result.context_memories_used == []


class TestWritingAssistantExpand:
    def test_expand_basic(self, assistant):
        result = assistant.expand("Short text.", user_id="u1")
        assert result.operation == "expand"
        assert result.text
        assert result.context_memories_used

    def test_expand_with_target_length(self, assistant, mock_llm):
        result = assistant.expand("Short.", target_length=200, user_id="u1")
        assert any("200" in call for call in mock_llm._calls)

    def test_expand_no_user_id(self, assistant):
        result = assistant.expand("Short.")
        assert result.context_memories_used == []


class TestWritingAssistantSummarize:
    def test_summarize_basic(self, assistant):
        result = assistant.summarize("Long text here...", user_id="u1")
        assert result.operation == "summarize"
        assert result.text
        assert result.context_memories_used

    def test_summarize_with_max_words(self, assistant, mock_llm):
        result = assistant.summarize("Long text...", max_words=50, user_id="u1")
        assert any("50" in call for call in mock_llm._calls)

    def test_summarize_no_user_id(self, assistant):
        result = assistant.summarize("Long text...")
        assert result.context_memories_used == []


class TestWritingAssistantIntegration:
    def test_all_operations_use_same_context_pipeline(self, assistant, mock_llm):
        write_res = assistant.write("topic", user_id="u1")
        rewrite_res = assistant.rewrite("text", user_id="u1")
        expand_res = assistant.expand("text", user_id="u1")
        summarize_res = assistant.summarize("text", user_id="u1")

        for res in (write_res, rewrite_res, expand_res, summarize_res):
            assert res.context_memories_used
            assert all(m.user_id == "u1" for m in res.context_memories_used)

    def test_template_manager_integration(self, assistant):
        result = assistant.write(
            "deploy notes for v2.2",
            user_id="u1",
            template_name="release_notes",
        )
        assert result.template_used == "release_notes"

    def test_suggestion_engine_available(self, assistant):
        assert assistant._suggestions is not None

    def test_context_injector_none(self, mock_llm):
        assistant = WritingAssistant(llm_caller=mock_llm, context_injector=None)
        result = assistant.write("hello", user_id="u1")
        assert result.context_memories_used == []
