"""LLMConversationExtractor unit tests — 16 tests."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from neuralmem.extraction.llm_conversation_extractor import (
    LLMConversationExtractor,
)


def _run(coro):
    """Helper to run async in sync tests."""
    return asyncio.run(coro)


@pytest.fixture
def extractor():
    return LLMConversationExtractor(llm_backend="ollama")


SAMPLE_MEMORIES = [
    {
        "content": "User prefers Python for backend development",
        "type": "preference",
        "role": "user",
        "confidence": 0.9,
        "entities": ["Python"],
    },
    {
        "content": "User works at Acme Corp",
        "type": "fact",
        "role": "user",
        "confidence": 0.85,
        "entities": ["Acme Corp"],
    },
]


# ==================== Init ====================


class TestExtractorInit:

    def test_extractor_init_ollama(self):
        ext = LLMConversationExtractor(llm_backend="ollama")
        assert ext._backend == "ollama"

    def test_extractor_init_openai(self):
        ext = LLMConversationExtractor(
            llm_backend="openai", model="gpt-4"
        )
        assert ext._backend == "openai"


# ==================== Format conversation ====================


class TestFormatConversation:

    def test_format_conversation(self, extractor):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = extractor._format_conversation(messages)
        assert "user: Hello" in result
        assert "assistant: Hi there" in result

    def test_format_conversation_empty(self, extractor):
        result = extractor._format_conversation([])
        assert result == ""

    def test_format_conversation_skips_empty(self, extractor):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "  "},
        ]
        result = extractor._format_conversation(messages)
        assert "user: Hello" in result
        assert result.count("\n") == 0  # only one line


# ==================== Extract ====================


class TestExtract:

    def test_extract_returns_list(self, extractor):
        """Mock LLM to return valid JSON."""
        extractor._call_llm = AsyncMock(
            return_value=json.dumps(SAMPLE_MEMORIES)
        )
        messages = [
            {"role": "user", "content": "I prefer Python"},
        ]
        result = _run(extractor.extract(messages))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_extract_with_instructions(self, extractor):
        captured = {}

        async def mock_llm(prompt):
            captured["prompt"] = prompt
            return json.dumps(SAMPLE_MEMORIES)

        extractor._call_llm = mock_llm
        messages = [{"role": "user", "content": "test"}]
        _run(
            extractor.extract(
                messages, instructions="Always extract in English"
            )
        )
        assert "Always extract in English" in captured["prompt"]

    def test_extract_empty_messages(self, extractor):
        result = _run(extractor.extract([]))
        assert result == []

    def test_extract_single_message(self, extractor):
        extractor._call_llm = AsyncMock(
            return_value=json.dumps([SAMPLE_MEMORIES[0]])
        )
        messages = [{"role": "user", "content": "I prefer Python"}]
        result = _run(extractor.extract(messages))
        assert len(result) == 1

    def test_extract_multi_turn(self, extractor):
        extractor._call_llm = AsyncMock(
            return_value=json.dumps(SAMPLE_MEMORIES)
        )
        messages = [
            {"role": "user", "content": "I prefer Python"},
            {
                "role": "assistant",
                "content": "Python is great for backend",
            },
            {"role": "user", "content": "I work at Acme Corp"},
        ]
        result = _run(extractor.extract(messages))
        assert len(result) == 2

    def test_extract_parses_json_response(self, extractor):
        extractor._call_llm = AsyncMock(
            return_value=json.dumps(SAMPLE_MEMORIES)
        )
        messages = [{"role": "user", "content": "test"}]
        result = _run(extractor.extract(messages))
        assert result[0]["content"] == SAMPLE_MEMORIES[0]["content"]

    def test_extract_handles_markdown_wrapper(self, extractor):
        raw = "```json\n" + json.dumps(SAMPLE_MEMORIES) + "\n```"
        extractor._call_llm = AsyncMock(return_value=raw)
        messages = [{"role": "user", "content": "test"}]
        result = _run(extractor.extract(messages))
        assert len(result) == 2

    def test_extract_handles_invalid_json(self, extractor):
        extractor._call_llm = AsyncMock(
            return_value="This is not JSON at all"
        )
        messages = [{"role": "user", "content": "test"}]
        result = _run(extractor.extract(messages))
        assert result == []

    def test_extract_confidence_range(self, extractor):
        memories = [
            {
                "content": "test",
                "type": "fact",
                "role": "user",
                "confidence": 0.75,
                "entities": [],
            }
        ]
        extractor._call_llm = AsyncMock(
            return_value=json.dumps(memories)
        )
        result = _run(
            extractor.extract(
                [{"role": "user", "content": "test"}]
            )
        )
        conf = result[0]["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_extract_entity_extraction(self, extractor):
        extractor._call_llm = AsyncMock(
            return_value=json.dumps(SAMPLE_MEMORIES)
        )
        result = _run(
            extractor.extract(
                [{"role": "user", "content": "test"}]
            )
        )
        assert "Python" in result[0]["entities"]

    def test_extract_type_classification(self, extractor):
        extractor._call_llm = AsyncMock(
            return_value=json.dumps(SAMPLE_MEMORIES)
        )
        result = _run(
            extractor.extract(
                [{"role": "user", "content": "test"}]
            )
        )
        assert result[0]["type"] == "preference"
        assert result[1]["type"] == "fact"

    def test_call_llm_not_implemented(self, extractor):
        """Default _call_llm raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            _run(extractor._call_llm("test"))

    def test_extract_handles_dict_response(self, extractor):
        """LLM returns a dict with 'memories' key."""
        response = json.dumps({"memories": SAMPLE_MEMORIES})
        extractor._call_llm = AsyncMock(return_value=response)
        result = _run(
            extractor.extract(
                [{"role": "user", "content": "test"}]
            )
        )
        assert len(result) == 2
