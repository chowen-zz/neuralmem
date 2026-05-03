"""对话记忆提取器单元测试 — 20+ tests"""
from __future__ import annotations

import pytest

from neuralmem.core.types import MemoryType
from neuralmem.extraction.conversation_extractor import (
    ConversationExtractor,
    ExtractedMemory,
)


@pytest.fixture
def extractor():
    return ConversationExtractor()


# ==================== Basic extraction ====================


class TestBasicExtraction:

    def test_extract_empty_messages(self, extractor):
        result = extractor.extract([])
        assert result == []

    def test_extract_single_user_message(self, extractor):
        messages = [
            {"role": "user", "content": "I prefer Python for data science"},
        ]
        result = extractor.extract(messages)
        assert len(result) > 0
        assert result[0].source_role == "user"

    def test_extract_single_assistant_message(self, extractor):
        messages = [
            {
                "role": "assistant",
                "content": "Python is a popular programming language",
            },
        ]
        result = extractor.extract(messages)
        assert len(result) > 0
        assert result[0].source_role == "assistant"

    def test_extract_multi_turn(self, extractor):
        messages = [
            {"role": "user", "content": "I love using React for frontend"},
            {
                "role": "assistant",
                "content": "React is a great choice for UI development",
            },
            {"role": "user", "content": "I also prefer TypeScript over JavaScript"},
        ]
        result = extractor.extract(messages)
        assert len(result) >= 3

    def test_extract_skips_empty_content(self, extractor):
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "  "},
            {"role": "user", "content": "Hello world"},
        ]
        result = extractor.extract(messages)
        assert len(result) >= 1

    def test_extract_skips_very_short_content(self, extractor):
        messages = [
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "hi"},
        ]
        result = extractor.extract(messages)
        # Very short content (< 3 chars) is skipped
        assert len(result) == 0


# ==================== Memory type classification ====================


class TestTypeClassification:

    def test_preference_detection_en(self, extractor):
        messages = [
            {"role": "user", "content": "I prefer dark mode for my IDE"},
        ]
        result = extractor.extract(messages)
        types = [m.memory_type for m in result]
        assert MemoryType.PREFERENCE in types

    def test_preference_detection_zh(self, extractor):
        messages = [
            {"role": "user", "content": "我喜欢用 Python 写后端"},
        ]
        result = extractor.extract(messages)
        types = [m.memory_type for m in result]
        assert MemoryType.PREFERENCE in types

    def test_fact_detection_en(self, extractor):
        messages = [
            {
                "role": "user",
                "content": "My name is John and I live in New York",
            },
        ]
        result = extractor.extract(messages)
        types = [m.memory_type for m in result]
        assert MemoryType.FACT in types

    def test_procedural_detection_en(self, extractor):
        messages = [
            {
                "role": "user",
                "content": "First install the package, then run the tests",
            },
        ]
        result = extractor.extract(messages)
        types = [m.memory_type for m in result]
        assert MemoryType.PROCEDURAL in types

    def test_episodic_detection_en(self, extractor):
        messages = [
            {
                "role": "user",
                "content": "Yesterday we deployed the new version to production",
            },
        ]
        result = extractor.extract(messages)
        types = [m.memory_type for m in result]
        assert MemoryType.EPISODIC in types

    def test_override_memory_type(self, extractor):
        messages = [
            {"role": "user", "content": "The sky is blue today"},
        ]
        result = extractor.extract(
            messages, memory_type=MemoryType.WORKING
        )
        for m in result:
            assert m.memory_type is MemoryType.WORKING


# ==================== Language detection ====================


class TestLanguageDetection:

    def test_detect_english(self, extractor):
        messages = [
            {"role": "user", "content": "Hello, how are you today?"},
        ]
        result = extractor.extract(messages)
        assert result[0].metadata["detected_language"] == "en"

    def test_detect_chinese(self, extractor):
        messages = [
            {"role": "user", "content": "你好，今天天气怎么样？"},
        ]
        result = extractor.extract(messages)
        assert result[0].metadata["detected_language"] == "zh"

    def test_chinese_extraction_preserves_language(self, extractor):
        messages = [
            {"role": "user", "content": "我喜欢用 Python 编程"},
        ]
        result = extractor.extract(messages)
        # Content should be in original language
        assert any("Python" in m.content for m in result)


# ==================== Confidence scoring ====================


class TestConfidenceScoring:

    def test_confidence_range(self, extractor):
        messages = [
            {"role": "user", "content": "I like Python very much"},
            {
                "role": "assistant",
                "content": "Python is a versatile language used in many domains",
            },
        ]
        result = extractor.extract(messages)
        for m in result:
            assert 0.0 <= m.confidence <= 1.0

    def test_user_preference_higher_confidence(self, extractor):
        messages = [
            {"role": "user", "content": "I prefer using Docker containers"},
        ]
        result = extractor.extract(messages)
        # User preferences should have higher confidence
        assert all(m.confidence >= 0.7 for m in result)


# ==================== Tags extraction ====================


class TestTagExtraction:

    def test_technology_tag(self, extractor):
        messages = [
            {"role": "user", "content": "I use python and git daily"},
        ]
        result = extractor.extract(messages)
        all_tags = [tag for m in result for tag in m.tags]
        assert "technology" in all_tags

    def test_issue_tag_zh(self, extractor):
        messages = [
            {"role": "user", "content": "这个项目有个问题需要修复"},
        ]
        result = extractor.extract(messages)
        all_tags = [tag for m in result for tag in m.tags]
        assert "issue" in all_tags or "work" in all_tags


# ==================== Statement splitting ====================


class TestStatementSplitting:

    def test_split_multiple_sentences_en(self, extractor):
        messages = [
            {
                "role": "user",
                "content": "I like Python. It is versatile. I use it for ML.",
            },
        ]
        result = extractor.extract(messages)
        # Should split into at least 2 memories
        assert len(result) >= 2

    def test_split_multiple_sentences_zh(self, extractor):
        messages = [
            {
                "role": "user",
                "content": "我喜欢 Python。它很通用。我用它做机器学习。",
            },
        ]
        result = extractor.extract(messages)
        assert len(result) >= 2


# ==================== ExtractedMemory dataclass ====================


class TestExtractedMemory:

    def test_extracted_memory_defaults(self):
        m = ExtractedMemory(
            content="test", memory_type=MemoryType.FACT
        )
        assert m.source_role == "user"
        assert m.confidence == 0.8
        assert m.tags == []
        assert m.metadata == {}

    def test_extracted_memory_custom_values(self):
        m = ExtractedMemory(
            content="test content",
            memory_type=MemoryType.PREFERENCE,
            source_role="assistant",
            confidence=0.95,
            tags=["tech"],
            metadata={"lang": "en"},
        )
        assert m.source_role == "assistant"
        assert m.confidence == 0.95
        assert "tech" in m.tags
