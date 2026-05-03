"""记忆合并器单元测试 — 15+ tests"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from neuralmem.core.types import Memory, MemoryType
from neuralmem.extraction.merger import MemoryMerger, MergeResult


@pytest.fixture
def merger():
    return MemoryMerger(similarity_threshold=0.90)


def _make_memory(content: str, **kwargs) -> Memory:
    """Helper to create a Memory with defaults."""
    defaults = {
        "content": content,
        "memory_type": MemoryType.SEMANTIC,
    }
    defaults.update(kwargs)
    return Memory(**defaults)


# ==================== Initialization ====================


class TestInitialization:

    def test_default_threshold(self):
        m = MemoryMerger()
        assert m.similarity_threshold == 0.90

    def test_custom_threshold(self):
        m = MemoryMerger(similarity_threshold=0.85)
        assert m.similarity_threshold == 0.85

    def test_invalid_threshold_high(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            MemoryMerger(similarity_threshold=1.5)

    def test_invalid_threshold_low(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            MemoryMerger(similarity_threshold=-0.1)


# ==================== Duplicate detection ====================


class TestDuplicateDetection:

    def test_find_exact_duplicate(self, merger):
        m1 = _make_memory("Python is a programming language")
        m2 = _make_memory("Python is a programming language")
        dupes = merger.find_duplicates(m1, [m2])
        assert len(dupes) == 1
        assert dupes[0].id == m2.id

    def test_find_similar_duplicate(self, merger):
        m1 = _make_memory("Python is a great programming language for AI")
        m2 = _make_memory("Python is a great programming language for AI")
        dupes = merger.find_duplicates(m1, [m2])
        assert len(dupes) == 1

    def test_find_near_duplicate_low_threshold(self):
        low_merger = MemoryMerger(similarity_threshold=0.70)
        m1 = _make_memory("Python is a great programming language")
        m2 = _make_memory("Python is a great programming language for ML")
        dupes = low_merger.find_duplicates(m1, [m2])
        assert len(dupes) == 1

    def test_no_duplicate_different_content(self, merger):
        m1 = _make_memory("Python is a programming language")
        m2 = _make_memory("The weather is nice today")
        dupes = merger.find_duplicates(m1, [m2])
        assert len(dupes) == 0

    def test_skip_inactive_memories(self, merger):
        m1 = _make_memory("Python is a programming language")
        m2 = _make_memory(
            "Python is a programming language",
            is_active=False,
        )
        dupes = merger.find_duplicates(m1, [m2])
        assert len(dupes) == 0

    def test_skip_self_comparison(self, merger):
        m1 = _make_memory("Python is a programming language")
        dupes = merger.find_duplicates(m1, [m1])
        assert len(dupes) == 0

    def test_custom_similarity_function(self, merger):
        m1 = _make_memory("Hello world")
        m2 = _make_memory("Hello world")

        def always_one(a, b):
            return 1.0

        dupes = merger.find_duplicates(
            m1, [m2], get_similarity=always_one
        )
        assert len(dupes) == 1


# ==================== Merge strategy ====================


class TestMergeStrategy:

    def test_merge_keeps_longer_content(self, merger):
        shorter = _make_memory("Python is great")
        longer = _make_memory(
            "Python is a great programming language for data science"
        )
        result = merger.merge(shorter, longer)
        assert result.was_merged
        assert len(result.merged_memory.content) >= len(shorter.content)

    def test_merge_combines_tags(self, merger):
        m1 = _make_memory("Test content", tags=("tag1",))
        m2 = _make_memory("Test content", tags=("tag2",))
        result = merger.merge(m1, m2)
        combined = result.merged_memory.tags
        assert "tag1" in combined
        assert "tag2" in combined

    def test_merge_takes_higher_importance(self, merger):
        m1 = _make_memory("Test content", importance=0.3)
        m2 = _make_memory("Test content", importance=0.8)
        result = merger.merge(m1, m2)
        assert result.merged_memory.importance == 0.8

    def test_merge_updates_timestamp(self, merger):
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        m1 = _make_memory("Test content")
        m2 = _make_memory("Test content")
        m2 = m2.model_copy(update={"updated_at": old_time})
        result = merger.merge(m1, m2)
        assert result.merged_memory.updated_at > old_time

    def test_merge_preserves_duplicate_id(self, merger):
        m1 = _make_memory("Test content")
        m2 = _make_memory("Test content")
        result = merger.merge(m1, m2)
        assert result.merged_memory.id == m2.id

    def test_merge_records_merged_with(self, merger):
        m1 = _make_memory("Test content")
        m2 = _make_memory("Test content")
        result = merger.merge(m1, m2)
        assert m2.id in result.merged_with

    def test_merge_not_flagged_when_no_duplicate(self, merger):
        m = _make_memory("Unique content here")
        result = MergeResult(
            merged_memory=m, was_merged=False
        )
        assert not result.was_merged


# ==================== Batch merge ====================


class TestBatchMerge:

    def test_merge_batch_with_duplicates(self, merger):
        m1 = _make_memory("Python is a programming language")
        m2 = _make_memory("Python is a programming language")
        m3 = _make_memory("The weather is nice today")

        results = merger.merge_batch(
            [m1, m3], existing_memories=[m2]
        )
        assert len(results) == 2

    def test_merge_batch_no_duplicates(self, merger):
        m1 = _make_memory("Alpha content")
        m2 = _make_memory("Beta content")

        results = merger.merge_batch(
            [m1], existing_memories=[m2]
        )
        assert len(results) == 1
        assert not results[0].was_merged

    def test_merge_batch_empty_existing(self, merger):
        m1 = _make_memory("Some content")
        results = merger.merge_batch([m1], existing_memories=[])
        assert len(results) == 1
        assert not results[0].was_merged

    def test_merge_batch_empty_new(self, merger):
        m1 = _make_memory("Existing content")
        results = merger.merge_batch([], existing_memories=[m1])
        assert len(results) == 0


# ==================== Content similarity ====================


class TestContentSimilarity:

    def test_identical_content(self):
        sim = MemoryMerger._content_similarity(
            "hello world", "hello world"
        )
        assert sim == 1.0

    def test_completely_different(self):
        sim = MemoryMerger._content_similarity(
            "hello world", "foo bar"
        )
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = MemoryMerger._content_similarity(
            "hello world foo", "hello world bar"
        )
        assert 0.0 < sim < 1.0

    def test_empty_strings(self):
        sim = MemoryMerger._content_similarity("", "")
        assert sim == 1.0

    def test_one_empty(self):
        sim = MemoryMerger._content_similarity("hello", "")
        assert sim == 0.0
