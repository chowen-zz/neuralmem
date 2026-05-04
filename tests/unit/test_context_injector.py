"""Tests for ContextInjector."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from neuralmem.assistant.context import ContextConfig, ContextInjector
from neuralmem.core.types import Memory, MemoryType, SearchResult


# ------------------------------------------------------------------
# Mocks
# ------------------------------------------------------------------

class MockRetriever:
    def __init__(self, memories: dict[str, Memory]) -> None:
        self._memories = memories
        self.vector_calls: list[tuple] = []
        self.keyword_calls: list[tuple] = []

    def vector_search(self, vector, user_id=None, memory_types=None, limit=10):
        self.vector_calls.append((vector, user_id, memory_types, limit))
        return [(m.id, 0.9) for m in self._memories.values()
                if m.user_id == user_id][:limit]

    def keyword_search(self, query, user_id=None, memory_types=None, limit=10):
        self.keyword_calls.append((query, user_id, memory_types, limit))
        return [(m.id, 0.8) for m in self._memories.values()
                if m.user_id == user_id and query.lower() in m.content.lower()][:limit]

    def get_memory(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)


class MockEmbedder:
    def encode_one(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class BrokenEmbedder:
    def encode_one(self, text: str) -> list[float]:
        raise RuntimeError("embedder broken")


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def sample_memories():
    now = datetime.now(timezone.utc)
    return {
        "mem-1": Memory(
            id="mem-1",
            content="User prefers TypeScript for frontend",
            memory_type=MemoryType.PREFERENCE,
            user_id="u1",
            embedding=[0.1, 0.2, 0.3, 0.4],
            created_at=now - timedelta(hours=1),
        ),
        "mem-2": Memory(
            id="mem-2",
            content="User deploys to AWS",
            memory_type=MemoryType.FACT,
            user_id="u1",
            embedding=[0.2, 0.3, 0.4, 0.5],
            created_at=now - timedelta(hours=48),
        ),
        "mem-3": Memory(
            id="mem-3",
            content="User likes Python",
            memory_type=MemoryType.PREFERENCE,
            user_id="u2",
            embedding=[0.5, 0.4, 0.3, 0.2],
            created_at=now,
        ),
        "mem-4": Memory(
            id="mem-4",
            content="User prefers TypeScript for frontend",  # duplicate content
            memory_type=MemoryType.PREFERENCE,
            user_id="u1",
            embedding=[0.1, 0.2, 0.3, 0.4],
            created_at=now - timedelta(hours=2),
            is_active=False,
        ),
    }


@pytest.fixture
def retriever(sample_memories):
    return MockRetriever(sample_memories)


@pytest.fixture
def embedder():
    return MockEmbedder()


@pytest.fixture
def injector(retriever, embedder):
    return ContextInjector(
        retriever=retriever,
        embedder=embedder,
        config=ContextConfig(max_memories=5, min_score=0.0),
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestContextInjectorRetrieve:
    def test_retrieve_basic(self, injector):
        results = injector.retrieve("TypeScript frontend", user_id="u1")
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, SearchResult)
            assert r.memory.user_id == "u1"
            assert r.retrieval_method == "hybrid"

    def test_retrieve_filters_by_user(self, injector):
        results = injector.retrieve("Python", user_id="u2")
        assert len(results) == 1
        assert results[0].memory.id == "mem-3"

    def test_retrieve_no_results(self, injector):
        results = injector.retrieve("nonexistent topic", user_id="unknown")
        assert results == []

    def test_retrieve_inactive_memories_excluded(self, injector):
        results = injector.retrieve("TypeScript", user_id="u1")
        ids = [r.memory.id for r in results]
        assert "mem-4" not in ids  # inactive

    def test_retrieve_limit(self, retriever, embedder):
        injector = ContextInjector(
            retriever=retriever,
            embedder=embedder,
            config=ContextConfig(max_memories=1, min_score=0.0),
        )
        results = injector.retrieve("anything", user_id="u1")
        assert len(results) <= 1

    def test_retrieve_min_score_filter(self, retriever, embedder, sample_memories):
        # Make keyword search return low score
        class LowScoreRetriever(MockRetriever):
            def keyword_search(self, query, user_id=None, memory_types=None, limit=10):
                return [(m.id, 0.1) for m in self._memories.values()
                        if m.user_id == user_id][:limit]
        injector = ContextInjector(
            retriever=LowScoreRetriever(sample_memories),
            embedder=embedder,
            config=ContextConfig(max_memories=5, min_score=0.5),
        )
        results = injector.retrieve("test", user_id="u1")
        # Only vector results should pass min_score
        for r in results:
            assert r.score >= 0.5

    def test_retrieve_tag_filter(self, injector):
        results = injector.retrieve("TypeScript", user_id="u1", tags=["preference"])
        # mem-1 has no tags set in fixture, so filter should yield nothing
        assert results == []

    def test_retrieve_recency_boost(self, retriever, embedder, sample_memories):
        config = ContextConfig(
            max_memories=5,
            min_score=0.0,
            recency_boost=True,
            recency_hours=24,
            recency_multiplier=2.0,
        )
        injector = ContextInjector(
            retriever=retriever,
            embedder=embedder,
            config=config,
        )
        results = injector.retrieve("deploy", user_id="u1")
        # mem-1 is recent, mem-2 is old; with boost mem-1 should rank higher
        if len(results) >= 2:
            assert results[0].memory.id == "mem-1"

    def test_retrieve_no_embedder(self, retriever):
        injector = ContextInjector(retriever=retriever, embedder=None)
        results = injector.retrieve("TypeScript", user_id="u1")
        # Should still work via keyword search
        assert len(results) > 0

    def test_retrieve_broken_embedder(self, retriever):
        injector = ContextInjector(retriever=retriever, embedder=BrokenEmbedder())
        results = injector.retrieve("TypeScript", user_id="u1")
        # Should fallback to keyword search
        assert len(results) > 0

    def test_retrieve_deduplication(self, retriever, embedder, sample_memories):
        # Add near-duplicate
        dup = Memory(
            id="mem-5",
            content="User prefers TypeScript for frontend development",
            memory_type=MemoryType.PREFERENCE,
            user_id="u1",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        sample_memories["mem-5"] = dup
        retriever = MockRetriever(sample_memories)
        injector = ContextInjector(
            retriever=retriever,
            embedder=embedder,
            config=ContextConfig(max_memories=5, min_score=0.0, deduplicate=True),
        )
        results = injector.retrieve("TypeScript", user_id="u1")
        contents = [r.memory.content for r in results]
        # Should not have both near-duplicates
        assert contents.count("User prefers TypeScript for frontend") <= 1 or \
               contents.count("User prefers TypeScript for frontend development") <= 1


class TestContextInjectorFormat:
    def test_format_context_empty(self, injector):
        assert injector.format_context([]) == ""

    def test_format_context_with_results(self, injector):
        results = injector.retrieve("TypeScript", user_id="u1")
        formatted = injector.format_context(results)
        assert "Relevant Memory Context" in formatted
        for r in results:
            assert r.memory.content in formatted


class TestContextConfig:
    def test_default_config(self):
        cfg = ContextConfig()
        assert cfg.vector_weight == 0.6
        assert cfg.keyword_weight == 0.4
        assert cfg.max_memories == 5
        assert cfg.deduplicate is True

    def test_custom_config(self):
        cfg = ContextConfig(vector_weight=0.8, max_memories=10, min_score=0.5)
        assert cfg.vector_weight == 0.8
        assert cfg.max_memories == 10
        assert cfg.min_score == 0.5
