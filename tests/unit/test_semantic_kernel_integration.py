"""Tests for Semantic Kernel memory integration."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from neuralmem.core.types import (
    Memory,
    MemoryType,
    SearchResult,
)


@pytest.fixture
def mock_neural_mem() -> MagicMock:
    """Create a MagicMock NeuralMem."""
    mem = MagicMock()
    mem.remember.return_value = []
    mem.recall.return_value = []
    mem.forget.return_value = 0
    return mem


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Create sample SearchResult objects."""
    now = datetime.now(timezone.utc)
    memories = [
        Memory(
            id="mem001",
            content="SK content",
            memory_type=MemoryType.SEMANTIC,
            user_id="sk_user",
            created_at=now,
            updated_at=now,
            source="some description",
        ),
    ]
    return [
        SearchResult(
            memory=memories[0],
            score=0.92,
            retrieval_method="vector",
        ),
    ]


def _run(coro):  # type: ignore[no-untyped-def]
    """Run an async coroutine in tests."""
    return asyncio.run(coro)


class TestSemanticKernelMemory:
    """Tests for SemanticKernelMemory."""

    def test_init(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.semantic_kernel_memory import (
            SemanticKernelMemory,
        )

        memory = SemanticKernelMemory(
            mock_neural_mem, user_id="sk1"
        )
        assert memory.neural_mem is mock_neural_mem
        assert memory.user_id == "sk1"

    def test_init_default_user(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.semantic_kernel_memory import (
            SemanticKernelMemory,
        )

        memory = SemanticKernelMemory(mock_neural_mem)
        assert memory.user_id == "default"

    def test_save_information(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.semantic_kernel_memory import (
            SemanticKernelMemory,
        )

        memory = SemanticKernelMemory(
            mock_neural_mem, user_id="sk1"
        )
        _run(
            memory.save_information(
                collection="test",
                id="info1",
                text="some info",
            )
        )
        mock_neural_mem.remember.assert_called_once()
        call = mock_neural_mem.remember.call_args
        assert call.args[0] == "some info"
        assert call.kwargs["user_id"] == "sk1"
        assert (
            "semantic_kernel" in call.kwargs["tags"]
        )
        assert "test" in call.kwargs["tags"]

    def test_save_information_with_description(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.semantic_kernel_memory import (
            SemanticKernelMemory,
        )

        memory = SemanticKernelMemory(
            mock_neural_mem, user_id="sk1"
        )
        _run(
            memory.save_information(
                collection="col",
                id="id1",
                text="info",
                description="my desc",
            )
        )
        call = mock_neural_mem.remember.call_args
        assert call.kwargs["source"] == "my desc"

    def test_save_information_with_metadata(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.semantic_kernel_memory import (
            SemanticKernelMemory,
        )

        memory = SemanticKernelMemory(
            mock_neural_mem, user_id="sk1"
        )
        _run(
            memory.save_information(
                collection="col",
                id="id1",
                text="info",
                additional_metadata="extra_tag",
            )
        )
        call = mock_neural_mem.remember.call_args
        assert "extra_tag" in call.kwargs["tags"]

    def test_save_infer_false(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.semantic_kernel_memory import (
            SemanticKernelMemory,
        )

        memory = SemanticKernelMemory(
            mock_neural_mem, user_id="sk1"
        )
        _run(
            memory.save_information(
                collection="c", id="i", text="t"
            )
        )
        call = mock_neural_mem.remember.call_args
        assert call.kwargs["infer"] is False

    def test_save_user_id_passed(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.semantic_kernel_memory import (
            SemanticKernelMemory,
        )

        memory = SemanticKernelMemory(
            mock_neural_mem, user_id="u1"
        )
        _run(
            memory.save_information(
                collection="c", id="i", text="t"
            )
        )
        call = mock_neural_mem.remember.call_args
        assert call.kwargs["user_id"] == "u1"

    def test_search_empty(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.semantic_kernel_memory import (
            SemanticKernelMemory,
        )

        mock_neural_mem.recall.return_value = []
        memory = SemanticKernelMemory(
            mock_neural_mem, user_id="sk1"
        )
        results = _run(memory.search("test query"))
        assert results == []
        mock_neural_mem.recall.assert_called_once_with(
            "test query",
            user_id="sk1",
            limit=5,
            min_score=0.0,
        )

    def test_search_with_limit(
        self,
        mock_neural_mem: MagicMock,
        sample_search_results: list[SearchResult],
    ) -> None:
        from neuralmem.integrations.semantic_kernel_memory import (
            SemanticKernelMemory,
        )

        mock_neural_mem.recall.return_value = (
            sample_search_results
        )
        memory = SemanticKernelMemory(
            mock_neural_mem, user_id="sk1"
        )
        results = _run(
            memory.search("query", limit=3)
        )
        assert len(results) == 1
        assert results[0]["text"] == "SK content"
        assert results[0]["id"] == "mem001"
        assert results[0]["score"] == 0.92
        assert results[0]["description"] == "some description"

    def test_search_with_min_relevance(
        self,
        mock_neural_mem: MagicMock,
        sample_search_results: list[SearchResult],
    ) -> None:
        from neuralmem.integrations.semantic_kernel_memory import (
            SemanticKernelMemory,
        )

        mock_neural_mem.recall.return_value = (
            sample_search_results
        )
        memory = SemanticKernelMemory(
            mock_neural_mem, user_id="sk1"
        )
        _run(
            memory.search(
                "query", min_relevance_score=0.5
            )
        )
        call = mock_neural_mem.recall.call_args
        assert call.kwargs["min_score"] == 0.5

    def test_search_passes_limit(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.semantic_kernel_memory import (
            SemanticKernelMemory,
        )

        mock_neural_mem.recall.return_value = []
        memory = SemanticKernelMemory(
            mock_neural_mem, user_id="sk1"
        )
        _run(memory.search("q", limit=10))
        call = mock_neural_mem.recall.call_args
        assert call.kwargs["limit"] == 10

    def test_get_async_found(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.semantic_kernel_memory import (
            SemanticKernelMemory,
        )

        now = datetime.now(timezone.utc)
        mem = Memory(
            id="mem001",
            content="stored info",
            memory_type=MemoryType.SEMANTIC,
            user_id="sk1",
            created_at=now,
            updated_at=now,
        )
        mock_neural_mem.recall.return_value = [
            SearchResult(
                memory=mem,
                score=1.0,
                retrieval_method="vector",
            )
        ]
        memory = SemanticKernelMemory(
            mock_neural_mem, user_id="sk1"
        )
        result = _run(
            memory.get_async(
                collection="col", id="mem001"
            )
        )
        assert result is not None
        assert result["text"] == "stored info"
        assert result["id"] == "mem001"

    def test_get_async_not_found(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.semantic_kernel_memory import (
            SemanticKernelMemory,
        )

        mock_neural_mem.recall.return_value = []
        memory = SemanticKernelMemory(
            mock_neural_mem, user_id="sk1"
        )
        result = _run(
            memory.get_async(
                collection="col", id="missing"
            )
        )
        assert result is None

    def test_search_passes_min_relevance_score(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.semantic_kernel_memory import (
            SemanticKernelMemory,
        )

        mock_neural_mem.recall.return_value = []
        memory = SemanticKernelMemory(
            mock_neural_mem, user_id="sk1"
        )
        _run(
            memory.search(
                "q",
                collection="col",
                min_relevance_score=0.7,
            )
        )
        call = mock_neural_mem.recall.call_args
        assert call.kwargs["min_score"] == 0.7

    def test_save_tags_contain_semantic_kernel(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.semantic_kernel_memory import (
            SemanticKernelMemory,
        )

        memory = SemanticKernelMemory(
            mock_neural_mem, user_id="sk1"
        )
        _run(
            memory.save_information(
                collection="c", id="i", text="t"
            )
        )
        call = mock_neural_mem.remember.call_args
        assert "semantic_kernel" in call.kwargs["tags"]
