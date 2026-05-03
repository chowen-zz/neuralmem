"""Tests for CrewAI memory integration."""
from __future__ import annotations

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
            content="Query: What is Python?",
            memory_type=MemoryType.SEMANTIC,
            user_id="crew1",
            created_at=now,
            updated_at=now,
        ),
        Memory(
            id="mem002",
            content="Result: Python is a language",
            memory_type=MemoryType.SEMANTIC,
            user_id="crew1",
            created_at=now,
            updated_at=now,
        ),
    ]
    return [
        SearchResult(
            memory=memories[0],
            score=0.9,
            retrieval_method="vector",
        ),
        SearchResult(
            memory=memories[1],
            score=0.8,
            retrieval_method="vector",
        ),
    ]


class TestCrewAIMemory:
    """Tests for CrewAIMemory."""

    def test_init(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.crewai_memory import (
            CrewAIMemory,
        )

        memory = CrewAIMemory(
            mock_neural_mem, user_id="crew1"
        )
        assert memory.neural_mem is mock_neural_mem
        assert memory.user_id == "crew1"

    def test_init_default_user(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.crewai_memory import (
            CrewAIMemory,
        )

        memory = CrewAIMemory(mock_neural_mem)
        assert memory.user_id == "default"

    def test_save_both(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.crewai_memory import (
            CrewAIMemory,
        )

        memory = CrewAIMemory(
            mock_neural_mem, user_id="crew1"
        )
        memory.save("What is AI?", "AI is intelligence")

        assert mock_neural_mem.remember.call_count == 2
        calls = mock_neural_mem.remember.call_args_list
        content0 = calls[0].args[0]
        content1 = calls[1].args[0]
        assert "Query: What is AI?" in content0
        assert "Result: AI is intelligence" in content1

    def test_save_query_only(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.crewai_memory import (
            CrewAIMemory,
        )

        memory = CrewAIMemory(
            mock_neural_mem, user_id="c1"
        )
        memory.save("test query", "")

        assert mock_neural_mem.remember.call_count == 1
        call = mock_neural_mem.remember.call_args
        assert "Query: test query" in call.args[0]

    def test_save_result_only(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.crewai_memory import (
            CrewAIMemory,
        )

        memory = CrewAIMemory(
            mock_neural_mem, user_id="c1"
        )
        memory.save("", "test result")

        assert mock_neural_mem.remember.call_count == 1
        call = mock_neural_mem.remember.call_args
        assert "Result: test result" in call.args[0]

    def test_save_empty(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.crewai_memory import (
            CrewAIMemory,
        )

        memory = CrewAIMemory(mock_neural_mem)
        memory.save("", "")
        mock_neural_mem.remember.assert_not_called()

    def test_save_passes_tags(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.crewai_memory import (
            CrewAIMemory,
        )

        memory = CrewAIMemory(
            mock_neural_mem, user_id="c1"
        )
        memory.save("q", "r")
        calls = mock_neural_mem.remember.call_args_list
        assert "crewai" in calls[0].kwargs["tags"]
        assert "query" in calls[0].kwargs["tags"]
        assert "result" in calls[1].kwargs["tags"]

    def test_search_empty(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.crewai_memory import (
            CrewAIMemory,
        )

        mock_neural_mem.recall.return_value = []
        memory = CrewAIMemory(
            mock_neural_mem, user_id="c1"
        )
        results = memory.search("test query")
        assert results == []
        mock_neural_mem.recall.assert_called_once_with(
            "test query",
            user_id="c1",
            limit=5,
        )

    def test_search_with_results(
        self,
        mock_neural_mem: MagicMock,
        sample_search_results: list[SearchResult],
    ) -> None:
        from neuralmem.integrations.crewai_memory import (
            CrewAIMemory,
        )

        mock_neural_mem.recall.return_value = (
            sample_search_results
        )
        memory = CrewAIMemory(
            mock_neural_mem, user_id="crew1"
        )
        results = memory.search("query", limit=2)
        assert len(results) == 2
        assert (
            results[0]["content"] == "Query: What is Python?"
        )
        assert results[0]["score"] == 0.9
        assert results[0]["id"] == "mem001"

    def test_clear(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.crewai_memory import (
            CrewAIMemory,
        )

        memory = CrewAIMemory(
            mock_neural_mem, user_id="crew1"
        )
        memory.clear()
        mock_neural_mem.forget.assert_called_once_with(
            user_id="crew1"
        )

    def test_search_passes_limit(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.crewai_memory import (
            CrewAIMemory,
        )

        memory = CrewAIMemory(
            mock_neural_mem, user_id="c1"
        )
        memory.search("test", limit=10)
        mock_neural_mem.recall.assert_called_once_with(
            "test",
            user_id="c1",
            limit=10,
        )

    def test_save_user_id_passed(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.crewai_memory import (
            CrewAIMemory,
        )

        memory = CrewAIMemory(
            mock_neural_mem, user_id="c1"
        )
        memory.save("q", "r")
        calls = mock_neural_mem.remember.call_args_list
        assert calls[0].kwargs["user_id"] == "c1"
