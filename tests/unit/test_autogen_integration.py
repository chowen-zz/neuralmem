"""Tests for AutoGen memory integration."""
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
            content="AutoGen content",
            memory_type=MemoryType.SEMANTIC,
            user_id="agent1",
            created_at=now,
            updated_at=now,
        ),
        Memory(
            id="mem002",
            content="More content",
            memory_type=MemoryType.EPISODIC,
            user_id="agent1",
            created_at=now,
            updated_at=now,
        ),
    ]
    return [
        SearchResult(
            memory=memories[0],
            score=0.95,
            retrieval_method="vector",
        ),
        SearchResult(
            memory=memories[1],
            score=0.85,
            retrieval_method="vector",
        ),
    ]


class TestAutoGenMemory:
    """Tests for AutoGenMemory."""

    def test_init(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.autogen_memory import (
            AutoGenMemory,
        )

        memory = AutoGenMemory(
            mock_neural_mem, user_id="agent1"
        )
        assert memory.neural_mem is mock_neural_mem
        assert memory.user_id == "agent1"

    def test_init_default_user(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.autogen_memory import (
            AutoGenMemory,
        )

        memory = AutoGenMemory(mock_neural_mem)
        assert memory.user_id == "default"

    def test_add(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.autogen_memory import (
            AutoGenMemory,
        )

        memory = AutoGenMemory(
            mock_neural_mem, user_id="agent1"
        )
        memory.add("some content")
        mock_neural_mem.remember.assert_called_once_with(
            "some content",
            user_id="agent1",
            tags=["autogen"],
            infer=False,
        )

    def test_add_user_id_passed(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.autogen_memory import (
            AutoGenMemory,
        )

        memory = AutoGenMemory(
            mock_neural_mem, user_id="a1"
        )
        memory.add("test")
        call = mock_neural_mem.remember.call_args
        assert call.kwargs["user_id"] == "a1"

    def test_add_tags_contains_autogen(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.autogen_memory import (
            AutoGenMemory,
        )

        memory = AutoGenMemory(
            mock_neural_mem, user_id="a1"
        )
        memory.add("test")
        call = mock_neural_mem.remember.call_args
        assert "autogen" in call.kwargs["tags"]

    def test_add_infer_false(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.autogen_memory import (
            AutoGenMemory,
        )

        memory = AutoGenMemory(
            mock_neural_mem, user_id="a1"
        )
        memory.add("test")
        call = mock_neural_mem.remember.call_args
        assert call.kwargs["infer"] is False

    def test_search_empty(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.autogen_memory import (
            AutoGenMemory,
        )

        mock_neural_mem.recall.return_value = []
        memory = AutoGenMemory(
            mock_neural_mem, user_id="agent1"
        )
        results = memory.search("query")
        assert results == []
        mock_neural_mem.recall.assert_called_once_with(
            "query",
            user_id="agent1",
            limit=5,
        )

    def test_search_with_n_results(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.autogen_memory import (
            AutoGenMemory,
        )

        memory = AutoGenMemory(
            mock_neural_mem, user_id="agent1"
        )
        memory.search("query", n_results=3)
        mock_neural_mem.recall.assert_called_once_with(
            "query",
            user_id="agent1",
            limit=3,
        )

    def test_search_returns_dicts(
        self,
        mock_neural_mem: MagicMock,
        sample_search_results: list[SearchResult],
    ) -> None:
        from neuralmem.integrations.autogen_memory import (
            AutoGenMemory,
        )

        mock_neural_mem.recall.return_value = (
            sample_search_results
        )
        memory = AutoGenMemory(
            mock_neural_mem, user_id="agent1"
        )
        results = memory.search("query", n_results=2)
        assert len(results) == 2
        assert results[0]["content"] == "AutoGen content"
        assert results[0]["score"] == 0.95
        assert results[0]["id"] == "mem001"
        assert results[1]["content"] == "More content"
        assert results[1]["score"] == 0.85

    def test_search_passes_limit(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.autogen_memory import (
            AutoGenMemory,
        )

        memory = AutoGenMemory(
            mock_neural_mem, user_id="a1"
        )
        memory.search("test", n_results=10)
        call = mock_neural_mem.recall.call_args
        assert call.kwargs["limit"] == 10

    def test_clear(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.autogen_memory import (
            AutoGenMemory,
        )

        memory = AutoGenMemory(
            mock_neural_mem, user_id="agent1"
        )
        memory.clear()
        mock_neural_mem.forget.assert_called_once_with(
            user_id="agent1"
        )

    def test_add_content_passed(
        self, mock_neural_mem: MagicMock
    ) -> None:
        from neuralmem.integrations.autogen_memory import (
            AutoGenMemory,
        )

        memory = AutoGenMemory(
            mock_neural_mem, user_id="a1"
        )
        memory.add("my content")
        call = mock_neural_mem.remember.call_args
        assert call.args[0] == "my content"
