"""Tests for LlamaIndex memory integration."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from neuralmem.core.types import Memory, MemoryType, SearchResult

# --- Fixtures ---


@pytest.fixture
def mock_neural_mem():
    """Create a MagicMock NeuralMem."""
    mem = MagicMock()
    mem.remember.return_value = []
    mem.recall.return_value = []
    mem.forget.return_value = 0
    return mem


@pytest.fixture
def sample_results():
    """Create sample SearchResult objects."""
    now = datetime.now(timezone.utc)
    mems = [
        Memory(
            id=f"mem{i:03d}",
            content=f"Memory item {i}",
            memory_type=MemoryType.SEMANTIC,
            user_id="alice",
            created_at=now,
            updated_at=now,
        )
        for i in range(3)
    ]
    return [
        SearchResult(memory=m, score=0.9 - i * 0.1, retrieval_method="vector")
        for i, m in enumerate(mems)
    ]


# --- Tests ---


class TestLlamaIndexMemory:
    def test_llamaindex_memory_init(self, mock_neural_mem):
        from neuralmem.integrations.llamaindex_memory import (
            NeuralMemLlamaIndexMemory,
        )

        mem = NeuralMemLlamaIndexMemory(
            mock_neural_mem, user_id="alice"
        )
        assert mem.neural_mem is mock_neural_mem
        assert mem.user_id == "alice"

    def test_llamaindex_memory_get(self, mock_neural_mem, sample_results):
        from neuralmem.integrations.llamaindex_memory import (
            NeuralMemLlamaIndexMemory,
        )

        mock_neural_mem.recall.return_value = sample_results
        mem = NeuralMemLlamaIndexMemory(
            mock_neural_mem, user_id="alice"
        )

        result = mem.get()
        assert "Memory item 0" in result
        assert "Memory item 1" in result
        assert "Memory item 2" in result
        # Verify numbered format
        assert "1. Memory item 0" in result
        assert "2. Memory item 1" in result
        assert "3. Memory item 2" in result

    def test_llamaindex_memory_put(self, mock_neural_mem):
        from neuralmem.integrations.llamaindex_memory import (
            NeuralMemLlamaIndexMemory,
        )

        mem = NeuralMemLlamaIndexMemory(
            mock_neural_mem, user_id="alice"
        )
        mem.put("User likes dark mode")

        mock_neural_mem.remember.assert_called_once_with(
            "User likes dark mode",
            user_id="alice",
            infer=False,
        )

    def test_llamaindex_memory_get_all(
        self, mock_neural_mem, sample_results
    ):
        from neuralmem.integrations.llamaindex_memory import (
            NeuralMemLlamaIndexMemory,
        )

        mock_neural_mem.recall.return_value = sample_results
        mem = NeuralMemLlamaIndexMemory(mock_neural_mem)

        all_memories = mem.get_all()
        assert len(all_memories) == 3
        assert all_memories[0] == "Memory item 0"
        assert all_memories[1] == "Memory item 1"
        assert all_memories[2] == "Memory item 2"

    def test_llamaindex_memory_clear(self, mock_neural_mem):
        from neuralmem.integrations.llamaindex_memory import (
            NeuralMemLlamaIndexMemory,
        )

        mem = NeuralMemLlamaIndexMemory(
            mock_neural_mem, user_id="alice"
        )
        mem.clear()

        mock_neural_mem.forget.assert_called_once_with(
            user_id="alice"
        )

    def test_llamaindex_memory_empty(self, mock_neural_mem):
        from neuralmem.integrations.llamaindex_memory import (
            NeuralMemLlamaIndexMemory,
        )

        mock_neural_mem.recall.return_value = []
        mem = NeuralMemLlamaIndexMemory(mock_neural_mem)

        assert mem.get() == ""
        assert mem.get_all() == []

    def test_llamaindex_memory_with_user_id(self, mock_neural_mem):
        from neuralmem.integrations.llamaindex_memory import (
            NeuralMemLlamaIndexMemory,
        )

        mem = NeuralMemLlamaIndexMemory(
            mock_neural_mem, user_id="bob"
        )
        mem.put("Test memory")

        args = mock_neural_mem.remember.call_args
        assert args.kwargs["user_id"] == "bob"

    def test_llamaindex_memory_get_format(
        self, mock_neural_mem, sample_results
    ):
        from neuralmem.integrations.llamaindex_memory import (
            NeuralMemLlamaIndexMemory,
        )

        mock_neural_mem.recall.return_value = sample_results
        mem = NeuralMemLlamaIndexMemory(mock_neural_mem)

        result = mem.get()
        lines = result.strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines, 1):
            assert line.startswith(f"{i}. ")

    def test_llamaindex_memory_put_calls_remember(
        self, mock_neural_mem
    ):
        from neuralmem.integrations.llamaindex_memory import (
            NeuralMemLlamaIndexMemory,
        )

        mem = NeuralMemLlamaIndexMemory(mock_neural_mem)
        mem.put("first")
        mem.put("second")

        assert mock_neural_mem.remember.call_count == 2

    def test_llamaindex_memory_with_mock(self, mock_neural_mem):
        """Full lifecycle test with mock."""
        from neuralmem.integrations.llamaindex_memory import (
            NeuralMemLlamaIndexMemory,
        )

        now = datetime.now(timezone.utc)
        stored_mem = Memory(
            id="m1",
            content="Test",
            memory_type=MemoryType.SEMANTIC,
            user_id="default",
            created_at=now,
            updated_at=now,
        )
        mock_neural_mem.remember.return_value = [stored_mem]

        mem = NeuralMemLlamaIndexMemory(mock_neural_mem)
        mem.put("Test")
        mock_neural_mem.remember.assert_called_once()

        # Simulate recall for get
        mock_neural_mem.recall.return_value = [
            SearchResult(
                memory=stored_mem,
                score=0.95,
                retrieval_method="vector",
            )
        ]
        result = mem.get()
        assert "Test" in result

        all_mems = mem.get_all()
        assert all_mems == ["Test"]

        mem.clear()
        mock_neural_mem.forget.assert_called_once()
