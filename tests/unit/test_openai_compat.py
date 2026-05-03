"""Tests for OpenAI SDK compatibility layer."""
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
    mem.get.return_value = None
    mem.update.return_value = None
    mem.forget.return_value = 0
    return mem


@pytest.fixture
def sample_memory():
    """A single sample Memory object."""
    now = datetime.now(timezone.utc)
    return Memory(
        id="mem001",
        content="User likes dark mode",
        memory_type=MemoryType.SEMANTIC,
        user_id="alice",
        importance=0.8,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def sample_search_results(sample_memory):
    """Sample search results."""
    return [
        SearchResult(
            memory=sample_memory,
            score=0.95,
            retrieval_method="vector",
        ),
    ]


# --- Tests ---


class TestOpenAICompat:
    def test_openai_compat_init(self, mock_neural_mem):
        from neuralmem.integrations.openai_compat import (
            NeuralMemOpenAICompat,
        )

        client = NeuralMemOpenAICompat(
            mock_neural_mem, user_id="alice"
        )
        assert client._neural_mem is mock_neural_mem
        assert client._user_id == "alice"
        assert client._memories is None

    def test_openai_compat_memories_property(self, mock_neural_mem):
        from neuralmem.integrations.openai_compat import (
            NeuralMemOpenAICompat,
        )

        client = NeuralMemOpenAICompat(mock_neural_mem)
        m = client.memories
        assert m is not None
        assert m is client.memories  # same instance on repeated access

    def test_openai_compat_memories_create(
        self, mock_neural_mem, sample_memory
    ):
        from neuralmem.integrations.openai_compat import (
            NeuralMemOpenAICompat,
        )

        mock_neural_mem.remember.return_value = [sample_memory]
        client = NeuralMemOpenAICompat(
            mock_neural_mem, user_id="alice"
        )

        result = client.memories.create("User likes dark mode")
        assert result["id"] == "mem001"
        assert result["object"] == "memory"
        assert result["content"] == "User likes dark mode"
        assert result["user_id"] == "alice"
        mock_neural_mem.remember.assert_called_once_with(
            "User likes dark mode", user_id="alice"
        )

    def test_openai_compat_memories_list(
        self, mock_neural_mem, sample_search_results
    ):
        from neuralmem.integrations.openai_compat import (
            NeuralMemOpenAICompat,
        )

        mock_neural_mem.recall.return_value = sample_search_results
        client = NeuralMemOpenAICompat(
            mock_neural_mem, user_id="alice"
        )

        result = client.memories.list()
        assert result["object"] == "list"
        assert len(result["data"]) == 1
        assert result["data"][0]["id"] == "mem001"
        assert result["data"][0]["content"] == "User likes dark mode"

    def test_openai_compat_memories_retrieve(
        self, mock_neural_mem, sample_memory
    ):
        from neuralmem.integrations.openai_compat import (
            NeuralMemOpenAICompat,
        )

        mock_neural_mem.get.return_value = sample_memory
        client = NeuralMemOpenAICompat(
            mock_neural_mem, user_id="alice"
        )

        result = client.memories.retrieve("mem001")
        assert result["id"] == "mem001"
        assert result["object"] == "memory"
        assert result["content"] == "User likes dark mode"

    def test_openai_compat_memories_update(
        self, mock_neural_mem, sample_memory
    ):
        from neuralmem.integrations.openai_compat import (
            NeuralMemOpenAICompat,
        )

        updated_mem = sample_memory.model_copy(
            update={"content": "Updated content"}
        )
        mock_neural_mem.update.return_value = updated_mem
        client = NeuralMemOpenAICompat(mock_neural_mem)

        result = client.memories.update("mem001", "Updated content")
        assert result["content"] == "Updated content"
        mock_neural_mem.update.assert_called_once_with(
            "mem001", "Updated content"
        )

    def test_openai_compat_memories_delete(self, mock_neural_mem):
        from neuralmem.integrations.openai_compat import (
            NeuralMemOpenAICompat,
        )

        client = NeuralMemOpenAICompat(mock_neural_mem)
        result = client.memories.delete("mem001")

        assert result["id"] == "mem001"
        assert result["object"] == "memory"
        assert result["deleted"] is True
        mock_neural_mem.forget.assert_called_once_with(
            memory_id="mem001"
        )

    def test_openai_compat_memories_search(
        self, mock_neural_mem, sample_search_results
    ):
        from neuralmem.integrations.openai_compat import (
            NeuralMemOpenAICompat,
        )

        mock_neural_mem.recall.return_value = sample_search_results
        client = NeuralMemOpenAICompat(
            mock_neural_mem, user_id="alice"
        )

        result = client.memories.search("color preference")
        assert result["object"] == "list"
        assert len(result["data"]) == 1
        assert result["data"][0]["score"] == 0.95
        mock_neural_mem.recall.assert_called_once_with(
            "color preference", user_id="alice"
        )

    def test_openai_compat_response_format(
        self, mock_neural_mem, sample_memory
    ):
        """Verify OpenAI-style dict structure on create response."""
        from neuralmem.integrations.openai_compat import (
            NeuralMemOpenAICompat,
        )

        mock_neural_mem.remember.return_value = [sample_memory]
        client = NeuralMemOpenAICompat(mock_neural_mem)

        result = client.memories.create("test")
        required_keys = {
            "id", "object", "content",
            "memory_type", "user_id", "importance",
            "created_at",
        }
        assert required_keys.issubset(result.keys())
        assert isinstance(result["id"], str)
        assert result["object"] == "memory"
        assert isinstance(result["content"], str)
        assert isinstance(result["importance"], float)

    def test_openai_compat_with_mock_neural_mem(self, mock_neural_mem):
        """Full lifecycle using only mock."""
        from neuralmem.integrations.openai_compat import (
            NeuralMemOpenAICompat,
        )

        now = datetime.now(timezone.utc)
        m = Memory(
            id="x1",
            content="test",
            memory_type=MemoryType.SEMANTIC,
            user_id="default",
            importance=0.5,
            created_at=now,
            updated_at=now,
        )
        mock_neural_mem.remember.return_value = [m]
        mock_neural_mem.get.return_value = m
        mock_neural_mem.update.return_value = m
        mock_neural_mem.recall.return_value = [
            SearchResult(memory=m, score=0.9, retrieval_method="vector")
        ]

        client = NeuralMemOpenAICompat(mock_neural_mem)

        # Create
        r = client.memories.create("test")
        assert r["object"] == "memory"

        # List
        r = client.memories.list()
        assert r["object"] == "list"

        # Retrieve
        r = client.memories.retrieve("x1")
        assert r["id"] == "x1"

        # Update
        r = client.memories.update("x1", "new")
        assert r["object"] == "memory"

        # Search
        r = client.memories.search("query")
        assert r["object"] == "list"

        # Delete
        r = client.memories.delete("x1")
        assert r["deleted"] is True

    def test_openai_compat_retrieve_not_found(self, mock_neural_mem):
        """Should raise ValueError when memory not found."""
        from neuralmem.integrations.openai_compat import (
            NeuralMemOpenAICompat,
        )

        mock_neural_mem.get.return_value = None
        client = NeuralMemOpenAICompat(mock_neural_mem)

        with pytest.raises(ValueError, match="not found"):
            client.memories.retrieve("nonexistent")

    def test_openai_compat_update_not_found(self, mock_neural_mem):
        """Should raise ValueError when updating nonexistent memory."""
        from neuralmem.integrations.openai_compat import (
            NeuralMemOpenAICompat,
        )

        mock_neural_mem.update.return_value = None
        client = NeuralMemOpenAICompat(mock_neural_mem)

        with pytest.raises(ValueError, match="not found"):
            client.memories.update("nonexistent", "content")

    def test_openai_compat_create_empty_result(self, mock_neural_mem):
        """Handle empty remember result gracefully."""
        from neuralmem.integrations.openai_compat import (
            NeuralMemOpenAICompat,
        )

        mock_neural_mem.remember.return_value = []
        client = NeuralMemOpenAICompat(mock_neural_mem)

        result = client.memories.create("nothing stored")
        assert result["id"] == ""
        assert result["object"] == "memory"
