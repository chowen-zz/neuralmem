"""Tests for LangChain memory integration."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from neuralmem.core.types import Memory, MemoryType, SearchResult

# --- Fixtures ---


@pytest.fixture
def mock_neural_mem():
    """Create a MagicMock NeuralMem with realistic method behavior."""
    mem = MagicMock()
    mem.remember.return_value = []
    mem.recall.return_value = []
    mem.forget.return_value = 0
    return mem


@pytest.fixture
def sample_search_results():
    """Create sample SearchResult objects."""
    now = datetime.now(timezone.utc)
    memories = [
        Memory(
            id="mem001",
            content="User likes Python",
            memory_type=MemoryType.SEMANTIC,
            user_id="alice",
            created_at=now,
            updated_at=now,
        ),
        Memory(
            id="mem002",
            content="User works on AI projects",
            memory_type=MemoryType.SEMANTIC,
            user_id="alice",
            created_at=now,
            updated_at=now,
        ),
    ]
    return [
        SearchResult(memory=memories[0], score=0.9, retrieval_method="vector"),
        SearchResult(memory=memories[1], score=0.8, retrieval_method="vector"),
    ]


@pytest.fixture
def chat_search_results():
    """Create chat-style SearchResult objects."""
    now = datetime.now(timezone.utc)
    mems = [
        Memory(
            id="chat001",
            content="Human: Hello!",
            memory_type=MemoryType.EPISODIC,
            user_id="alice",
            created_at=now,
            updated_at=now,
        ),
        Memory(
            id="chat002",
            content="AI: Hi there!",
            memory_type=MemoryType.EPISODIC,
            user_id="alice",
            created_at=now,
            updated_at=now,
        ),
    ]
    return [
        SearchResult(memory=mems[0], score=0.95, retrieval_method="vector"),
        SearchResult(memory=mems[1], score=0.9, retrieval_method="vector"),
    ]


# --- NeuralMemLangChainMemory Tests ---


class TestLangChainMemory:
    def test_langchain_memory_init(self, mock_neural_mem):
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainMemory,
        )

        mem = NeuralMemLangChainMemory(
            mock_neural_mem, user_id="alice", k=5
        )
        assert mem.neural_mem is mock_neural_mem
        assert mem.user_id == "alice"
        assert mem.k == 5

    def test_langchain_memory_variables(self, mock_neural_mem):
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainMemory,
        )

        mem = NeuralMemLangChainMemory(mock_neural_mem)
        assert mem.memory_variables == ["history"]

    def test_langchain_load_memory_variables(
        self, mock_neural_mem, sample_search_results
    ):
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainMemory,
        )

        mock_neural_mem.recall.return_value = sample_search_results
        mem = NeuralMemLangChainMemory(
            mock_neural_mem, user_id="alice"
        )

        result = mem.load_memory_variables({"input": "What does the user like?"})

        assert "history" in result
        assert "User likes Python" in result["history"]
        assert "User works on AI projects" in result["history"]
        mock_neural_mem.recall.assert_called_once_with(
            "What does the user like?",
            user_id="alice",
            limit=10,
        )

    def test_langchain_load_with_user_id(
        self, mock_neural_mem, sample_search_results
    ):
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainMemory,
        )

        mock_neural_mem.recall.return_value = sample_search_results
        mem = NeuralMemLangChainMemory(
            mock_neural_mem, user_id="bob", k=3
        )
        mem.load_memory_variables({"query": "test"})

        mock_neural_mem.recall.assert_called_once_with(
            "test", user_id="bob", limit=3,
        )

    def test_langchain_save_context(self, mock_neural_mem):
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainMemory,
        )

        mem = NeuralMemLangChainMemory(
            mock_neural_mem, user_id="alice"
        )
        mem.save_context(
            {"input": "Hello"}, {"output": "Hi there!"}
        )

        assert mock_neural_mem.remember.call_count == 2
        calls = mock_neural_mem.remember.call_args_list
        assert "Human: Hello" in calls[0][0][0]
        assert "AI: Hi there!" in calls[1][0][0]

    def test_langchain_save_context_with_both_inputs_outputs(
        self, mock_neural_mem
    ):
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainMemory,
        )

        mem = NeuralMemLangChainMemory(
            mock_neural_mem, user_id="alice"
        )
        mem.save_context(
            {"input": "question", "extra": "data"},
            {"output": "answer", "debug": "info"},
        )

        calls = mock_neural_mem.remember.call_args_list
        assert "Human: question" in calls[0][0][0]
        assert "AI: answer" in calls[1][0][0]
        # Verify user_id is passed
        assert calls[0].kwargs["user_id"] == "alice"

    def test_langchain_clear(self, mock_neural_mem):
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainMemory,
        )

        mem = NeuralMemLangChainMemory(
            mock_neural_mem, user_id="alice"
        )
        mem.clear()

        mock_neural_mem.forget.assert_called_once_with(
            user_id="alice"
        )

    def test_langchain_memory_with_mock_neural_mem(self, mock_neural_mem):
        """Verify the adapter works entirely with mocked NeuralMem."""
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainMemory,
        )

        mock_neural_mem.recall.return_value = []
        mem = NeuralMemLangChainMemory(mock_neural_mem)

        # All operations should work without error
        result = mem.load_memory_variables({"input": "test"})
        assert result == {"history": ""}

        mem.save_context({"input": "hi"}, {"output": "hello"})
        assert mock_neural_mem.remember.call_count == 2

        mem.clear()
        mock_neural_mem.forget.assert_called_once()

    def test_langchain_memory_recall_format(
        self, mock_neural_mem, sample_search_results
    ):
        """Verify the recall result is formatted as newline-joined strings."""
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainMemory,
        )

        mock_neural_mem.recall.return_value = sample_search_results
        mem = NeuralMemLangChainMemory(mock_neural_mem)

        result = mem.load_memory_variables({"input": "preferences"})
        lines = result["history"].split("\n")
        assert len(lines) == 2
        assert lines[0] == "User likes Python"
        assert lines[1] == "User works on AI projects"

    def test_langchain_load_empty_input(self, mock_neural_mem):
        """When no recognizable input key exists, use default query."""
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainMemory,
        )

        mock_neural_mem.recall.return_value = []
        mem = NeuralMemLangChainMemory(mock_neural_mem)

        result = mem.load_memory_variables({})
        mock_neural_mem.recall.assert_called_once_with(
            "recent conversation history",
            user_id="default",
            limit=10,
        )
        assert result == {"history": ""}


# --- NeuralMemLangChainChatHistory Tests ---


class TestLangChainChatHistory:
    def test_langchain_chat_history_init(self, mock_neural_mem):
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainChatHistory,
        )

        hist = NeuralMemLangChainChatHistory(
            mock_neural_mem, user_id="alice"
        )
        assert hist.neural_mem is mock_neural_mem
        assert hist.user_id == "alice"

    def test_langchain_chat_history_add_user_message(
        self, mock_neural_mem
    ):
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainChatHistory,
        )

        hist = NeuralMemLangChainChatHistory(
            mock_neural_mem, user_id="alice"
        )
        hist.add_user_message("Hello!")

        mock_neural_mem.remember.assert_called_once()
        args = mock_neural_mem.remember.call_args
        assert "Human: Hello!" in args[0][0]
        assert args.kwargs["user_id"] == "alice"
        assert "human" in args.kwargs["tags"]

    def test_langchain_chat_history_add_ai_message(
        self, mock_neural_mem
    ):
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainChatHistory,
        )

        hist = NeuralMemLangChainChatHistory(
            mock_neural_mem, user_id="alice"
        )
        hist.add_ai_message("Hi there!")

        mock_neural_mem.remember.assert_called_once()
        args = mock_neural_mem.remember.call_args
        assert "AI: Hi there!" in args[0][0]
        assert "ai" in args.kwargs["tags"]

    def test_langchain_chat_history_messages_property(
        self, mock_neural_mem, chat_search_results
    ):
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainChatHistory,
        )

        mock_neural_mem.recall.return_value = chat_search_results
        hist = NeuralMemLangChainChatHistory(
            mock_neural_mem, user_id="alice"
        )

        messages = hist.messages
        assert len(messages) == 2
        # Without langchain installed, returns dicts
        if isinstance(messages[0], dict):
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Hello!"
            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "Hi there!"

    def test_langchain_chat_history_clear(self, mock_neural_mem):
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainChatHistory,
        )

        hist = NeuralMemLangChainChatHistory(
            mock_neural_mem, user_id="alice"
        )
        hist.clear()

        mock_neural_mem.forget.assert_called_once_with(
            user_id="alice"
        )

    def test_langchain_chat_history_empty(self, mock_neural_mem):
        from neuralmem.integrations.langchain_memory import (
            NeuralMemLangChainChatHistory,
        )

        mock_neural_mem.recall.return_value = []
        hist = NeuralMemLangChainChatHistory(mock_neural_mem)

        messages = hist.messages
        assert messages == []
