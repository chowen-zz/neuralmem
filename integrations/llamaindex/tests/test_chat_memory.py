"""Tests for NeuralMemChatMemory."""
from __future__ import annotations

import pytest
from llama_index.core.llms import ChatMessage, MessageRole

from neuralmem_llamaindex import NeuralMemChatMemory


def test_put_calls_remember(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem)
    msg = ChatMessage(role=MessageRole.USER, content="Hello, I like Python")
    memory.put(msg)
    mock_mem.remember.assert_called_once()
    call_args = mock_mem.remember.call_args
    assert call_args.args[0] == "Hello, I like Python"
    assert call_args.kwargs["user_id"] == "default"


def test_put_tags_role(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem)
    msg = ChatMessage(role=MessageRole.ASSISTANT, content="Got it")
    memory.put(msg)
    tags = mock_mem.remember.call_args.kwargs["tags"]
    assert "assistant" in tags


def test_get_with_input_calls_recall(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem)
    result = memory.get(input="user preferences")
    mock_mem.recall.assert_called_once()
    assert mock_mem.recall.call_args.args[0] == "user preferences"
    assert isinstance(result, list)
    assert len(result) == 1


def test_get_without_input_uses_storage(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem)
    result = memory.get(input=None)
    mock_mem.storage.list_memories.assert_called_once()
    assert isinstance(result, list)


def test_get_returns_chat_messages(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem)
    result = memory.get(input="query")
    assert len(result) == 1
    assert isinstance(result[0], ChatMessage)
    assert result[0].content == "User prefers Python for backend development"


def test_reset_calls_forget(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem, user_id="alice")
    memory.reset()
    mock_mem.forget.assert_called_once_with(user_id="alice")


def test_set_resets_then_puts(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem)
    messages = [
        ChatMessage(role=MessageRole.USER, content="msg1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="msg2"),
    ]
    memory.set(messages)
    mock_mem.forget.assert_called_once()
    assert mock_mem.remember.call_count == 2


def test_get_all_uses_storage(mock_mem):
    memory = NeuralMemChatMemory(mem=mock_mem)
    result = memory.get_all()
    mock_mem.storage.list_memories.assert_called_once()
    assert isinstance(result, list)
