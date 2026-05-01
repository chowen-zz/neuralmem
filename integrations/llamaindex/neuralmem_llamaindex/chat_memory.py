"""NeuralMem LlamaIndex BaseMemory adapter."""
from __future__ import annotations

from typing import Any

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import BaseMemory
from pydantic import PrivateAttr

from neuralmem.core.types import Memory, MemoryType, SearchResult


class NeuralMemChatMemory(BaseMemory):
    """
    LlamaIndex BaseMemory backed by NeuralMem.
    Stores chat messages as EPISODIC memories, with role stored in tags.

    Usage:
        from neuralmem import NeuralMem
        from neuralmem_llamaindex import NeuralMemChatMemory

        mem = NeuralMem()
        memory = NeuralMemChatMemory(mem=mem, user_id="alice")
    """

    _mem: Any = PrivateAttr()
    _user_id: str = PrivateAttr()
    _window_size: int = PrivateAttr()

    def __init__(
        self,
        mem: Any,
        user_id: str = "default",
        window_size: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._mem = mem
        self._user_id = user_id
        self._window_size = window_size

    @classmethod
    def from_defaults(cls, **kwargs: Any) -> NeuralMemChatMemory:  # type: ignore[override]
        return cls(**kwargs)

    def put(self, message: ChatMessage) -> None:
        self._mem.remember(
            message.content,
            user_id=self._user_id,
            memory_type=MemoryType.EPISODIC,
            tags=[message.role.value],
        )

    def get(self, input: str | None = None, **kwargs: Any) -> list[ChatMessage]:
        if input:
            results = self._mem.recall(input, user_id=self._user_id, limit=self._window_size)
            return [_result_to_chat_message(r) for r in results]
        memories = self._mem.storage.list_memories(user_id=self._user_id, limit=self._window_size)
        return [
            _memory_to_chat_message(m)
            for m in sorted(memories, key=lambda m: m.created_at)
        ]

    def get_all(self) -> list[ChatMessage]:
        memories = self._mem.storage.list_memories(user_id=self._user_id, limit=10_000)
        return [
            _memory_to_chat_message(m)
            for m in sorted(memories, key=lambda m: m.created_at)
        ]

    def reset(self) -> None:
        self._mem.forget(user_id=self._user_id)

    def set(self, messages: list[ChatMessage]) -> None:
        self.reset()
        for msg in messages:
            self.put(msg)

    def to_string(self) -> str:
        messages = self.get_all()
        return "\n".join(f"{m.role.value}: {m.content}" for m in messages)


def _result_to_chat_message(result: SearchResult) -> ChatMessage:
    """Convert a recall() SearchResult to a ChatMessage, recovering role from tags."""
    tags = list(result.memory.tags)
    role = _role_from_tags(tags)
    return ChatMessage(role=role, content=result.memory.content)


def _memory_to_chat_message(memory: Memory) -> ChatMessage:
    """Convert a Memory to a ChatMessage for the chronological fallback path."""
    tags = list(memory.tags)
    role = _role_from_tags(tags)
    return ChatMessage(role=role, content=memory.content)


def _role_from_tags(tags: list[str]) -> MessageRole:
    """Recover MessageRole from tags stored during put(). Defaults to USER."""
    for candidate in (MessageRole.ASSISTANT, MessageRole.SYSTEM, MessageRole.USER):
        if candidate.value in tags:
            return candidate
    return MessageRole.USER
