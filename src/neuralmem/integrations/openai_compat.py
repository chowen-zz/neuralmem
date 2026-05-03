"""OpenAI SDK compatibility layer for NeuralMem.

Wraps NeuralMem in an OpenAI-style resource pattern, letting users
interact with memories using familiar create/list/retrieve/update/delete/search
methods that mirror the OpenAI Python SDK's resource interface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuralmem.core.memory import NeuralMem


class NeuralMemOpenAICompat:
    """OpenAI-compatible memory API wrapper.

    Provides an ``openai.resources``-style interface for NeuralMem,
    making it feel familiar to OpenAI SDK users.

    Usage::

        from neuralmem import NeuralMem
        from neuralmem.integrations import NeuralMemOpenAICompat

        mem = NeuralMem()
        client = NeuralMemOpenAICompat(mem, user_id="alice")

        # Create a memory
        result = client.memories.create("User likes dark mode")

        # List memories
        all_mems = client.memories.list()

        # Search memories
        results = client.memories.search("color preference")
    """

    def __init__(
        self,
        neural_mem: NeuralMem,
        user_id: str = "default",
    ) -> None:
        self._neural_mem = neural_mem
        self._user_id = user_id
        self._memories: NeuralMemOpenAICompat.Memories | None = None

    @property
    def memories(self) -> NeuralMemOpenAICompat.Memories:
        """Lazily initialize and return the Memories resource."""
        if self._memories is None:
            self._memories = self.Memories(
                neural_mem=self._neural_mem,
                user_id=self._user_id,
            )
        return self._memories

    class Memories:
        """Nested class mimicking ``openai.resources.memories``."""

        def __init__(
            self,
            neural_mem: NeuralMem,
            user_id: str = "default",
        ) -> None:
            self._neural_mem = neural_mem
            self._user_id = user_id

        def create(self, content: str, **kwargs: Any) -> dict[str, Any]:
            """Store a new memory and return an OpenAI-style response.

            Args:
                content: The text content to remember.
                **kwargs: Additional arguments passed to remember().

            Returns:
                Dict mimicking an OpenAI resource response.
            """
            memories = self._neural_mem.remember(
                content,
                user_id=self._user_id,
                **kwargs,
            )
            if memories:
                m = memories[0]
                return {
                    "id": m.id,
                    "object": "memory",
                    "content": m.content,
                    "memory_type": m.memory_type.value,
                    "user_id": m.user_id,
                    "importance": m.importance,
                    "created_at": m.created_at.isoformat(),
                }
            return {
                "id": "",
                "object": "memory",
                "content": "",
                "memory_type": "semantic",
                "user_id": self._user_id,
                "importance": 0.0,
                "created_at": "",
            }

        def list(self, **kwargs: Any) -> dict[str, Any]:
            """List all memories for this user.

            Returns:
                OpenAI-style list response with ``data`` array.
            """
            results = self._neural_mem.recall(
                "",
                user_id=self._user_id,
                limit=100,
                min_score=0.0,
            )
            data: list[dict[str, Any]] = []
            for r in results:
                m = r.memory
                data.append({
                    "id": m.id,
                    "object": "memory",
                    "content": m.content,
                    "memory_type": m.memory_type.value,
                    "user_id": m.user_id,
                    "importance": m.importance,
                    "created_at": m.created_at.isoformat(),
                })
            return {"object": "list", "data": data}

        def retrieve(self, memory_id: str) -> dict[str, Any]:
            """Retrieve a single memory by ID.

            Args:
                memory_id: The memory identifier.

            Returns:
                Dict with the memory data.

            Raises:
                ValueError: If the memory is not found.
            """
            m = self._neural_mem.get(memory_id)
            if m is None:
                raise ValueError(
                    f"Memory with id '{memory_id}' not found."
                )
            return {
                "id": m.id,
                "object": "memory",
                "content": m.content,
                "memory_type": m.memory_type.value,
                "user_id": m.user_id,
                "importance": m.importance,
                "created_at": m.created_at.isoformat(),
            }

        def update(
            self, memory_id: str, content: str, **kwargs: Any
        ) -> dict[str, Any]:
            """Update an existing memory's content.

            Args:
                memory_id: The memory to update.
                content: New content text.

            Returns:
                Dict with the updated memory data.

            Raises:
                ValueError: If the memory is not found.
            """
            m = self._neural_mem.update(memory_id, content, **kwargs)
            if m is None:
                raise ValueError(
                    f"Memory with id '{memory_id}' not found."
                )
            return {
                "id": m.id,
                "object": "memory",
                "content": m.content,
                "memory_type": m.memory_type.value,
                "user_id": m.user_id,
                "importance": m.importance,
                "created_at": m.created_at.isoformat(),
            }

        def delete(self, memory_id: str) -> dict[str, Any]:
            """Delete a memory by ID.

            Args:
                memory_id: The memory to delete.

            Returns:
                OpenAI-style deletion confirmation dict.
            """
            self._neural_mem.forget(memory_id=memory_id)
            return {
                "id": memory_id,
                "object": "memory",
                "deleted": True,
            }

        def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
            """Search memories by query.

            Args:
                query: The search query text.
                **kwargs: Additional arguments passed to recall().

            Returns:
                OpenAI-style list response with matching memories.
            """
            results = self._neural_mem.recall(
                query,
                user_id=self._user_id,
                **kwargs,
            )
            data: list[dict[str, Any]] = []
            for r in results:
                m = r.memory
                data.append({
                    "id": m.id,
                    "object": "memory",
                    "content": m.content,
                    "memory_type": m.memory_type.value,
                    "user_id": m.user_id,
                    "importance": m.importance,
                    "score": r.score,
                    "created_at": m.created_at.isoformat(),
                })
            return {"object": "list", "data": data}
