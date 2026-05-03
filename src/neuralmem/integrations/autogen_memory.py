"""AutoGen memory adapter for NeuralMem.

Provides an AutoGen-compatible memory interface that wraps NeuralMem,
allowing seamless integration with AutoGen agents.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuralmem.core.memory import NeuralMem


class AutoGenMemory:
    """AutoGen memory adapter for NeuralMem.

    Implements AutoGen's memory interface for use with
    AutoGen conversable agents.

    Usage::

        from neuralmem import NeuralMem
        from neuralmem.integrations import AutoGenMemory

        mem = NeuralMem()
        memory = AutoGenMemory(mem, user_id="agent1")
        memory.add("Python is a great language")
        results = memory.search("programming", n_results=3)
    """

    def __init__(
        self,
        neural_mem: NeuralMem,
        user_id: str = "default",
    ) -> None:
        self.neural_mem = neural_mem
        self.user_id = user_id

    def add(self, content: str) -> None:
        """Add content to memory.

        Parameters
        ----------
        content:
            The text content to store.
        """
        self.neural_mem.remember(
            content,
            user_id=self.user_id,
            tags=["autogen"],
            infer=False,
        )

    def search(
        self, query: str, n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Search for relevant memories.

        Parameters
        ----------
        query:
            The search query.
        n_results:
            Maximum number of results.

        Returns
        -------
        list[dict[str, Any]]
            List of dicts with 'content', 'score',
            and 'id'.
        """
        results = self.neural_mem.recall(
            query,
            user_id=self.user_id,
            limit=n_results,
        )
        return [
            {
                "content": r.memory.content,
                "score": r.score,
                "id": r.memory.id,
            }
            for r in results
        ]

    def clear(self) -> None:
        """Clear all memories for this user."""
        self.neural_mem.forget(user_id=self.user_id)
