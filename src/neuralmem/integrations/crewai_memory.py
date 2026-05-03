"""CrewAI memory adapter for NeuralMem.

Provides a CrewAI-compatible memory interface that wraps NeuralMem,
allowing seamless integration with CrewAI agents and crews.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuralmem.core.memory import NeuralMem


class CrewAIMemory:
    """CrewAI memory adapter for NeuralMem.

    Implements CrewAI's memory interface for use with
    CrewAI agents and crews.

    Usage::

        from neuralmem import NeuralMem
        from neuralmem.integrations import CrewAIMemory

        mem = NeuralMem()
        memory = CrewAIMemory(mem, user_id="crew1")
        memory.save("query", "result")
        results = memory.search("query", limit=5)
    """

    def __init__(
        self,
        neural_mem: NeuralMem,
        user_id: str = "default",
    ) -> None:
        self.neural_mem = neural_mem
        self.user_id = user_id

    def save(self, query: str, result: str) -> None:
        """Save a query/result pair to NeuralMem.

        Stores both the query and result as separate
        memories for later retrieval.

        Parameters
        ----------
        query:
            The query or task description.
        result:
            The result or output produced.
        """
        if query:
            self.neural_mem.remember(
                f"Query: {query}",
                user_id=self.user_id,
                tags=["crewai", "query"],
                infer=False,
            )
        if result:
            self.neural_mem.remember(
                f"Result: {result}",
                user_id=self.user_id,
                tags=["crewai", "result"],
                infer=False,
            )

    def search(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Search for relevant memories.

        Parameters
        ----------
        query:
            The search query.
        limit:
            Maximum number of results.

        Returns
        -------
        list[dict[str, Any]]
            List of dicts with 'content' and 'score'.
        """
        results = self.neural_mem.recall(
            query, user_id=self.user_id, limit=limit,
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
