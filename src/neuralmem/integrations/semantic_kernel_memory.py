"""Semantic Kernel memory adapter for NeuralMem.

Provides a Semantic Kernel-compatible memory interface that wraps
NeuralMem, allowing seamless integration with Semantic Kernel
planners and functions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuralmem.core.memory import NeuralMem


class SemanticKernelMemory:
    """Semantic Kernel memory adapter for NeuralMem.

    Implements Semantic Kernel's memory interface for use with
    SK planners, functions, and semantic memory.

    Usage::

        from neuralmem import NeuralMem
        from neuralmem.integrations import (
            SemanticKernelMemory,
        )

        mem = NeuralMem()
        sk_mem = SemanticKernelMemory(
            mem, user_id="sk_user"
        )
        await sk_mem.save_information(
            "collection", "id1", "Python is great"
        )
        results = await sk_mem.search("programming")
    """

    def __init__(
        self,
        neural_mem: NeuralMem,
        user_id: str = "default",
    ) -> None:
        self.neural_mem = neural_mem
        self.user_id = user_id

    async def save_information(
        self,
        collection: str,
        id: str,
        text: str,
        description: str = "",
        additional_metadata: str = "",
    ) -> None:
        """Save information to memory.

        Parameters
        ----------
        collection:
            The collection/namespace (mapped to tags).
        id:
            The unique identifier (used as memory id).
        text:
            The text content to store.
        description:
            Optional description (stored as source).
        additional_metadata:
            Optional metadata (stored as tag).
        """
        tags = ["semantic_kernel", collection]
        if additional_metadata:
            tags.append(additional_metadata)
        self.neural_mem.remember(
            text,
            user_id=self.user_id,
            tags=tags,
            source=description or None,
            infer=False,
        )

    async def search(
        self,
        query: str,
        collection: str = "",
        limit: int = 5,
        min_relevance_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search for relevant memories.

        Parameters
        ----------
        query:
            The search query.
        collection:
            Optional collection filter (mapped to tags).
        limit:
            Maximum number of results.
        min_relevance_score:
            Minimum relevance score threshold.

        Returns
        -------
        list[dict[str, Any]]
            List of dicts with 'id', 'text', 'score',
            and 'description'.
        """
        results = self.neural_mem.recall(
            query,
            user_id=self.user_id,
            limit=limit,
            min_score=min_relevance_score,
        )
        return [
            {
                "id": r.memory.id,
                "text": r.memory.content,
                "score": r.score,
                "description": r.memory.source or "",
            }
            for r in results
        ]

    async def get_async(
        self,
        collection: str,
        id: str,
    ) -> dict[str, Any] | None:
        """Get a specific memory by id.

        Parameters
        ----------
        collection:
            The collection/namespace.
        id:
            The unique identifier.

        Returns
        -------
        dict[str, Any] | None
            The memory dict or None if not found.
        """
        # Use recall with the id as query
        results = self.neural_mem.recall(
            id,
            user_id=self.user_id,
            limit=100,
            min_score=0.0,
        )
        for r in results:
            if r.memory.id == id:
                return {
                    "id": r.memory.id,
                    "text": r.memory.content,
                    "description": r.memory.source or "",
                }
        return None

    async def remove_async(
        self,
        collection: str,
        id: str,
    ) -> None:
        """Remove a memory by id.

        Parameters
        ----------
        collection:
            The collection/namespace.
        id:
            The unique identifier.
        """
        self.neural_mem.forget(memory_id=id)
