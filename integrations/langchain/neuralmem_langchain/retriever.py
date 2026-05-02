"""NeuralMem LangChain BaseRetriever adapter."""
from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from neuralmem.core.types import SearchResult


class NeuralMemRetriever(BaseRetriever):
    """
    LangChain BaseRetriever backed by NeuralMem.

    Usage:
        from neuralmem import NeuralMem
        from neuralmem_langchain import NeuralMemRetriever

        mem = NeuralMem()
        retriever = NeuralMemRetriever(mem=mem, user_id="alice", k=5)

        # Sync:
        docs = retriever.invoke("user preferences")

        # Async (in LCEL chain):
        docs = await retriever.ainvoke("user preferences")
    """

    mem: Any
    user_id: str = "default"
    k: int = 5
    min_score: float = 0.3

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        results = self.mem.recall(
            query,
            user_id=self.user_id,
            limit=self.k,
            min_score=self.min_score,
        )
        return [_to_document(r) for r in results]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        results = await asyncio.to_thread(
            self.mem.recall,
            query,
            user_id=self.user_id,
            limit=self.k,
            min_score=self.min_score,
        )
        return [_to_document(r) for r in results]


def _to_document(result: SearchResult) -> Document:
    """Convert a NeuralMem SearchResult to a LangChain Document."""
    return Document(
        page_content=result.memory.content,
        metadata={
            "memory_id": result.memory.id,
            "score": result.score,
            "retrieval_method": result.retrieval_method,
            "memory_type": result.memory.memory_type.value,
            "tags": list(result.memory.tags),
            "created_at": result.memory.created_at.isoformat(),
            "user_id": result.memory.user_id,
        },
    )
