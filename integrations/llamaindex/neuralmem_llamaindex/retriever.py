"""NeuralMem LlamaIndex BaseRetriever adapter."""
from __future__ import annotations

from typing import Any

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from neuralmem.core.types import SearchResult


class NeuralMemRetriever(BaseRetriever):
    """
    LlamaIndex BaseRetriever backed by NeuralMem.

    Usage:
        from neuralmem import NeuralMem
        from neuralmem_llamaindex import NeuralMemRetriever

        mem = NeuralMem()
        retriever = NeuralMemRetriever(mem=mem, user_id="alice", k=5)
        nodes = retriever.retrieve("user preferences")
    """

    def __init__(
        self,
        mem: Any,
        user_id: str = "default",
        k: int = 5,
        min_score: float = 0.3,
        **kwargs: Any,
    ) -> None:
        self._mem = mem
        self._user_id = user_id
        self._k = k
        self._min_score = min_score
        super().__init__(**kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        results = self._mem.recall(
            query_bundle.query_str,
            user_id=self._user_id,
            limit=self._k,
            min_score=self._min_score,
        )
        return [_to_node_with_score(r) for r in results]


def _to_node_with_score(result: SearchResult) -> NodeWithScore:
    """Convert a NeuralMem SearchResult to a LlamaIndex NodeWithScore."""
    return NodeWithScore(
        node=TextNode(
            text=result.memory.content,
            metadata={
                "memory_id": result.memory.id,
                "memory_type": result.memory.memory_type.value,
                "tags": list(result.memory.tags),
                "created_at": result.memory.created_at.isoformat(),
                "user_id": result.memory.user_id,
            },
        ),
        score=result.score,
    )
