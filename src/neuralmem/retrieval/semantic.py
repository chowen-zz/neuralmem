"""语义向量检索策略"""
from __future__ import annotations
import logging
from neuralmem.core.protocols import StorageProtocol, EmbedderProtocol
from neuralmem.core.types import MemoryType
from neuralmem.retrieval.fusion import RankedItem

_logger = logging.getLogger(__name__)


class SemanticStrategy:
    """策略1: 向量语义搜索 — 捕获语义相近的记忆"""

    def __init__(self, storage: StorageProtocol, embedder: EmbedderProtocol):
        self._storage = storage
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 20,
    ) -> list[RankedItem]:
        try:
            vector = self._embedder.encode_one(query)
            results = self._storage.vector_search(
                vector=vector, user_id=user_id, memory_types=memory_types, limit=limit
            )
            return [RankedItem(id=mid, score=score, method="semantic") for mid, score in results]
        except Exception as e:
            _logger.warning("Semantic search failed: %s", e)
            return []
