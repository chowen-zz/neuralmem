"""BM25 关键词检索策略"""
from __future__ import annotations

import logging

from neuralmem.core.protocols import StorageProtocol
from neuralmem.core.types import MemoryType
from neuralmem.retrieval.fusion import RankedItem

_logger = logging.getLogger(__name__)


class KeywordStrategy:
    """策略2: BM25 关键词搜索 — 捕获精确术语匹配"""

    def __init__(self, storage: StorageProtocol):
        self._storage = storage

    def retrieve(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 20,
    ) -> list[RankedItem]:
        try:
            results = self._storage.keyword_search(
                query=query, user_id=user_id, memory_types=memory_types, limit=limit
            )
            return [RankedItem(id=mid, score=score, method="keyword") for mid, score in results]
        except Exception as e:
            _logger.warning("Keyword search failed: %s", e)
            return []
