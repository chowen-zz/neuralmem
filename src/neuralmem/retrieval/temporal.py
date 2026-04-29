"""时序加权检索策略"""
from __future__ import annotations
import logging
from neuralmem.core.protocols import StorageProtocol, EmbedderProtocol
from neuralmem.retrieval.fusion import RankedItem

_logger = logging.getLogger(__name__)


class TemporalStrategy:
    """策略4: 时序加权检索 — 近期记忆权重更高"""

    def __init__(self, storage: StorageProtocol, embedder: EmbedderProtocol):
        self._storage = storage
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        user_id: str | None = None,
        time_range: tuple[object, object] | None = None,
        recency_weight: float = 0.3,
        limit: int = 20,
    ) -> list[RankedItem]:
        try:
            vector = self._embedder.encode_one(query)
            results = self._storage.temporal_search(
                vector=vector,
                user_id=user_id,
                time_range=time_range,
                recency_weight=recency_weight,
                limit=limit,
            )
            return [RankedItem(id=mid, score=score, method="temporal") for mid, score in results]
        except Exception as e:
            _logger.warning("Temporal search failed: %s", e)
            return []
