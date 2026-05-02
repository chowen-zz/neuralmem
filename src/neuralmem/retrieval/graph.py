"""图谱遍历检索策略"""
from __future__ import annotations

import logging

from neuralmem.core.protocols import GraphStoreProtocol
from neuralmem.retrieval.fusion import RankedItem

_logger = logging.getLogger(__name__)


class GraphStrategy:
    """策略3: 知识图谱遍历 — 通过实体关系找到关联记忆"""

    def __init__(self, graph: GraphStoreProtocol):
        self._graph = graph

    def retrieve(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[RankedItem]:
        try:
            entities = self._graph.find_entities(query)
            if not entities:
                return []
            memory_scores = self._graph.traverse_for_memories(
                entity_ids=[e.id for e in entities],
                depth=2,
                user_id=user_id,
            )
            return [
                RankedItem(id=mid, score=score, method="graph")
                for mid, score in memory_scores[:limit]
            ]
        except Exception as e:
            _logger.warning("Graph search failed: %s", e)
            return []
