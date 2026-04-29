"""MemoryConsolidator — stub 实现（Week 9 完整实现）"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod

_logger = logging.getLogger(__name__)


class _ConsolidatorBase(ABC):
    @abstractmethod
    def merge_similar(self, user_id: str | None = None) -> int:
        raise NotImplementedError  # pragma: no cover


class MemoryConsolidator(_ConsolidatorBase):
    """
    记忆合并去重（Week 1-8 stub）。
    # TODO(week-9): 实现相似记忆聚合 + 去重
    """

    def merge_similar(self, user_id: str | None = None) -> int:
        _logger.debug("MemoryConsolidator.merge_similar stub called (Week 9 TODO)")
        return 0
