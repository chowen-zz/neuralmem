"""ImportanceScorer — stub 实现（Week 9 完整实现）"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from neuralmem.core.types import Memory

_logger = logging.getLogger(__name__)


class _ImportanceBase(ABC):
    @abstractmethod
    def score(self, memory: Memory) -> float:
        raise NotImplementedError  # pragma: no cover


class ImportanceScorer(_ImportanceBase):
    """
    重要性评分（Week 1-8 stub：返回记忆自带的 importance 值）。
    # TODO(week-9): 实现基于访问频次、关联实体数量、时效性的综合评分
    """

    def score(self, memory: Memory) -> float:
        return memory.importance
