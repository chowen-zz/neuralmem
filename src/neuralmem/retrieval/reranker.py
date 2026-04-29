"""CrossEncoderReranker — stub 实现（Week 9 完整实现）"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod

_logger = logging.getLogger(__name__)


class _RerankerBase(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        raise NotImplementedError  # pragma: no cover


class CrossEncoderReranker(_RerankerBase):
    """
    Cross-Encoder 重排序（Week 1-8 stub）。
    # TODO(week-9): 使用 sentence-transformers cross-encoder 实现真实重排序
    接口稳定，Week 9 不会改变方法签名。
    """

    def rerank(self, query: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        _logger.debug("CrossEncoderReranker stub: returning original order (Week 9 TODO)")
        return candidates  # stub: 保持原顺序
