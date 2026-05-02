"""Embedding 后端抽象基类"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingBackend(ABC):
    """所有 Embedding 后端必须继承此类"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度（all-MiniLM-L6-v2 = 384）"""
        ...

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码文本为向量"""
        ...

    def encode_one(self, text: str) -> list[float]:
        """编码单条文本（默认实现：调用 encode）"""
        return self.encode([text])[0]
