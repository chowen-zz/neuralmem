"""存储后端抽象基类"""
from __future__ import annotations
from abc import ABC, abstractmethod
from neuralmem.core.types import Memory, MemoryType


class StorageBackend(ABC):
    """所有存储后端必须继承此类"""

    @abstractmethod
    def save_memory(self, memory: Memory) -> str: ...

    @abstractmethod
    def get_memory(self, memory_id: str) -> Memory | None: ...

    @abstractmethod
    def update_memory(self, memory_id: str, **kwargs: object) -> None: ...

    @abstractmethod
    def delete_memories(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        before: object = None,
        tags: list[str] | None = None,
        max_importance: float | None = None,
    ) -> int: ...

    @abstractmethod
    def vector_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]: ...

    @abstractmethod
    def keyword_search(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]: ...

    @abstractmethod
    def temporal_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        time_range: tuple[object, object] | None = None,
        recency_weight: float = 0.3,
        limit: int = 10,
    ) -> list[tuple[str, float]]: ...

    @abstractmethod
    def find_similar(
        self, vector: list[float], user_id: str | None = None, threshold: float = 0.95
    ) -> list[Memory]: ...

    @abstractmethod
    def record_access(self, memory_id: str) -> None: ...

    @abstractmethod
    def get_stats(self, user_id: str | None = None) -> dict[str, object]: ...

    @abstractmethod
    def list_memories(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Memory]: ...

    @abstractmethod
    def load_graph_snapshot(self) -> dict | None: ...

    @abstractmethod
    def save_graph_snapshot(self, data: dict) -> None: ...
