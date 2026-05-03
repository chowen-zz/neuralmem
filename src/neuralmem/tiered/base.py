"""TieredStorage ABC - 分层存储抽象基类

为 HotStore（内存热层）和 DeepStore（磁盘冷层）提供统一接口。
符合 StorageProtocol 契约。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from neuralmem.core.types import Memory


class TieredStorage(ABC):
    """分层存储层抽象基类。

    所有分层存储实现（HotStore / DeepStore）必须继承此类。
    提供与 StorageBackend 对齐的核心 CRUD 接口，同时暴露
    分层特有的访问频率统计和迁移相关钩子。
    """

    # ------------------------------------------------------------------
    # 核心 CRUD
    # ------------------------------------------------------------------

    @abstractmethod
    def save_memory(self, memory: Memory) -> str:
        """保存或更新一条记忆，返回 memory_id。"""
        ...

    @abstractmethod
    def get_memory(self, memory_id: str) -> Memory | None:
        """按 ID 读取记忆；若不存在返回 None。"""
        ...

    @abstractmethod
    def update_memory(self, memory_id: str, **kwargs: object) -> None:
        """部分更新记忆字段。"""
        ...

    @abstractmethod
    def delete_memories(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        before: object = None,
        tags: list[str] | None = None,
        max_importance: float | None = None,
    ) -> int:
        """按条件删除记忆，返回删除条数。"""
        ...

    # ------------------------------------------------------------------
    # 搜索
    # ------------------------------------------------------------------

    @abstractmethod
    def vector_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        memory_types: list[object] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """向量相似度搜索，返回 (memory_id, score) 列表。"""
        ...

    @abstractmethod
    def keyword_search(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[object] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """关键词搜索，返回 (memory_id, score) 列表。"""
        ...

    @abstractmethod
    def temporal_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        time_range: tuple[object, object] | None = None,
        recency_weight: float = 0.3,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """时间加权向量搜索。"""
        ...

    @abstractmethod
    def find_similar(
        self, vector: list[float], user_id: str | None = None, threshold: float = 0.95
    ) -> list[Memory]:
        """查找与向量足够相似的记忆对象。"""
        ...

    # ------------------------------------------------------------------
    # 访问统计
    # ------------------------------------------------------------------

    @abstractmethod
    def record_access(self, memory_id: str) -> None:
        """记录一次访问（access_count + 1, last_accessed 更新）。"""
        ...

    @abstractmethod
    def batch_record_access(self, memory_ids: list[str]) -> None:
        """批量记录访问。"""
        ...

    # ------------------------------------------------------------------
    # 元数据 / 管理
    # ------------------------------------------------------------------

    @abstractmethod
    def get_stats(self, user_id: str | None = None) -> dict[str, object]:
        """返回当前层的统计信息。"""
        ...

    @abstractmethod
    def list_memories(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Memory]:
        """列出当前层中的记忆。"""
        ...

    # ------------------------------------------------------------------
    # 分层特有：迁移钩子
    # ------------------------------------------------------------------

    @abstractmethod
    def pop_for_eviction(self, limit: int = 1) -> list[Memory]:
        """取出最适合被淘汰（迁移到更深层）的记忆。

        HotStore 通常按 LRU / 低访问频率选出；
        DeepStore 可返回空列表（无需向上迁移）。
        """
        ...

    @abstractmethod
    def ingest_from_upper(self, memories: list[Memory]) -> None:
        """接收从上层淘汰下来的记忆（写入当前层）。"""
        ...

    @abstractmethod
    def peek_access_metadata(self, memory_id: str) -> dict[str, Any] | None:
        """仅读取访问元数据（access_count, last_accessed），不触发缓存加载。"""
        ...

    # ------------------------------------------------------------------
    # 图谱快照（可选，默认 no-op）
    # ------------------------------------------------------------------

    def load_graph_snapshot(self) -> dict | None:
        """加载图谱快照。默认 no-op。"""
        return None

    def save_graph_snapshot(self, data: dict) -> None:
        """保存图谱快照。默认 no-op。"""
        pass

    # ------------------------------------------------------------------
    # 生命周期（可选，默认 no-op）
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """清理过期记忆。默认 no-op。"""
        return 0
