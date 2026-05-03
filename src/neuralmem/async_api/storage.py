"""AsyncStorage — 异步存储包装器，将同步 StorageBackend 包装为 async API。

使用 asyncio.to_thread 将阻塞的 I/O 操作（SQLite）卸载到线程池中执行，
保持主事件循环的响应性。所有方法签名与 StorageBackend 保持一致，仅添加 async 前缀。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from neuralmem.core.types import Memory, MemoryType
from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)


class AsyncStorage:
    """异步存储包装器 — 将同步 StorageBackend 包装为 async 接口。

    所有底层 I/O 操作通过 asyncio.to_thread 在线程池中执行，
    避免阻塞 asyncio 事件循环。支持批量操作的并发执行优化。
    """

    def __init__(self, storage: StorageBackend) -> None:
        self._storage = storage

    # ------------------------------------------------------------------
    # 核心 CRUD
    # ------------------------------------------------------------------

    async def save_memory(self, memory: Memory) -> str:
        """异步保存记忆到存储后端。"""
        return await asyncio.to_thread(self._storage.save_memory, memory)

    async def get_memory(self, memory_id: str) -> Memory | None:
        """异步按 ID 获取记忆。"""
        return await asyncio.to_thread(self._storage.get_memory, memory_id)

    async def update_memory(self, memory_id: str, **kwargs: Any) -> None:
        """异步更新记忆字段。"""
        return await asyncio.to_thread(self._storage.update_memory, memory_id, **kwargs)

    async def delete_memories(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        before: object = None,
        tags: list[str] | None = None,
        max_importance: float | None = None,
    ) -> int:
        """异步删除符合条件的记忆，返回删除数量。"""
        return await asyncio.to_thread(
            self._storage.delete_memories,
            memory_id=memory_id,
            user_id=user_id,
            before=before,
            tags=tags,
            max_importance=max_importance,
        )

    # ------------------------------------------------------------------
    # 搜索接口
    # ------------------------------------------------------------------

    async def vector_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """异步向量相似度搜索。"""
        return await asyncio.to_thread(
            self._storage.vector_search,
            vector,
            user_id=user_id,
            memory_types=memory_types,
            limit=limit,
        )

    async def keyword_search(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """异步关键词搜索（BM25/FTS）。"""
        return await asyncio.to_thread(
            self._storage.keyword_search,
            query,
            user_id=user_id,
            memory_types=memory_types,
            limit=limit,
        )

    async def temporal_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        time_range: tuple[object, object] | None = None,
        recency_weight: float = 0.3,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """异步时序加权搜索。"""
        return await asyncio.to_thread(
            self._storage.temporal_search,
            vector,
            user_id=user_id,
            time_range=time_range,
            recency_weight=recency_weight,
            limit=limit,
        )

    async def find_similar(
        self,
        vector: list[float],
        user_id: str | None = None,
        threshold: float = 0.95,
    ) -> list[Memory]:
        """异步查找相似记忆（用于去重/冲突检测）。"""
        return await asyncio.to_thread(
            self._storage.find_similar,
            vector,
            user_id=user_id,
            threshold=threshold,
        )

    # ------------------------------------------------------------------
    # 访问追踪 & 统计
    # ------------------------------------------------------------------

    async def record_access(self, memory_id: str) -> None:
        """异步记录单条记忆访问。"""
        return await asyncio.to_thread(self._storage.record_access, memory_id)

    async def batch_record_access(self, memory_ids: list[str]) -> None:
        """异步批量记录记忆访问。"""
        return await asyncio.to_thread(self._storage.batch_record_access, memory_ids)

    async def get_stats(self, user_id: str | None = None) -> dict[str, Any]:
        """异步获取存储统计信息。"""
        return await asyncio.to_thread(self._storage.get_stats, user_id)

    async def list_memories(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Memory]:
        """异步列出记忆。"""
        return await asyncio.to_thread(
            self._storage.list_memories, user_id=user_id, limit=limit
        )

    # ------------------------------------------------------------------
    # 历史记录
    # ------------------------------------------------------------------

    async def save_history(
        self,
        memory_id: str,
        old_content: str | None,
        new_content: str,
        event: str,
        metadata: dict | None = None,
    ) -> None:
        """异步保存记忆历史记录。"""
        return await asyncio.to_thread(
            self._storage.save_history,
            memory_id,
            old_content,
            new_content,
            event,
            metadata=metadata,
        )

    async def get_history(self, memory_id: str) -> list[dict]:
        """异步获取记忆历史记录。"""
        return await asyncio.to_thread(self._storage.get_history, memory_id)

    # ------------------------------------------------------------------
    # 图谱快照
    # ------------------------------------------------------------------

    async def load_graph_snapshot(self) -> dict | None:
        """异步加载图谱快照。"""
        return await asyncio.to_thread(self._storage.load_graph_snapshot)

    async def save_graph_snapshot(self, data: dict) -> None:
        """异步保存图谱快照。"""
        return await asyncio.to_thread(self._storage.save_graph_snapshot, data)

    # ------------------------------------------------------------------
    # 批量并发操作（AsyncStorage 特有优化）
    # ------------------------------------------------------------------

    async def batch_get_memories(self, memory_ids: list[str]) -> list[Memory | None]:
        """并发批量获取多条记忆，使用 asyncio.gather 并行执行。"""
        if not memory_ids:
            return []
        return await asyncio.gather(
            *[self.get_memory(mid) for mid in memory_ids],
            return_exceptions=True,
        )

    async def batch_save_memories(self, memories: list[Memory]) -> list[str]:
        """并发批量保存多条记忆，使用 asyncio.gather 并行执行。"""
        if not memories:
            return []
        return await asyncio.gather(
            *[self.save_memory(m) for m in memories],
            return_exceptions=True,
        )

    async def cleanup_expired(self) -> int:
        """异步清理已过期的记忆。

        尝试调用底层 storage 的 cleanup_expired 方法；
        若不存在（旧版本兼容），返回 0。
        """
        if hasattr(self._storage, "cleanup_expired"):
            return await asyncio.to_thread(self._storage.cleanup_expired)  # type: ignore[misc]
        _logger.debug("cleanup_expired not available on underlying storage")
        return 0

    @property
    def underlying(self) -> StorageBackend:
        """返回底层的同步 StorageBackend 实例（用于需要直接访问的场景）。"""
        return self._storage
