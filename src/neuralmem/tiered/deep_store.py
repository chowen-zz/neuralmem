"""DeepStore - 磁盘冷层存储（低频数据按需加载）

基于 SQLite 实现持久化存储，承载全量记忆数据。
读操作由 TieredManager 按需将数据提升到 HotStore；
写操作接收从 HotStore 淘汰下来的记忆，以及直接写入的冷数据。

接口与 StorageProtocol / TieredStorage 对齐，内部复用 SQLiteStorage 的能力。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import StorageError
from neuralmem.core.types import Memory
from neuralmem.storage.sqlite import SQLiteStorage
from neuralmem.tiered.base import TieredStorage

_logger = logging.getLogger(__name__)


class DeepStore(TieredStorage):
    """磁盘冷层：全量持久化存储，基于 SQLiteStorage 构建。

    Args:
        config: NeuralMemConfig，用于初始化 SQLiteStorage。
    """

    def __init__(self, config: NeuralMemConfig | None = None) -> None:
        if config is None:
            config = NeuralMemConfig()
        self._backend = SQLiteStorage(config)
        self._config = config

    # ------------------------------------------------------------------
    # TieredStorage 实现（委托给 SQLiteStorage）
    # ------------------------------------------------------------------

    def save_memory(self, memory: Memory) -> str:
        return self._backend.save_memory(memory)

    def get_memory(self, memory_id: str) -> Memory | None:
        return self._backend.get_memory(memory_id)

    def update_memory(self, memory_id: str, **kwargs: object) -> None:
        self._backend.update_memory(memory_id, **kwargs)

    def delete_memories(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        before: object = None,
        tags: list[str] | None = None,
        max_importance: float | None = None,
    ) -> int:
        return self._backend.delete_memories(
            memory_id=memory_id,
            user_id=user_id,
            before=before,
            tags=tags,
            max_importance=max_importance,
        )

    def vector_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        memory_types: list[object] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        return self._backend.vector_search(
            vector=vector,
            user_id=user_id,
            memory_types=memory_types,  # type: ignore[arg-type]
            limit=limit,
        )

    def keyword_search(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[object] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        return self._backend.keyword_search(
            query=query,
            user_id=user_id,
            memory_types=memory_types,  # type: ignore[arg-type]
            limit=limit,
        )

    def temporal_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        time_range: tuple[object, object] | None = None,
        recency_weight: float = 0.3,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        return self._backend.temporal_search(
            vector=vector,
            user_id=user_id,
            time_range=time_range,
            recency_weight=recency_weight,
            limit=limit,
        )

    def find_similar(
        self, vector: list[float], user_id: str | None = None, threshold: float = 0.95
    ) -> list[Memory]:
        return self._backend.find_similar(vector, user_id=user_id, threshold=threshold)

    def record_access(self, memory_id: str) -> None:
        self._backend.record_access(memory_id)

    def batch_record_access(self, memory_ids: list[str]) -> None:
        self._backend.batch_record_access(memory_ids)

    def get_stats(self, user_id: str | None = None) -> dict[str, object]:
        stats = self._backend.get_stats(user_id=user_id)
        stats["tier"] = "deep"
        return stats

    def list_memories(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Memory]:
        return self._backend.list_memories(user_id=user_id, limit=limit)

    # ------------------------------------------------------------------
    # 分层迁移钩子
    # ------------------------------------------------------------------

    def pop_for_eviction(self, limit: int = 1) -> list[Memory]:
        """DeepStore 作为最底层，不需要向上淘汰；返回空列表。"""
        return []

    def ingest_from_upper(self, memories: list[Memory]) -> None:
        """接收从 HotStore 淘汰下来的记忆，持久化到 SQLite。"""
        for mem in memories:
            try:
                self._backend.save_memory(mem)
            except Exception as exc:
                _logger.warning("DeepStore ingest failed for %s: %s", mem.id, exc)

    def peek_access_metadata(self, memory_id: str) -> dict[str, Any] | None:
        """仅读取访问元数据，不加载完整 Memory 对象。"""
        rows = self._backend._fetchall(
            "SELECT access_count, last_accessed FROM memories WHERE id = ?",
            (memory_id,),
        )
        if not rows:
            return None
        row = rows[0]
        return {
            "access_count": row["access_count"],
            "last_accessed": (
                datetime.fromisoformat(str(row["last_accessed"]))
                if row["last_accessed"]
                else None
            ),
        }

    # ------------------------------------------------------------------
    # 图谱快照（透传）
    # ------------------------------------------------------------------

    def load_graph_snapshot(self) -> dict | None:
        return self._backend.load_graph_snapshot()

    def save_graph_snapshot(self, data: dict) -> None:
        self._backend.save_graph_snapshot(data)

    # ------------------------------------------------------------------
    # 生命周期（透传）
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        return self._backend.cleanup_expired()

    # ------------------------------------------------------------------
    # 额外管理接口
    # ------------------------------------------------------------------

    def fetch_for_promotion(
        self,
        memory_ids: list[str],
    ) -> list[Memory]:
        """按 ID 列表批量加载记忆，用于 TieredManager 提升到 HotStore。"""
        results: list[Memory] = []
        for mid in memory_ids:
            mem = self._backend.get_memory(mid)
            if mem:
                results.append(mem)
        return results

    def get_cold_candidates(
        self,
        user_id: str | None = None,
        limit: int = 100,
        min_access_count: int = 0,
        max_access_count: int = 2,
    ) -> list[tuple[str, Memory]]:
        """获取适合驻留 DeepStore 的"冷"候选记忆（低访问频率）。

        用于 TieredManager 在启动预热或动态平衡时决定哪些数据
        不值得提升到 HotStore。
        """
        memories = self._backend.list_memories(user_id=user_id, limit=limit * 2)
        candidates: list[tuple[str, Memory]] = []
        for mem in memories:
            if min_access_count <= mem.access_count <= max_access_count:
                candidates.append((mem.id, mem))
            if len(candidates) >= limit:
                break
        return candidates
