"""TieredManager - 分层记忆管理器

自动在 HotStore（内存 LRU）和 DeepStore（磁盘 SQLite）之间迁移数据：
- 读未命中：从 DeepStore 加载到 HotStore（promotion）
- 写操作：先写 HotStore + write-through DeepStore
- 热层满：LRU 淘汰到 DeepStore（eviction）
- 支持批量预热、访问频率统计、动态再平衡。

对外暴露与 StorageProtocol 一致的接口，可无缝替换现有 StorageBackend。
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import StorageError
from neuralmem.core.types import Memory
from neuralmem.tiered.base import TieredStorage
from neuralmem.tiered.deep_store import DeepStore
from neuralmem.tiered.hot_store import HotStore

_logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TieredManager(TieredStorage):
    """分层记忆管理器。

    Args:
        config: 配置对象，用于初始化 DeepStore。
        hot_capacity: HotStore 最大容量。
        promotion_threshold: 连续 miss 多少次后触发提升（默认 1，即立即提升）。
        access_window: 访问频率滑动窗口大小（用于再平衡决策）。
    """

    def __init__(
        self,
        config: NeuralMemConfig | None = None,
        hot_capacity: int = 10_000,
        promotion_threshold: int = 1,
        access_window: int = 100,
    ) -> None:
        self._deep = DeepStore(config)
        self._hot = HotStore(capacity=hot_capacity, deep_store=self._deep)
        self._promotion_threshold = max(1, promotion_threshold)
        self._access_window = access_window
        # miss 计数器：记录连续未命中次数，用于延迟提升决策
        self._miss_counts: dict[str, int] = {}
        self._miss_lock = threading.Lock()
        # 全局统计
        self._promotions = 0
        self._evictions = 0
        self._stat_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 内部：提升 / 淘汰
    # ------------------------------------------------------------------

    def _promote(self, memory_id: str) -> Memory | None:
        """从 DeepStore 提升到 HotStore。"""
        mem = self._deep.get_memory(memory_id)
        if mem is None:
            return None
        # 使用 HotStore 的 save_memory 会触发 write-through，但 deep 已有数据，
        # 为避免重复写入，直接操作 HotStore 内部缓存
        self._hot._cache[mem.id] = mem  # type: ignore[attr-defined]
        self._hot._meta[mem.id] = {  # type: ignore[attr-defined]
            "access_count": 0,
            "last_accessed": mem.last_accessed,
        }
        self._hot._touch(mem.id)  # type: ignore[attr-defined]
        with self._stat_lock:
            self._promotions += 1
        _logger.debug("Promoted memory %s to HotStore", memory_id)
        return mem

    def _maybe_promote(self, memory_id: str) -> Memory | None:
        """根据 miss 计数决定是否提升到 HotStore。"""
        with self._miss_lock:
            count = self._miss_counts.get(memory_id, 0) + 1
            self._miss_counts[memory_id] = count
            if count < self._promotion_threshold:
                return None
            self._miss_counts.pop(memory_id, None)
        return self._promote(memory_id)

    def _evict_and_flush(self) -> None:
        """手动触发 HotStore 淘汰并刷回 DeepStore（通常由 HotStore 自动处理）。"""
        evicted = self._hot.pop_for_eviction(limit=1)
        if evicted:
            self._deep.ingest_from_upper(evicted)
            with self._stat_lock:
                self._evictions += len(evicted)

    # ------------------------------------------------------------------
    # TieredStorage 实现
    # ------------------------------------------------------------------

    def save_memory(self, memory: Memory) -> str:
        """保存记忆：写入 HotStore（自动 write-through DeepStore）。"""
        return self._hot.save_memory(memory)

    def get_memory(self, memory_id: str) -> Memory | None:
        """读取记忆：先查 HotStore，miss 则查 DeepStore 并可能提升。"""
        mem = self._hot.get_memory(memory_id)
        if mem is not None:
            return mem
        # HotStore miss，查 DeepStore
        mem = self._deep.get_memory(memory_id)
        if mem is None:
            return None
        # 根据策略决定是否提升到 HotStore
        promoted = self._maybe_promote(memory_id)
        if promoted is not None:
            return promoted
        # 不提升，但返回 DeepStore 的数据
        return mem

    def update_memory(self, memory_id: str, **kwargs: object) -> None:
        """更新记忆：先尝试更新 HotStore，同时透传 DeepStore。"""
        self._hot.update_memory(memory_id, **kwargs)

    def delete_memories(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        before: object = None,
        tags: list[str] | None = None,
        max_importance: float | None = None,
    ) -> int:
        """删除记忆：同时清理 HotStore 和 DeepStore。"""
        return self._hot.delete_memories(
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
        """向量搜索：合并 HotStore 和 DeepStore 结果，去重归一化。"""
        hot_results = self._hot.vector_search(
            vector=vector,
            user_id=user_id,
            memory_types=memory_types,
            limit=limit,
        )
        # 若 HotStore 结果充足，直接返回
        if len(hot_results) >= limit:
            return hot_results[:limit]

        deep_results = self._deep.vector_search(
            vector=vector,
            user_id=user_id,
            memory_types=memory_types,
            limit=limit,
        )
        # 合并去重：HotStore 结果优先（分数更高或相等时保留热层）
        merged: dict[str, float] = {}
        for mid, score in hot_results:
            merged[mid] = max(merged.get(mid, 0.0), score)
            # 记录访问
            self._hot.record_access(mid)
        for mid, score in deep_results:
            if mid in merged:
                # 若 DeepStore 分数更高，更新；否则保留 HotStore 分数
                merged[mid] = max(merged[mid], score)
            else:
                merged[mid] = score

        sorted_results = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]

    def keyword_search(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[object] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """关键词搜索：合并两层结果。"""
        hot_results = self._hot.keyword_search(
            query=query,
            user_id=user_id,
            memory_types=memory_types,
            limit=limit,
        )
        if len(hot_results) >= limit:
            return hot_results[:limit]

        deep_results = self._deep.keyword_search(
            query=query,
            user_id=user_id,
            memory_types=memory_types,
            limit=limit,
        )
        merged: dict[str, float] = {}
        for mid, score in hot_results:
            merged[mid] = max(merged.get(mid, 0.0), score)
        for mid, score in deep_results:
            merged[mid] = max(merged.get(mid, 0.0), score)

        sorted_results = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]

    def temporal_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        time_range: tuple[object, object] | None = None,
        recency_weight: float = 0.3,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """时间加权搜索：合并两层结果。"""
        hot_results = self._hot.temporal_search(
            vector=vector,
            user_id=user_id,
            time_range=time_range,
            recency_weight=recency_weight,
            limit=limit,
        )
        if len(hot_results) >= limit:
            return hot_results[:limit]

        deep_results = self._deep.temporal_search(
            vector=vector,
            user_id=user_id,
            time_range=time_range,
            recency_weight=recency_weight,
            limit=limit,
        )
        merged: dict[str, float] = {}
        for mid, score in hot_results:
            merged[mid] = max(merged.get(mid, 0.0), score)
        for mid, score in deep_results:
            merged[mid] = max(merged.get(mid, 0.0), score)

        sorted_results = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]

    def find_similar(
        self, vector: list[float], user_id: str | None = None, threshold: float = 0.95
    ) -> list[Memory]:
        """查找相似记忆：先查 HotStore，不足时查 DeepStore。"""
        hot_similar = self._hot.find_similar(vector, user_id=user_id, threshold=threshold)
        if len(hot_similar) >= 10:
            return hot_similar

        deep_similar = self._deep.find_similar(vector, user_id=user_id, threshold=threshold)
        # 合并去重（按 ID）
        seen = {mem.id for mem in hot_similar}
        merged = list(hot_similar)
        for mem in deep_similar:
            if mem.id not in seen:
                merged.append(mem)
                seen.add(mem.id)
        return merged

    def record_access(self, memory_id: str) -> None:
        """记录访问：优先更新 HotStore，若不在热层则更新 DeepStore。"""
        self._hot.record_access(memory_id)
        # 若 HotStore 未命中（无该 memory_id），也记录到 DeepStore
        if memory_id not in self._hot._cache:  # type: ignore[attr-defined]
            self._deep.record_access(memory_id)

    def batch_record_access(self, memory_ids: list[str]) -> None:
        """批量记录访问。"""
        self._hot.batch_record_access(memory_ids)
        # 找出不在热层的 ID，批量记录到 DeepStore
        hot_ids = set(self._hot._cache.keys())  # type: ignore[attr-defined]
        cold_ids = [mid for mid in memory_ids if mid not in hot_ids]
        if cold_ids:
            self._deep.batch_record_access(cold_ids)

    def get_stats(self, user_id: str | None = None) -> dict[str, object]:
        """合并两层的统计信息。"""
        hot_stats = self._hot.get_stats(user_id=user_id)
        deep_stats = self._deep.get_stats(user_id=user_id)
        with self._stat_lock:
            return {
                "tier": "tiered",
                "hot": hot_stats,
                "deep": deep_stats,
                "promotions": self._promotions,
                "evictions": self._evictions,
                "total": hot_stats.get("total", 0) + deep_stats.get("total", 0),
            }

    def list_memories(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Memory]:
        """列出所有记忆：先 HotStore，再 DeepStore，去重。"""
        hot_mems = self._hot.list_memories(user_id=user_id, limit=limit)
        hot_ids = {mem.id for mem in hot_mems}
        remaining = limit - len(hot_mems)
        if remaining <= 0:
            return hot_mems
        deep_mems = self._deep.list_memories(user_id=user_id, limit=remaining)
        merged = list(hot_mems)
        for mem in deep_mems:
            if mem.id not in hot_ids:
                merged.append(mem)
        return merged

    # ------------------------------------------------------------------
    # 分层迁移钩子
    # ------------------------------------------------------------------

    def pop_for_eviction(self, limit: int = 1) -> list[Memory]:
        """从 HotStore 淘汰记忆到 DeepStore。"""
        evicted = self._hot.pop_for_eviction(limit=limit)
        if evicted:
            self._deep.ingest_from_upper(evicted)
            with self._stat_lock:
                self._evictions += len(evicted)
        return evicted

    def ingest_from_upper(self, memories: list[Memory]) -> None:
        """TieredManager 作为顶层聚合器，不接收来自上层的记忆。"""
        _logger.debug("TieredManager.ingest_from_upper called (no-op)")

    def peek_access_metadata(self, memory_id: str) -> dict[str, Any] | None:
        """先查 HotStore 元数据，miss 则查 DeepStore。"""
        meta = self._hot.peek_access_metadata(memory_id)
        if meta is not None:
            return meta
        return self._deep.peek_access_metadata(memory_id)

    # ------------------------------------------------------------------
    # 图谱快照（透传 DeepStore）
    # ------------------------------------------------------------------

    def load_graph_snapshot(self) -> dict | None:
        return self._deep.load_graph_snapshot()

    def save_graph_snapshot(self, data: dict) -> None:
        self._deep.save_graph_snapshot(data)

    # ------------------------------------------------------------------
    # 生命周期（透传 DeepStore）
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """清理 DeepStore 中的过期记忆；同时清理 HotStore 中已失效的条目。"""
        count = self._deep.cleanup_expired()
        # 同步清理 HotStore：检查 expires_at
        now = datetime.now(timezone.utc)
        with self._hot._lock:  # type: ignore[attr-defined]
            to_remove = [
                mid
                for mid, mem in self._hot._cache.items()  # type: ignore[attr-defined]
                if mem.expires_at is not None and mem.expires_at <= now
            ]
            for mid in to_remove:
                del self._hot._cache[mid]  # type: ignore[attr-defined]
                self._hot._meta.pop(mid, None)  # type: ignore[attr-defined]
        count += len(to_remove)
        return count

    # ------------------------------------------------------------------
    # 管理接口：预热 / 再平衡 / 诊断
    # ------------------------------------------------------------------

    def warm_hot_store(
        self,
        memory_ids: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> int:
        """预热 HotStore：从 DeepStore 加载指定记忆到热层。

        Args:
            memory_ids: 指定 ID 列表；若提供则忽略 user_id / limit。
            user_id: 按用户加载最近访问的记忆。
            limit: 加载条数上限。

        Returns:
            实际加载条数。
        """
        if memory_ids:
            memories = self._deep.fetch_for_promotion(memory_ids)
        else:
            # 加载最近访问的记忆作为预热候选
            all_mems = self._deep.list_memories(user_id=user_id, limit=limit * 2)
            all_mems.sort(key=lambda m: m.last_accessed, reverse=True)
            memories = all_mems[:limit]

        if not memories:
            return 0
        self._hot.warm(memories)
        return len(memories)

    def rebalance(self, target_hot_ratio: float = 0.1) -> dict[str, int]:
        """根据访问频率动态再平衡两层数据。

        将 DeepStore 中近期高频访问的记忆提升到 HotStore，
        将 HotStore 中低频的淘汰到 DeepStore。

        Returns:
            {"promoted": int, "evicted": int}
        """
        promoted = 0
        evicted = 0

        # 1. 从 DeepStore 找出高访问频率候选
        deep_mems = self._deep.list_memories(limit=self._hot._capacity)  # type: ignore[attr-defined]
        deep_mems.sort(key=lambda m: (m.access_count, m.last_accessed), reverse=True)
        hot_ids = set(self._hot._cache.keys())  # type: ignore[attr-defined]
        promotion_candidates = [m for m in deep_mems if m.id not in hot_ids]

        # 2. 计算目标热层大小
        total = len(deep_mems) + len(self._hot._cache)  # type: ignore[attr-defined]
        target_hot = max(1, int(total * target_hot_ratio))
        current_hot = len(self._hot._cache)  # type: ignore[attr-defined]

        # 3. 若热层有空位，提升候选
        free_slots = target_hot - current_hot
        if free_slots > 0:
            to_promote = promotion_candidates[:free_slots]
            for mem in to_promote:
                self._promote(mem.id)
                promoted += 1

        # 4. 若热层超载，淘汰
        overload = current_hot + promoted - target_hot
        if overload > 0:
            evicted_mems = self._hot.pop_for_eviction(limit=overload)
            self._deep.ingest_from_upper(evicted_mems)
            evicted = len(evicted_mems)
            with self._stat_lock:
                self._evictions += evicted

        return {"promoted": promoted, "evicted": evicted}

    def diagnose(self) -> dict[str, Any]:
        """诊断当前分层状态，返回可读统计。"""
        stats = self.get_stats()
        hot_ids = set(self._hot._cache.keys())  # type: ignore[attr-defined]
        return {
            "hot_capacity": self._hot._capacity,  # type: ignore[attr-defined]
            "hot_size": len(hot_ids),
            "deep_total": stats["deep"].get("total", 0),
            "promotions": stats["promotions"],
            "evictions": stats["evictions"],
            "hit_rate": stats["hot"].get("hit_rate", 0.0),
        }
