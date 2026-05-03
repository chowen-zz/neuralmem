"""HotStore - 内存热层存储（LRU 缓存高频数据）

基于 Python dict + collections.OrderedDict 实现 O(1) LRU 淘汰。
所有写入同时同步到 DeepStore（write-through），确保数据持久性。
读操作优先命中 HotStore，未命中时由 TieredManager 负责从 DeepStore 加载。
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from neuralmem.core.exceptions import StorageError
from neuralmem.core.types import Memory
from neuralmem.tiered.base import TieredStorage

_logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class HotStore(TieredStorage):
    """内存热层：LRU 缓存，容量受限，访问 O(1)。

    Args:
        capacity: 最大缓存条数，默认 10_000。
        deep_store: 底层存储引用，用于 write-through 和 eviction 写入。
    """

    def __init__(self, capacity: int = 10_000, deep_store: TieredStorage | None = None) -> None:
        self._capacity = max(1, capacity)
        self._deep = deep_store
        # OrderedDict 作为 LRU 缓存：move_to_end 标记最近使用
        self._cache: OrderedDict[str, Memory] = OrderedDict()
        # 访问元数据：access_count, last_accessed（与 Memory 对象保持同步）
        self._meta: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._total_hits = 0
        self._total_misses = 0

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _touch(self, memory_id: str) -> None:
        """将 memory_id 标记为最近使用（移至 OrderedDict 末尾）。"""
        if memory_id in self._cache:
            self._cache.move_to_end(memory_id)

    def _evict_if_needed(self) -> list[Memory]:
        """若超出容量，淘汰最久未使用的条目，返回被淘汰的记忆列表。"""
        evicted: list[Memory] = []
        while len(self._cache) > self._capacity:
            oldest_id, oldest_mem = self._cache.popitem(last=False)
            meta = self._meta.pop(oldest_id, {})
            # 同步 access_count / last_accessed 到 memory 对象以便落盘
            oldest_mem = self._sync_meta_to_memory(oldest_mem, meta)
            evicted.append(oldest_mem)
        return evicted

    @staticmethod
    def _sync_meta_to_memory(memory: Memory, meta: dict[str, Any]) -> Memory:
        """将热层累积的访问元数据写回 Memory 对象（用于持久化）。"""
        if not meta:
            return memory
        updates: dict[str, Any] = {}
        if "access_count" in meta:
            updates["access_count"] = memory.access_count + meta["access_count"]
        if "last_accessed" in meta:
            updates["last_accessed"] = meta["last_accessed"]
        if updates:
            return memory.model_copy(update=updates)
        return memory

    def _flush_evicted(self, evicted: list[Memory]) -> None:
        """将被淘汰的记忆写回 DeepStore。"""
        if not self._deep or not evicted:
            return
        for mem in evicted:
            try:
                self._deep.save_memory(mem)
            except Exception as exc:
                _logger.warning("HotStore eviction flush failed for %s: %s", mem.id, exc)

    # ------------------------------------------------------------------
    # TieredStorage 实现
    # ------------------------------------------------------------------

    def save_memory(self, memory: Memory) -> str:
        """保存记忆到热层，同时 write-through 到 DeepStore。"""
        with self._lock:
            self._cache[memory.id] = memory
            self._meta[memory.id] = {
                "access_count": 0,
                "last_accessed": memory.last_accessed,
            }
            self._touch(memory.id)
            evicted = self._evict_if_needed()
        # 在锁外执行磁盘写入，减少锁持有时间
        self._flush_evicted(evicted)
        if self._deep:
            try:
                self._deep.save_memory(memory)
            except Exception as exc:
                _logger.warning("HotStore write-through failed for %s: %s", memory.id, exc)
        return memory.id

    def get_memory(self, memory_id: str) -> Memory | None:
        """从热层读取；命中则更新 LRU 顺序并累加访问计数。"""
        with self._lock:
            mem = self._cache.get(memory_id)
            if mem is not None:
                self._touch(memory_id)
                meta = self._meta[memory_id]
                meta["access_count"] = meta.get("access_count", 0) + 1
                meta["last_accessed"] = datetime.now(timezone.utc)
                self._total_hits += 1
                # 返回时同步最新元数据
                return self._sync_meta_to_memory(mem, meta)
            self._total_misses += 1
            return None

    def update_memory(self, memory_id: str, **kwargs: object) -> None:
        """更新热层中的记忆；若存在则同时 write-through。"""
        with self._lock:
            mem = self._cache.get(memory_id)
            if mem is not None:
                # 重建 Memory 对象（pydantic v2 model_copy）
                updated = mem.model_copy(update=kwargs)
                self._cache[memory_id] = updated
                self._touch(memory_id)
                self._meta.setdefault(memory_id, {})
                self._meta[memory_id]["last_accessed"] = datetime.now(timezone.utc)
                # 在锁外同步到 deep
                mem_to_flush = updated
            else:
                mem_to_flush = None

        if self._deep:
            if mem_to_flush is not None:
                try:
                    self._deep.update_memory(memory_id, **kwargs)
                except Exception as exc:
                    _logger.warning("HotStore update write-through failed: %s", exc)
            else:
                # 热层没有，直接透传
                self._deep.update_memory(memory_id, **kwargs)

    def delete_memories(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        before: object = None,
        tags: list[str] | None = None,
        max_importance: float | None = None,
    ) -> int:
        """删除热层中匹配的记忆，并透传删除条件到 DeepStore。"""
        deleted_count = 0
        with self._lock:
            if memory_id is not None:
                if memory_id in self._cache:
                    del self._cache[memory_id]
                    self._meta.pop(memory_id, None)
                    deleted_count += 1
            else:
                # 按条件扫描（热层数据量小，全量扫描可接受）
                to_remove: list[str] = []
                for mid, mem in self._cache.items():
                    if user_id is not None and mem.user_id != user_id:
                        continue
                    if before is not None:
                        cutoff = before if isinstance(before, datetime) else datetime.fromisoformat(str(before))
                        if mem.created_at < cutoff:
                            continue
                    if tags and not any(t in mem.tags for t in tags):
                        continue
                    if max_importance is not None and mem.importance >= max_importance:
                        continue
                    to_remove.append(mid)
                for mid in to_remove:
                    del self._cache[mid]
                    self._meta.pop(mid, None)
                    deleted_count += 1

        if self._deep:
            try:
                deep_deleted = self._deep.delete_memories(
                    memory_id=memory_id,
                    user_id=user_id,
                    before=before,
                    tags=tags,
                    max_importance=max_importance,
                )
                # 若按 ID 删除且热层已命中，deep_deleted 可能为 0，以热层为准
                if memory_id is None:
                    deleted_count = deep_deleted
            except Exception as exc:
                _logger.warning("HotStore delete write-through failed: %s", exc)
        return deleted_count

    def vector_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        memory_types: list[object] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """在热层内做暴力向量相似度搜索（数据量小，numpy 足够快）。"""
        import numpy as np

        with self._lock:
            candidates: list[tuple[str, np.ndarray]] = []
            for mid, mem in self._cache.items():
                if user_id is not None and mem.user_id != user_id:
                    continue
                if memory_types and mem.memory_type not in memory_types:
                    continue
                if mem.embedding:
                    emb = np.array(mem.embedding, dtype=np.float32)
                    candidates.append((mid, emb))

            if not candidates:
                return []

            query_vec = np.array(vector, dtype=np.float32)
            scored: list[tuple[str, float]] = []
            for mid, emb in candidates:
                norm_q = np.linalg.norm(query_vec)
                norm_e = np.linalg.norm(emb)
                if norm_q == 0 or norm_e == 0:
                    continue
                score = float(np.dot(query_vec, emb) / (norm_q * norm_e))
                scored.append((mid, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:limit]
            for mid, _ in top:
                self._touch(mid)
                self._meta.setdefault(mid, {})
                self._meta[mid]["access_count"] = self._meta[mid].get("access_count", 0) + 1
                self._meta[mid]["last_accessed"] = datetime.now(timezone.utc)
            return top

    def keyword_search(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[object] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """在热层内做简单的关键词包含匹配（大小写不敏感）。"""
        query_lower = query.lower()
        with self._lock:
            scored: list[tuple[str, float]] = []
            for mid, mem in self._cache.items():
                if user_id is not None and mem.user_id != user_id:
                    continue
                if memory_types and mem.memory_type not in memory_types:
                    continue
                score = 0.0
                content_lower = mem.content.lower()
                if query_lower in content_lower:
                    # 简单分数：匹配长度占比
                    score = min(1.0, len(query_lower) / max(1, len(content_lower)) + 0.5)
                # 也匹配 tags
                if any(query_lower in tag.lower() for tag in mem.tags):
                    score = max(score, 0.8)
                if score > 0:
                    scored.append((mid, score))
                    self._touch(mid)
                    self._meta.setdefault(mid, {})
                    self._meta[mid]["access_count"] = self._meta[mid].get("access_count", 0) + 1
                    self._meta[mid]["last_accessed"] = datetime.now(timezone.utc)

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]

    def temporal_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        time_range: tuple[object, object] | None = None,
        recency_weight: float = 0.3,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """热层内的时间加权向量搜索。"""
        import numpy as np

        now = datetime.now(timezone.utc)
        with self._lock:
            candidates: list[tuple[str, Memory, np.ndarray]] = []
            for mid, mem in self._cache.items():
                if user_id is not None and mem.user_id != user_id:
                    continue
                if time_range is not None:
                    start, end = time_range
                    if start is not None:
                        s = start if isinstance(start, datetime) else datetime.fromisoformat(str(start))
                        if mem.created_at < s:
                            continue
                    if end is not None:
                        e = end if isinstance(end, datetime) else datetime.fromisoformat(str(end))
                        if mem.created_at > e:
                            continue
                if mem.embedding:
                    emb = np.array(mem.embedding, dtype=np.float32)
                    candidates.append((mid, mem, emb))

            if not candidates:
                return []

            query_vec = np.array(vector, dtype=np.float32)
            scored: list[tuple[str, float]] = []
            for mid, mem, emb in candidates:
                norm_q = np.linalg.norm(query_vec)
                norm_e = np.linalg.norm(emb)
                if norm_q == 0 or norm_e == 0:
                    continue
                sem_score = float(np.dot(query_vec, emb) / (norm_q * norm_e))
                days_ago = max(0.0, (now - mem.created_at).total_seconds() / 86400)
                time_score = max(0.0, 1.0 - days_ago / 365)
                final = sem_score * (1.0 - recency_weight) + time_score * recency_weight
                scored.append((mid, final))
                self._touch(mid)
                self._meta.setdefault(mid, {})
                self._meta[mid]["access_count"] = self._meta[mid].get("access_count", 0) + 1
                self._meta[mid]["last_accessed"] = now

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]

    def find_similar(
        self, vector: list[float], user_id: str | None = None, threshold: float = 0.95
    ) -> list[Memory]:
        """在热层内查找相似记忆。"""
        candidates = self.vector_search(vector, user_id=user_id, limit=100)
        similar_ids = [cid for cid, score in candidates if score >= threshold]
        if not similar_ids:
            return []
        with self._lock:
            results: list[Memory] = []
            for mid in similar_ids:
                mem = self._cache.get(mid)
                if mem:
                    meta = self._meta.get(mid, {})
                    results.append(self._sync_meta_to_memory(mem, meta))
                    self._touch(mid)
            return results

    def record_access(self, memory_id: str) -> None:
        with self._lock:
            if memory_id in self._cache:
                self._touch(memory_id)
                self._meta.setdefault(memory_id, {})
                self._meta[memory_id]["access_count"] = self._meta[memory_id].get("access_count", 0) + 1
                self._meta[memory_id]["last_accessed"] = datetime.now(timezone.utc)

    def batch_record_access(self, memory_ids: list[str]) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            for mid in memory_ids:
                if mid in self._cache:
                    self._touch(mid)
                    self._meta.setdefault(mid, {})
                    self._meta[mid]["access_count"] = self._meta[mid].get("access_count", 0) + 1
                    self._meta[mid]["last_accessed"] = now

    def get_stats(self, user_id: str | None = None) -> dict[str, object]:
        with self._lock:
            if user_id is not None:
                total = sum(1 for m in self._cache.values() if m.user_id == user_id)
            else:
                total = len(self._cache)
            return {
                "tier": "hot",
                "total": total,
                "capacity": self._capacity,
                "hits": self._total_hits,
                "misses": self._total_misses,
                "hit_rate": self._total_hits / max(1, self._total_hits + self._total_misses),
            }

    def list_memories(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Memory]:
        with self._lock:
            results: list[Memory] = []
            for mid, mem in self._cache.items():
                if user_id is not None and mem.user_id != user_id:
                    continue
                meta = self._meta.get(mid, {})
                results.append(self._sync_meta_to_memory(mem, meta))
                if len(results) >= limit:
                    break
            return results

    # ------------------------------------------------------------------
    # 分层迁移钩子
    # ------------------------------------------------------------------

    def pop_for_eviction(self, limit: int = 1) -> list[Memory]:
        """按 LRU 顺序取出最久未使用的记忆用于淘汰到 DeepStore。"""
        evicted: list[Memory] = []
        with self._lock:
            while len(evicted) < limit and self._cache:
                oldest_id, oldest_mem = self._cache.popitem(last=False)
                meta = self._meta.pop(oldest_id, {})
                evicted.append(self._sync_meta_to_memory(oldest_mem, meta))
        return evicted

    def ingest_from_upper(self, memories: list[Memory]) -> None:
        """HotStore 作为最上层，不会接收来自上层的记忆。"""
        _logger.debug("HotStore.ingest_from_upper called with %d items (no-op)", len(memories))

    def peek_access_metadata(self, memory_id: str) -> dict[str, Any] | None:
        with self._lock:
            meta = self._meta.get(memory_id)
            if meta:
                return dict(meta)
            mem = self._cache.get(memory_id)
            if mem:
                return {
                    "access_count": mem.access_count,
                    "last_accessed": mem.last_accessed,
                }
            return None

    # ------------------------------------------------------------------
    # 额外管理接口
    # ------------------------------------------------------------------

    def warm(self, memories: list[Memory]) -> None:
        """批量预热：将记忆加载到热层（用于启动时从 DeepStore 恢复热点数据）。"""
        with self._lock:
            for mem in memories:
                if mem.id in self._cache:
                    continue
                self._cache[mem.id] = mem
                self._meta[mem.id] = {
                    "access_count": mem.access_count,
                    "last_accessed": mem.last_accessed,
                }
                self._touch(mem.id)
            evicted = self._evict_if_needed()
        self._flush_evicted(evicted)

    def clear(self) -> int:
        """清空热层缓存，返回清空的条数。"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._meta.clear()
            self._total_hits = 0
            self._total_misses = 0
            return count
