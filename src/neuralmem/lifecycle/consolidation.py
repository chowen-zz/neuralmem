"""MemoryConsolidator — 相似记忆聚合与去重"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from neuralmem.core.types import Memory

if TYPE_CHECKING:
    from neuralmem.core.protocols import EmbedderProtocol, StorageProtocol

_logger = logging.getLogger(__name__)

# 相似度阈值：高于此值的两条记忆视为"可合并"
_MERGE_THRESHOLD = 0.85


class _ConsolidatorBase(ABC):
    @abstractmethod
    def merge_similar(self, user_id: str | None = None) -> int:
        raise NotImplementedError  # pragma: no cover


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度（纯 Python，无 numpy 依赖）"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _merge_memories(keeper: Memory, loser: Memory) -> dict:
    """生成合并后的记忆更新字段。

    策略：
    - 保留 keeper 的内容（更短/更精炼的）
    - 合并标签和实体 ID（去重）
    - 取更高的 importance
    - 更新时间戳为最新的
    - 累加 access_count
    """
    # 选择内容更短（更精炼）的作为保留内容
    content = keeper.content if len(keeper.content) <= len(loser.content) else loser.content

    # 合并去重的 tags
    merged_tags = tuple(dict.fromkeys(keeper.tags + loser.tags))

    # 合并去重的 entity_ids
    merged_entities = tuple(dict.fromkeys(keeper.entity_ids + loser.entity_ids))

    # 取更高的 importance
    importance = max(keeper.importance, loser.importance)

    # 累加访问次数
    access_count = keeper.access_count + loser.access_count

    # 取最新的时间戳
    last_accessed = max(keeper.last_accessed, loser.last_accessed)
    updated_at = datetime.now(timezone.utc)

    return {
        "content": content,
        "tags": merged_tags,
        "entity_ids": merged_entities,
        "importance": importance,
        "access_count": access_count,
        "last_accessed": last_accessed,
        "updated_at": updated_at,
    }


class MemoryConsolidator(_ConsolidatorBase):
    """
    记忆合并去重。

    通过 embedding 相似度检测相似记忆，合并为一条最优记忆。
    需要注入 storage 和 embedder 才能工作；如果未注入则退化为 stub。
    """

    def __init__(
        self,
        storage: StorageProtocol | None = None,
        embedder: EmbedderProtocol | None = None,
        merge_threshold: float = _MERGE_THRESHOLD,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._merge_threshold = merge_threshold

    def merge_similar(self, user_id: str | None = None) -> int:
        """
        检测并合并相似记忆。

        算法：
        1. 加载用户所有记忆
        2. 计算每条记忆的 embedding（使用已有的或重新生成）
        3. 贪心匹配：按创建时间排序，逐对比较
        4. 相似度 > 阈值 → 合并到更早的记忆，删除较晚的

        Returns:
            合并（删除）的记忆数量
        """
        if self._storage is None or self._embedder is None:
            _logger.debug("MemoryConsolidator.merge_similar called without storage/embedder (stub)")
            return 0

        memories = self._storage.list_memories(user_id=user_id)
        if len(memories) < 2:
            return 0

        # 按创建时间排序（保留最早的作为 keeper）
        memories.sort(key=lambda m: m.created_at)

        # 计算所有 embeddings（内存中）
        embeddings: dict[str, list[float]] = {}
        for mem in memories:
            if mem.embedding:
                embeddings[mem.id] = mem.embedding
            else:
                embeddings[mem.id] = self._embedder.encode_one(mem.content)

        # 贪心合并：用并查集思想，标记被合并的记忆
        to_delete: set[str] = set()
        merged_count = 0

        for i, mem_i in enumerate(memories):
            if mem_i.id in to_delete:
                continue
            for j in range(i + 1, len(memories)):
                mem_j = memories[j]
                if mem_j.id in to_delete:
                    continue

                sim = _cosine_similarity(embeddings[mem_i.id], embeddings[mem_j.id])
                if sim >= self._merge_threshold:
                    # 合并到 mem_i（keeper），标记 mem_j 待删除
                    updates = _merge_memories(mem_i, mem_j)
                    self._storage.update_memory(mem_i.id, **updates)
                    to_delete.add(mem_j.id)
                    merged_count += 1

                    _logger.debug(
                        "Merged memory '%s' into '%s' (sim=%.3f)",
                        mem_j.content[:30], mem_i.content[:30], sim,
                    )
                    # 更新 mem_i 的内存快照，继续比较
                    mem_i = Memory(
                        id=mem_i.id,
                        **{k: v for k, v in mem_i.__dict__.items()
                           if k not in ("id",) and k not in updates},
                        **updates,
                    )

        # 批量删除被合并的记忆
        if to_delete:
            for mid in to_delete:
                self._storage.delete_memories(memory_id=mid)

        _logger.debug(
            "merge_similar: merged %d memories for user=%s",
            merged_count, user_id,
        )
        return merged_count
