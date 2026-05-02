"""MemoryConsolidator — 相似记忆聚合与去重"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

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


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix using numpy.

    Args:
        embeddings: (N, D) array of L2-normalized (or raw) vectors.

    Returns:
        (N, N) similarity matrix with values in [-1, 1].
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero — rows with zero norm stay zero
    norms = np.where(norms == 0, 1.0, norms)
    normalized = embeddings / norms
    return normalized @ normalized.T


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


def _build_groups_from_sim_matrix(
    sim_matrix: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Build merge groups from a similarity matrix using greedy clustering.

    Memories are grouped so that every member is similar to at least one
    other member above *threshold*.  The algorithm is:

    1. Iterate memories in order.
    2. For each unvisited memory *i*, start a new group.
    3. Greedily add every later unvisited memory *j* whose similarity
       to *any* existing member of the group meets the threshold.

    Returns:
        List of groups, each group is a list of indices into the original
        memory list.  Singleton groups are excluded (nothing to merge).
    """
    n = sim_matrix.shape[0]
    visited: set[int] = set()
    groups: list[list[int]] = []

    for i in range(n):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, n):
            if j in visited:
                continue
            # Check similarity against ALL current group members —
            # only add if it's similar to at least one.
            if any(sim_matrix[j, k] >= threshold for k in group):
                group.append(j)
                visited.add(j)
        if len(group) > 1:
            groups.append(group)

    return groups


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

    # ------------------------------------------------------------------
    # merge_similar — full implementation
    # ------------------------------------------------------------------

    def merge_similar(
        self,
        user_id: str | None = None,
        similarity_threshold: float | None = None,
        limit: int = 100,
    ) -> int:
        """
        Detect and merge similar memories.

        Algorithm:
        1. Load active memories for *user_id* (up to *limit*).
        2. Compute / reuse embeddings.
        3. Compute pairwise cosine-similarity matrix (numpy).
        4. Greedily cluster memories whose similarity >= threshold.
        5. For each cluster:
           a. Pick the canonical (earliest created, highest importance).
           b. Append unique content from others to canonical.
           c. Set canonical importance = max(group).
           d. Mark others as superseded (is_active=False,
              superseded_by=canonical.id).
        6. Return count of memories marked as superseded.

        Args:
            user_id: Scope merge to a single user, or None for all.
            similarity_threshold: Override the instance default threshold.
            limit: Maximum number of memories to load and process.

        Returns:
            Number of memories that were merged (superseded).
        """
        if self._storage is None or self._embedder is None:
            _logger.debug(
                "MemoryConsolidator.merge_similar called without "
                "storage/embedder (stub)"
            )
            return 0

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._merge_threshold
        )

        memories = self._storage.list_memories(user_id=user_id, limit=limit)
        if len(memories) < 2:
            return 0

        # Sort by creation time — earlier memories are preferred as canonical
        memories.sort(key=lambda m: m.created_at)

        # ---- Step 2: compute embeddings ----
        embeddings_list: list[list[float]] = []
        for mem in memories:
            if mem.embedding:
                embeddings_list.append(mem.embedding)
            else:
                embeddings_list.append(self._embedder.encode_one(mem.content))

        emb_array = np.array(embeddings_list, dtype=np.float32)

        # ---- Step 3: pairwise cosine similarity ----
        sim_matrix = _cosine_similarity_matrix(emb_array)

        # ---- Step 4: greedy clustering ----
        groups = _build_groups_from_sim_matrix(sim_matrix, threshold)

        if not groups:
            return 0

        # ---- Step 5: merge each group ----
        merged_count = 0

        for group_indices in groups:
            group_mems = [memories[i] for i in group_indices]

            # Pick canonical: earliest created_at, break ties by importance
            canonical = min(
                group_mems,
                key=lambda m: (m.created_at, -m.importance),
            )
            others = [m for m in group_mems if m.id != canonical.id]

            # Build merged content: canonical + unique tails from others
            combined_parts = [canonical.content]
            for other in others:
                stripped = other.content.strip()
                if stripped and stripped not in canonical.content:
                    combined_parts.append(stripped)
            merged_content = "; ".join(combined_parts)

            # Merge tags / entity_ids (deduplicated)
            all_tags = list(canonical.tags)
            all_entities = list(canonical.entity_ids)
            for other in others:
                for t in other.tags:
                    if t not in all_tags:
                        all_tags.append(t)
                for e in other.entity_ids:
                    if e not in all_entities:
                        all_entities.append(e)

            # Importance = max of the group
            max_importance = max(m.importance for m in group_mems)

            # Access count = sum of group
            total_access = sum(m.access_count for m in group_mems)

            # Timestamps
            last_accessed = max(m.last_accessed for m in group_mems)
            updated_at = datetime.now(timezone.utc)

            # Update canonical memory
            update_fields = {
                "content": merged_content,
                "tags": tuple(all_tags),
                "entity_ids": tuple(all_entities),
                "importance": max_importance,
                "access_count": total_access,
                "last_accessed": last_accessed,
                "updated_at": updated_at,
            }
            self._storage.update_memory(canonical.id, **update_fields)

            # Mark others as superseded (NOT deleted)
            superseded_ids = []
            for other in others:
                self._storage.update_memory(
                    other.id,
                    is_active=False,
                    superseded_by=canonical.id,
                    updated_at=updated_at,
                )
                superseded_ids.append(other.id)
                merged_count += 1

            _logger.debug(
                "Merged %d memories into '%s' (canonical=%s, "
                "superseded=%s)",
                len(others),
                canonical.content[:40],
                canonical.id[:8],
                [sid[:8] for sid in superseded_ids],
            )

        _logger.debug(
            "merge_similar: merged %d memories for user=%s",
            merged_count,
            user_id,
        )
        return merged_count
