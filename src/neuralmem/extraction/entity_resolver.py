"""EntityResolver — 两阶段实体消歧（规则过滤 + Embedding 精排）"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from neuralmem.core.protocols import EmbedderProtocol
from neuralmem.core.types import Entity

_logger = logging.getLogger(__name__)
_EMBEDDING_THRESHOLD = 0.85
_MAX_EDIT_DISTANCE = 2


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein 编辑距离（动态规划）"""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _rules_candidate(new_name: str, existing_name: str) -> bool:
    """Stage 1: 规则快速过滤"""
    n1, n2 = new_name.lower(), existing_name.lower()
    return (
        n1 == n2
        or n1 in n2
        or n2 in n1
        or _edit_distance(n1, n2) <= _MAX_EDIT_DISTANCE
    )


def _merge(existing: Entity, new_name: str) -> Entity:
    """将 new_name 作为别名追加到 existing，返回更新后的实体"""
    current_aliases = set(existing.aliases)
    if new_name != existing.name and new_name not in current_aliases:
        current_aliases.add(new_name)
    return Entity(
        id=existing.id,
        name=existing.name,
        entity_type=existing.entity_type,
        aliases=tuple(current_aliases),
        attributes=dict(existing.attributes),
        first_seen=existing.first_seen,
        last_seen=datetime.now(timezone.utc),
    )


class EntityResolver:
    """
    两阶段实体消歧：
    1. 规则快速过滤（完全匹配 / 子串 / 编辑距离 ≤ 2）
    2. Embedding 余弦相似度精排（≥ 0.85 → 确认合并）
    """

    def __init__(self, embedder: EmbedderProtocol) -> None:
        self._embedder = embedder

    def resolve(
        self,
        new_entities: list[Entity],
        existing_entities: list[Entity],
    ) -> list[Entity]:
        """
        对每个 new_entity，尝试与 existing_entities 匹配。
        匹配 → 返回合并后的已有实体（含新别名）。
        不匹配 → 返回原新实体。
        """
        result: list[Entity] = []
        for new_e in new_entities:
            matched = self._find_match(new_e, existing_entities)
            if matched is not None:
                result.append(_merge(matched, new_e.name))
                _logger.debug(
                    "EntityResolver: merged '%s' → '%s'", new_e.name, matched.name
                )
            else:
                result.append(new_e)
        return result

    def _find_match(
        self, new_e: Entity, existing: list[Entity]
    ) -> Entity | None:
        # Stage 1: 规则候选
        candidates = [e for e in existing if _rules_candidate(new_e.name, e.name)]
        if not candidates:
            return None

        # 完全匹配（大小写不敏感）直接返回，跳过 embedding
        for c in candidates:
            if new_e.name.lower() == c.name.lower():
                return c

        # Stage 2: Embedding 精排
        new_vec = self._embedder.encode_one(new_e.name)
        for c in candidates:
            existing_vec = self._embedder.encode_one(c.name)
            sim = _cosine(new_vec, existing_vec)
            if sim >= _EMBEDDING_THRESHOLD:
                return c

        return None
