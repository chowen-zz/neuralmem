"""RRF (Reciprocal Rank Fusion) 融合器 — 独立可测试模块"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RankedItem:
    id: str
    score: float
    method: str


class RRFMerger:
    """
    倒数排名融合 (Reciprocal Rank Fusion)。
    RRF(d) = Σ 1/(k + rank_i(d))，k=60 为经典默认值。
    """

    def __init__(self, k: int = 60):
        self.k = k

    def merge(
        self,
        ranked_lists: dict[str, list[RankedItem]],
    ) -> list[tuple[str, float]]:
        """
        Args:
            ranked_lists: {strategy_name: [RankedItem sorted by score desc]}
        Returns:
            List of (id, normalized_rrf_score) sorted by score desc
        """
        rrf_scores: dict[str, float] = {}

        for _method, items in ranked_lists.items():
            # 按分数降序排列（防御性重排）
            sorted_items = sorted(items, key=lambda x: x.score, reverse=True)
            for rank, item in enumerate(sorted_items, start=1):
                rrf_scores[item.id] = rrf_scores.get(item.id, 0.0) + 1.0 / (self.k + rank)

        if not rrf_scores:
            return []

        # 归一化到 0-1
        max_score = max(rrf_scores.values())
        if max_score > 0:
            rrf_scores = {k: v / max_score for k, v in rrf_scores.items()}

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
