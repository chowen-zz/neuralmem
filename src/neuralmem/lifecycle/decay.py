"""DecayManager — Ebbinghaus 遗忘曲线实现"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from neuralmem.core.protocols import StorageProtocol

_logger = logging.getLogger(__name__)

_FORGOTTEN_THRESHOLD = 0.05


def _compute_r(importance: float, access_count: int, days_ago: float) -> float:
    """
    Ebbinghaus 保留率: R = e^(-t / S)
    S = importance × 10 × 1.5^access_count
    """
    S = importance * 10 * (1.5 ** access_count)
    if S <= 0:
        return 0.0
    return math.exp(-days_ago / S)


def _days_since(dt: datetime) -> float:
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (now - dt).total_seconds() / 86400)


class DecayManager:
    """记忆衰减管理器 — Ebbinghaus 遗忘曲线"""

    def __init__(self, storage: StorageProtocol) -> None:
        self._storage = storage

    def apply_decay(self, user_id: str | None = None) -> int:
        """
        对所有记忆应用遗忘曲线，更新 importance 字段。
        返回更新的记忆数量。
        """
        memories = self._storage.list_memories(user_id=user_id)
        updated = 0
        for mem in memories:
            t = _days_since(mem.last_accessed)
            r = _compute_r(mem.importance, mem.access_count, t)
            new_importance = mem.importance * r
            if abs(new_importance - mem.importance) > 1e-6:
                self._storage.update_memory(mem.id, importance=new_importance)
                updated += 1
        _logger.debug("apply_decay: updated %d memories for user=%s", updated, user_id)
        return updated

    def remove_forgotten(self, user_id: str | None = None) -> int:
        """
        物理删除 importance < 0.05 的记忆（已被遗忘）。
        返回删除数量。
        """
        count = self._storage.delete_memories(
            user_id=user_id,
            max_importance=_FORGOTTEN_THRESHOLD,
        )
        _logger.debug("remove_forgotten: deleted %d memories for user=%s", count, user_id)
        return count
