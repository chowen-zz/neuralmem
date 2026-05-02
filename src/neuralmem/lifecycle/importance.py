"""ImportanceScorer — 多维度重要性评分"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from neuralmem.core.types import Memory, MemoryType

_logger = logging.getLogger(__name__)


class _ImportanceBase(ABC):
    @abstractmethod
    def score(self, memory: Memory) -> float:
        raise NotImplementedError  # pragma: no cover


def _days_since(dt: datetime) -> float:
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (now - dt).total_seconds() / 86400)


def _recency_factor(last_accessed: datetime) -> float:
    """近因效应：最近访问越频繁，重要性越高。指数衰减。"""
    days = _days_since(last_accessed)
    return math.exp(-days / 30.0)  # 30 天半衰期


def _access_factor(access_count: int) -> float:
    """访问频次因子：对数增长，高频访问的记忆更稳定。"""
    if access_count <= 0:
        return 0.0
    return min(1.0, math.log1p(access_count) / 5.0)  # log(1+x)/5，归一化到 0~1


def _entity_factor(entity_count: int) -> float:
    """实体关联度：关联实体越多，说明该记忆在网络中更重要。"""
    if entity_count <= 0:
        return 0.0
    return min(1.0, entity_count / 5.0)  # 5 个实体满分


def _type_factor(memory_type: MemoryType) -> float:
    """记忆类型权重：语义和程序性记忆比事件记忆更持久。"""
    weights = {
        MemoryType.SEMANTIC: 1.0,
        MemoryType.PROCEDURAL: 0.95,
        MemoryType.EPISODIC: 0.7,
        MemoryType.WORKING: 0.3,
    }
    return weights.get(memory_type, 0.7)


class ImportanceScorer(_ImportanceBase):
    """
    多维度重要性评分器。

    综合考虑：
    - 基础重要性（来自记忆创建时的 importance 值）
    - 访问频次（高频访问 → 更重要）
    - 实体关联度（连接越多 → 越重要）
    - 近因效应（最近访问 → 更重要）
    - 记忆类型权重（语义 > 程序性 > 事件 > 工作记忆）
    """

    # 各因子权重（可调）
    _W_BASE = 0.35
    _W_ACCESS = 0.20
    _W_ENTITY = 0.15
    _W_RECENCY = 0.20
    _W_TYPE = 0.10

    def score(self, memory: Memory) -> float:
        base = memory.importance
        access = _access_factor(memory.access_count)
        entity = _entity_factor(len(memory.entity_ids))
        recency = _recency_factor(memory.last_accessed)
        mtype = _type_factor(memory.memory_type)

        raw = (
            self._W_BASE * base
            + self._W_ACCESS * access
            + self._W_ENTITY * entity
            + self._W_RECENCY * recency
            + self._W_TYPE * mtype
        )

        # 轻微非线性拉伸：高基础 importance 的记忆更容易保持高分
        result = raw * (0.8 + 0.2 * base)
        return max(0.0, min(1.0, result))
