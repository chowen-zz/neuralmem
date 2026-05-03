"""IntelligentDecay 单元测试 — 全部使用 mock, 不依赖外部 API."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from neuralmem.lifecycle.intelligent_decay import (
    IntelligentDecay,
    compute_adaptive_decay,
)


# --------------------------------------------------------------------------- #
# compute_adaptive_decay
# --------------------------------------------------------------------------- #

def test_compute_adaptive_decay_new_memory():
    """新记忆的衰减因子应接近 1.0."""
    mem = MagicMock()
    mem.created_at = datetime.now(timezone.utc)
    result = compute_adaptive_decay(mem, [], importance_score=0.5)
    assert result > 0.9


def test_compute_adaptive_decay_old_memory():
    """旧记忆的衰减因子应较低."""
    mem = MagicMock()
    mem.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    result = compute_adaptive_decay(mem, [], importance_score=0.5)
    assert result < 0.5


def test_compute_adaptive_decay_with_accesses():
    """近期访问应提升保留率."""
    mem = MagicMock()
    mem.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    accesses = [datetime.now(timezone.utc)] * 5
    result = compute_adaptive_decay(
        mem, accesses, importance_score=0.5, access_boost=0.1
    )
    # With 5 accesses, reinforcement = 0.5 (capped), so result should be higher
    assert result > 0.3  # 有访问时比无访问高


def test_compute_adaptive_decay_high_importance():
    """高重要性记忆衰减更慢."""
    mem = MagicMock()
    mem.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    low = compute_adaptive_decay(mem, [], importance_score=0.1)
    high = compute_adaptive_decay(mem, [], importance_score=0.9)
    assert high > low


def test_compute_adaptive_decay_clamped():
    """结果应被限制在 [0, 1]."""
    mem = MagicMock()
    mem.created_at = datetime.now(timezone.utc)
    result = compute_adaptive_decay(
        mem, [], importance_score=1.0, access_boost=1.0
    )
    assert 0.0 <= result <= 1.0


# --------------------------------------------------------------------------- #
# IntelligentDecay
# --------------------------------------------------------------------------- #

def test_intelligent_decay_init():
    decay = IntelligentDecay(half_life_days=60, forget_threshold=0.2)
    assert decay.half_life_days == 60
    assert decay.forget_threshold == 0.2


def test_evaluate():
    decay = IntelligentDecay()
    mem = MagicMock()
    mem.created_at = datetime.now(timezone.utc)
    mem.id = "test-id"
    result = decay.evaluate(mem, [], 0.5)
    assert 0.0 <= result <= 1.0


def test_select_for_forgetting():
    """选择低于阈值的记忆."""
    decay = IntelligentDecay(forget_threshold=0.5)

    old_mem = MagicMock()
    old_mem.id = "old"
    old_mem.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)

    new_mem = MagicMock()
    new_mem.id = "new"
    new_mem.created_at = datetime.now(timezone.utc)

    access_logs = {"old": [], "new": []}
    result = decay.select_for_forgetting(
        [old_mem, new_mem], access_logs
    )
    assert old_mem in result
    assert new_mem not in result


def test_batch_decay_importance():
    """批量衰减重要性."""
    decay = IntelligentDecay()

    mem = MagicMock()
    mem.id = "m1"
    mem.created_at = datetime.now(timezone.utc)
    mem.importance = 0.8

    result = decay.batch_decay_importance([mem], {})
    assert result["m1"] <= 0.8  # 衰减后应 <= 原值
