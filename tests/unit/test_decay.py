"""Ebbinghaus DecayManager 测试"""
from __future__ import annotations
import math
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
from neuralmem.lifecycle.decay import DecayManager, _compute_r
from neuralmem.core.types import Memory


def _make_storage(memories: list[Memory]):
    storage = MagicMock()
    storage.list_memories.return_value = memories
    storage.update_memory = MagicMock()
    storage.delete_memories = MagicMock(return_value=0)
    return storage


def _memory(importance: float, days_ago: float, access_count: int = 0) -> Memory:
    last_accessed = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return Memory(
        content="test memory",
        importance=importance,
        last_accessed=last_accessed,
        access_count=access_count,
    )


def test_compute_r_zero_time():
    r = _compute_r(importance=0.5, access_count=0, days_ago=0.0)
    assert r == pytest.approx(1.0, abs=0.01)


def test_compute_r_one_day():
    r = _compute_r(importance=0.5, access_count=0, days_ago=1.0)
    # S = 0.5 * 10 * 1.5^0 = 5; R = e^(-1/5)
    assert r == pytest.approx(math.exp(-1 / 5), rel=0.01)


def test_compute_r_high_access_decays_slower():
    r_low = _compute_r(importance=0.5, access_count=0, days_ago=10.0)
    r_high = _compute_r(importance=0.5, access_count=5, days_ago=10.0)
    assert r_high > r_low


def test_compute_r_approaches_zero_long_time():
    r = _compute_r(importance=0.1, access_count=0, days_ago=365.0)
    assert r < 0.01


def test_apply_decay_updates_importance():
    mem = _memory(importance=0.8, days_ago=30.0, access_count=0)
    storage = _make_storage([mem])
    dm = DecayManager(storage)
    count = dm.apply_decay()
    assert count == 1
    storage.update_memory.assert_called_once()
    call_kwargs = storage.update_memory.call_args[1]
    assert call_kwargs["importance"] < 0.8


def test_apply_decay_returns_count():
    memories = [
        _memory(importance=0.5, days_ago=10.0),
        _memory(importance=0.5, days_ago=20.0),
    ]
    storage = _make_storage(memories)
    dm = DecayManager(storage)
    count = dm.apply_decay()
    assert count == 2


def test_apply_decay_recently_accessed_minimal_change():
    mem = _memory(importance=0.8, days_ago=0.01, access_count=0)
    storage = _make_storage([mem])
    dm = DecayManager(storage)
    dm.apply_decay()
    call_kwargs = storage.update_memory.call_args[1]
    assert call_kwargs["importance"] > 0.78


def test_remove_forgotten_calls_delete_with_threshold():
    storage = MagicMock()
    storage.delete_memories.return_value = 3
    dm = DecayManager(storage)
    count = dm.remove_forgotten()
    assert count == 3
    call_kwargs = storage.delete_memories.call_args[1]
    assert call_kwargs.get("max_importance") == pytest.approx(0.05)


def test_remove_forgotten_with_user_id():
    storage = MagicMock()
    storage.delete_memories.return_value = 0
    dm = DecayManager(storage)
    dm.remove_forgotten(user_id="user-1")
    call_kwargs = storage.delete_memories.call_args[1]
    assert call_kwargs.get("user_id") == "user-1"


def test_apply_decay_with_user_id():
    storage = MagicMock()
    storage.list_memories.return_value = []
    dm = DecayManager(storage)
    count = dm.apply_decay(user_id="user-1")
    assert count == 0
    storage.list_memories.assert_called_with(user_id="user-1")
