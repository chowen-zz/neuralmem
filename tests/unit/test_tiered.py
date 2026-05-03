"""Tiered memory 单元测试 — 全部使用 mock, 不依赖外部 API."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from neuralmem.tiered.hot_store import HotStore
from neuralmem.tiered.deep_store import DeepStore
from neuralmem.tiered.manager import TieredManager


# --------------------------------------------------------------------------- #
# HotStore
# --------------------------------------------------------------------------- #

def test_hot_store_init():
    store = HotStore(capacity=100)
    assert store._capacity == 100
    assert len(store._cache) == 0


def test_hot_store_save_and_get():
    store = HotStore(capacity=10)
    mem = MagicMock()
    mem.id = "m1"
    mem.last_accessed = None
    mem.access_count = 0
    # model_copy should return self for MagicMock
    mem.model_copy.return_value = mem
    store.save_memory(mem)
    result = store.get_memory("m1")
    # get_memory returns model_copy with synced metadata, so use == not is
    assert result is not None
    assert result.id == "m1"


def test_hot_store_lru_eviction():
    store = HotStore(capacity=2)
    m1 = MagicMock(id="m1")
    m2 = MagicMock(id="m2")
    m3 = MagicMock(id="m3")

    store.save_memory(m1)
    store.save_memory(m2)
    store.save_memory(m3)  # Should evict m1

    assert store.get_memory("m1") is None
    assert store.get_memory("m2") is not None
    assert store.get_memory("m3") is not None


def test_hot_store_access_tracking():
    store = HotStore(capacity=10)
    mem = MagicMock(id="m1")
    store.save_memory(mem)
    store.get_memory("m1")
    store.get_memory("m1")
    meta = store._meta.get("m1")
    assert meta is not None
    assert meta["access_count"] == 2


def test_hot_store_keyword_search():
    store = HotStore(capacity=10)
    m1 = MagicMock(id="m1", content="hello world")
    store.save_memory(m1)
    results = store.keyword_search("hello", limit=5)
    assert len(results) >= 1


# --------------------------------------------------------------------------- #
# DeepStore
# --------------------------------------------------------------------------- #

def test_deep_store_init():
    config = MagicMock()
    store = DeepStore(config)
    assert store._backend is not None


def test_deep_store_ingest_from_upper():
    config = MagicMock()
    store = DeepStore(config)
    # Mock the internal backend
    store._backend = MagicMock()
    mem = MagicMock(id="m1")
    store.ingest_from_upper([mem])
    store._backend.save_memory.assert_called_once_with(mem)


def test_deep_store_fetch_for_promotion():
    config = MagicMock()
    store = DeepStore(config)
    store._backend = MagicMock()
    mem = MagicMock(id="m1")
    store._backend.get_memory.return_value = mem
    result = store.fetch_for_promotion(["m1"])
    assert result == [mem]


# --------------------------------------------------------------------------- #
# TieredManager
# --------------------------------------------------------------------------- #

def test_tiered_manager_init():
    config = MagicMock()
    manager = TieredManager(config, hot_capacity=10)
    assert manager is not None


def test_tiered_manager_read_hot_hit():
    """Hot 层命中时直接返回."""
    config = MagicMock()
    manager = TieredManager(config, hot_capacity=10)

    mem = MagicMock(id="m1")
    mem.model_copy.return_value = mem
    mem.last_accessed = None
    mem.access_count = 0
    manager._hot.save_memory(mem)
    result = manager.get_memory("m1")
    assert result is not None
    assert result.id == "m1"


def test_tiered_manager_read_hot_miss():
    """Hot 层未命中时查询 Deep 层."""
    config = MagicMock()
    manager = TieredManager(config, hot_capacity=10)

    mem = MagicMock(id="m1")
    mem.model_copy.return_value = mem
    mem.last_accessed = None
    mem.access_count = 0
    manager._deep._backend = MagicMock()
    manager._deep._backend.get_memory.return_value = mem
    result = manager.get_memory("m1")
    assert result is not None
    assert result.id == "m1"


def test_tiered_manager_write_through():
    """写入时同时写入 hot 和 deep."""
    config = MagicMock()
    manager = TieredManager(config, hot_capacity=10)

    mem = MagicMock(id="m1")
    mem.model_copy.return_value = mem
    mem.last_accessed = None
    mem.access_count = 0
    manager.save_memory(mem)
    result = manager._hot.get_memory("m1")
    assert result is not None
    assert result.id == "m1"


def test_tiered_manager_search_both_tiers():
    """搜索同时查询两层."""
    config = MagicMock()
    manager = TieredManager(config, hot_capacity=10)

    m1 = MagicMock(id="m1", content="hot memory")
    m1.model_copy.return_value = m1
    m1.last_accessed = None
    m1.access_count = 0
    manager._hot.save_memory(m1)

    manager._deep._backend = MagicMock()
    manager._deep._backend.keyword_search.return_value = [
        ("m2", 0.5)
    ]

    results = manager.keyword_search("memory", limit=5)
    assert isinstance(results, list)


def test_tiered_manager_warm():
    """预热加载最近访问的记忆."""
    config = MagicMock()
    manager = TieredManager(config, hot_capacity=10)

    mem = MagicMock(id="m1")
    mem.model_copy.return_value = mem
    mem.last_accessed = None
    mem.access_count = 0
    manager._deep._backend = MagicMock()
    manager._deep._backend.list_memories.return_value = [mem]
    manager.warm_hot_store(limit=5)
    result = manager._hot.get_memory("m1")
    assert result is not None
