"""Tests for LRU cache and CacheManager."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from neuralmem.cache.cache_manager import CacheManager, _cache_key
from neuralmem.cache.lru_cache import CacheStats, LRUCache

# ---------------------------------------------------------------------------
# LRUCache tests
# ---------------------------------------------------------------------------


class TestLRUCache:
    def test_put_and_get(self):
        cache = LRUCache(max_size=3, ttl_seconds=60)
        cache.put("a", 1)
        assert cache.get("a") == 1

    def test_get_missing_key_returns_none(self):
        cache = LRUCache(max_size=3, ttl_seconds=60)
        assert cache.get("missing") is None

    def test_lru_eviction_on_overflow(self):
        cache = LRUCache(max_size=2, ttl_seconds=60)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_lru_order_updates_on_get(self):
        cache = LRUCache(max_size=2, ttl_seconds=60)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # touch "a" → "b" becomes LRU
        cache.put("c", 3)  # evicts "b"
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_ttl_expiration(self):
        cache = LRUCache(max_size=10, ttl_seconds=0.05)
        cache.put("k", "v")
        assert cache.get("k") == "v"
        time.sleep(0.06)
        assert cache.get("k") is None

    def test_put_overwrites_existing(self):
        cache = LRUCache(max_size=3, ttl_seconds=60)
        cache.put("a", 1)
        cache.put("a", 99)
        assert cache.get("a") == 99
        assert len(cache) == 1

    def test_invalidate_existing(self):
        cache = LRUCache(max_size=3, ttl_seconds=60)
        cache.put("a", 1)
        assert cache.invalidate("a") is True
        assert cache.get("a") is None

    def test_invalidate_missing(self):
        cache = LRUCache(max_size=3, ttl_seconds=60)
        assert cache.invalidate("nope") is False

    def test_clear_returns_count(self):
        cache = LRUCache(max_size=3, ttl_seconds=60)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.clear() == 2
        assert len(cache) == 0

    def test_len(self):
        cache = LRUCache(max_size=5, ttl_seconds=60)
        assert len(cache) == 0
        cache.put("a", 1)
        assert len(cache) == 1

    def test_contains(self):
        cache = LRUCache(max_size=5, ttl_seconds=60)
        cache.put("a", 1)
        assert "a" in cache
        assert "b" not in cache

    def test_contains_expired(self):
        cache = LRUCache(max_size=5, ttl_seconds=0.05)
        cache.put("a", 1)
        time.sleep(0.06)
        assert "a" not in cache

    def test_stats_tracking(self):
        cache = LRUCache(max_size=2, ttl_seconds=60)
        cache.put("a", 1)
        cache.get("a")        # hit
        cache.get("b")        # miss
        cache.put("b", 2)
        cache.put("c", 3)     # evicts "a"
        cache.invalidate("b")

        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.evictions == 1
        assert stats.invalidations == 1

    def test_stats_hit_rate(self):
        stats = CacheStats(hits=7, misses=3)
        assert stats.total == 10
        assert abs(stats.hit_rate - 0.7) < 1e-9

    def test_stats_hit_rate_zero_total(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_max_size_property(self):
        cache = LRUCache(max_size=42, ttl_seconds=10)
        assert cache.max_size == 42
        assert cache.ttl_seconds == 10.0

    def test_thread_safety_concurrent_puts(self):
        cache = LRUCache(max_size=100, ttl_seconds=60)
        barrier = threading.Barrier(10)

        def writer(start: int):
            barrier.wait()
            for i in range(50):
                cache.put(f"t{start}_{i}", i)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Should not crash; length ≤ max_size
        assert len(cache) <= 100


# ---------------------------------------------------------------------------
# CacheManager tests
# ---------------------------------------------------------------------------


class TestCacheManager:
    def _make_engine(self, results=None):
        engine = MagicMock()
        engine.search = MagicMock(return_value=results or ["r1", "r2"])
        return engine

    def _make_query(self, query="q1", user_id="u1", memory_types=None):
        q = MagicMock()
        q.query = query
        q.user_id = user_id
        q.memory_types = memory_types
        return q

    def test_search_cache_hit(self):
        engine = self._make_engine()
        mgr = CacheManager(engine, max_size=10, ttl_seconds=60)
        q = self._make_query()

        r1 = mgr.search(q)
        r2 = mgr.search(q)

        assert r1 == ["r1", "r2"]
        assert r2 == ["r1", "r2"]
        assert engine.search.call_count == 1  # only called once

    def test_search_cache_miss_different_query(self):
        engine = self._make_engine()
        mgr = CacheManager(engine, max_size=10, ttl_seconds=60)

        mgr.search(self._make_query(query="q1"))
        mgr.search(self._make_query(query="q2"))

        assert engine.search.call_count == 2

    def test_search_ttl_expires(self):
        engine = self._make_engine()
        mgr = CacheManager(engine, max_size=10, ttl_seconds=0.05)
        q = self._make_query()

        mgr.search(q)
        time.sleep(0.06)
        mgr.search(q)

        assert engine.search.call_count == 2

    def test_invalidate_search_forces_miss(self):
        engine = self._make_engine()
        mgr = CacheManager(engine, max_size=10, ttl_seconds=60)
        q = self._make_query()

        mgr.search(q)
        key = _cache_key(q.query, q.user_id, q.memory_types)
        mgr.invalidate_search(key)
        mgr.search(q)

        assert engine.search.call_count == 2

    def test_embedding_cache_put_get(self):
        engine = self._make_engine()
        mgr = CacheManager(engine)
        mgr.put_embedding("hello", [0.1, 0.2])
        assert mgr.get_embedding("hello") == [0.1, 0.2]

    def test_embedding_cache_miss(self):
        engine = self._make_engine()
        mgr = CacheManager(engine)
        assert mgr.get_embedding("nope") is None

    def test_invalidate_embedding(self):
        engine = self._make_engine()
        mgr = CacheManager(engine)
        mgr.put_embedding("x", [1.0])
        mgr.invalidate_embedding("x")
        assert mgr.get_embedding("x") is None

    def test_clear_all(self):
        engine = self._make_engine()
        mgr = CacheManager(engine, max_size=10, ttl_seconds=60)
        q = self._make_query()
        mgr.search(q)
        mgr.put_embedding("x", [1.0])
        mgr.clear_all()
        # After clear, search should call engine again
        mgr.search(q)
        assert engine.search.call_count == 2
        assert mgr.get_embedding("x") is None

    def test_search_stats(self):
        engine = self._make_engine()
        mgr = CacheManager(engine, max_size=10, ttl_seconds=60)
        q = self._make_query()
        mgr.search(q)
        mgr.search(q)
        stats = mgr.search_stats
        assert stats.hits == 1
        assert stats.misses == 1

    def test_embedding_stats(self):
        engine = self._make_engine()
        mgr = CacheManager(engine)
        mgr.put_embedding("a", [0.1])
        mgr.get_embedding("a")   # hit
        mgr.get_embedding("b")   # miss
        stats = mgr.embedding_stats
        assert stats.hits == 1
        assert stats.misses == 1

    def test_cache_key_deterministic(self):
        k1 = _cache_key("q", "u", ("fact", "semantic"))
        k2 = _cache_key("q", "u", ("fact", "semantic"))
        assert k1 == k2

    def test_cache_key_differs_by_user(self):
        k1 = _cache_key("q", "u1", None)
        k2 = _cache_key("q", "u2", None)
        assert k1 != k2

    def test_cache_key_none_user(self):
        k = _cache_key("q", None, None)
        assert isinstance(k, str) and len(k) == 64
