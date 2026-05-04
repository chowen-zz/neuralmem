"""Tests for NeuralMem V1.8 workload-aware cache."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.perf.workload_cache import WorkloadAwareCache, CacheStrategy, WorkloadProfile


class TestCacheOperations:
    def test_put_and_get(self):
        cache = WorkloadAwareCache(capacity=10)
        cache.put("k1", "v1")
        assert cache.get("k1") == "v1"

    def test_miss_returns_none(self):
        cache = WorkloadAwareCache(capacity=10)
        assert cache.get("missing") is None

    def test_capacity_eviction(self):
        cache = WorkloadAwareCache(capacity=2)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        assert cache.get("k1") is None  # evicted

    def test_lru_eviction(self):
        cache = WorkloadAwareCache(capacity=2, strategy=CacheStrategy.LRU)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.get("k1")  # access k1
        cache.put("k3", "v3")  # evict k2
        assert cache.get("k2") is None
        assert cache.get("k1") == "v1"


class TestWorkloadAnalysis:
    def test_sequential_pattern(self):
        cache = WorkloadAwareCache()
        for i in range(100):
            cache.put(f"k{i}", f"v{i}")
            cache.get(f"k{i}")
            cache.get(f"k{i}")
        profile = cache.analyze_workload()
        assert profile.pattern in ("sequential", "mixed")

    def test_random_pattern(self):
        cache = WorkloadAwareCache()
        import random
        random.seed(42)
        for i in range(100):
            key = f"k{random.randint(0, 999)}"
            cache.put(key, f"v{i}")
            cache.get(key)
        profile = cache.analyze_workload()
        assert profile.access_pattern == "random"

    def test_hit_rate_calculation(self):
        cache = WorkloadAwareCache()
        cache.put("k1", "v1")
        for _ in range(10):
            cache.get("k1")
        for _ in range(10):
            cache.get("missing")
        profile = cache.analyze_workload()
        assert abs(profile.hit_rate - 0.5) < 0.01


class TestStrategySwitching:
    def test_recommend_lru_for_sequential(self):
        cache = WorkloadAwareCache()
        for i in range(50):
            cache.put(f"k{i}", f"v{i}")
            cache.get(f"k{i}")
            cache.get(f"k{i}")
        assert cache.recommend_strategy() == CacheStrategy.LRU

    def test_recommend_predictive_for_low_hit_rate(self):
        cache = WorkloadAwareCache()
        for i in range(100):
            cache.put(f"k{i}", f"v{i}")
            cache.get("missing")  # all misses
        assert cache.recommend_strategy() == CacheStrategy.PREDICTIVE

    def test_auto_switch_triggers_callback(self):
        cb = MagicMock()
        cache = WorkloadAwareCache(strategy=CacheStrategy.LFU)
        cache.set_strategy_switch_callback(cb)
        for i in range(50):
            cache.put(f"k{i}", f"v{i}")
            cache.get(f"k{i}")
            cache.get(f"k{i}")
        cache.auto_switch_strategy()
        cb.assert_called_once()


class TestStats:
    def test_get_stats(self):
        cache = WorkloadAwareCache(capacity=100)
        cache.put("k1", "v1")
        cache.get("k1")
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hit_count"] == 1
        assert stats["hit_rate"] == 1.0

    def test_reset(self):
        cache = WorkloadAwareCache()
        cache.put("k1", "v1")
        cache.get("k1")
        cache.reset()
        assert cache.get("k1") is None
        assert cache.get_stats()["hit_count"] == 0
