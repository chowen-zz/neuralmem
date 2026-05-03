"""Performance optimization 单元测试 — 全部使用 mock."""
from __future__ import annotations

from unittest.mock import MagicMock

from neuralmem.perf.query_cache import QueryCache
from neuralmem.perf.prefetch import PrefetchEngine
from neuralmem.perf.batch_processor import BatchProcessor
from neuralmem.perf.metrics import PerformanceMetrics


# --------------------------------------------------------------------------- #
# QueryCache
# --------------------------------------------------------------------------- #

def test_query_cache_init():
    cache = QueryCache(max_size=100)
    assert cache is not None


def test_query_cache_get_set():
    cache = QueryCache(max_size=100)
    cache.set("query1", ["result1"])
    result = cache.get("query1")
    assert result == ["result1"]


def test_query_cache_miss():
    cache = QueryCache(max_size=100)
    result = cache.get("nonexistent")
    assert result is None


def test_query_cache_ttl_expiry():
    cache = QueryCache(max_size=100, ttl_seconds=0.001)
    cache.set("query1", ["result1"])
    import time
    time.sleep(0.01)
    result = cache.get("query1")
    assert result is None


def test_query_cache_hit_rate():
    cache = QueryCache(max_size=100)
    cache.set("query1", ["result1"])
    cache.get("query1")  # hit
    cache.get("query2")  # miss
    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1


# --------------------------------------------------------------------------- #
# PrefetchEngine
# --------------------------------------------------------------------------- #

def test_prefetch_init():
    engine = PrefetchEngine()
    assert engine is not None


def test_prefetch_record_query():
    engine = PrefetchEngine()
    engine.record_query("query1")
    engine.record_query("query2")
    stats = engine.stats()
    assert stats.predictions_made == 0  # no predictions triggered yet


def test_prefetch_predict():
    engine = PrefetchEngine()
    engine.record_query("query1")
    engine.record_query("query2")
    predictions = engine.predict_next("query1")
    assert isinstance(predictions, list)


# --------------------------------------------------------------------------- #
# BatchProcessor
# --------------------------------------------------------------------------- #

def test_batch_processor_init():
    mem = MagicMock()
    bp = BatchProcessor(mem, max_workers=4)
    assert bp.max_workers == 4


def test_batch_processor_batch_search():
    mem = MagicMock()
    mem.recall = MagicMock(return_value=[MagicMock(id="m1")])
    bp = BatchProcessor(mem, max_workers=2)
    
    results = bp.batch_search(["q1", "q2"], limit=5)
    assert len(results.items) == 2


# --------------------------------------------------------------------------- #
# PerformanceMetrics
# --------------------------------------------------------------------------- #

def test_metrics_init():
    metrics = PerformanceMetrics()
    assert metrics is not None


def test_metrics_record_latency():
    metrics = PerformanceMetrics()
    metrics.record_latency("recall", 50.0)
    metrics.record_latency("recall", 100.0)
    metrics.record_latency("recall", 150.0)
    
    stats = metrics.get_operation_metrics("recall")
    assert stats.total_calls == 3


def test_metrics_health_check():
    metrics = PerformanceMetrics(p99_target_ms=100)
    # Record fast operations
    for _ in range(10):
        metrics.record_latency("recall", 50.0)
    
    healthy, reason = metrics.is_healthy()
    assert healthy is True


def test_metrics_unhealthy():
    metrics = PerformanceMetrics(p99_target_ms=10)
    # Record slow operations
    for _ in range(10):
        metrics.record_latency("recall", 100.0)
    
    healthy, reason = metrics.is_healthy()
    assert healthy is False
