"""Tests for neuralmem.ops.metrics — Prometheus-compatible MetricsCollector."""
from __future__ import annotations

import threading
import time

from neuralmem.ops.metrics import MetricsCollector

# ---------------------------------------------------------------------------
# Counter tests
# ---------------------------------------------------------------------------


class TestCounters:
    def test_inc_default(self):
        mc = MetricsCollector()
        mc.inc("total_recalls")
        assert mc.get_counter("total_recalls") == 1

    def test_inc_multiple(self):
        mc = MetricsCollector()
        mc.inc("total_recalls")
        mc.inc("total_recalls", 5)
        assert mc.get_counter("total_recalls") == 6

    def test_counter_zero_default(self):
        mc = MetricsCollector()
        assert mc.get_counter("nonexistent") == 0


# ---------------------------------------------------------------------------
# Gauge tests
# ---------------------------------------------------------------------------


class TestGauges:
    def test_set_gauge(self):
        mc = MetricsCollector()
        mc.set_gauge("memory_count", 42.0)
        assert mc.get_gauge("memory_count") == 42.0

    def test_gauge_overwrite(self):
        mc = MetricsCollector()
        mc.set_gauge("memory_count", 10.0)
        mc.set_gauge("memory_count", 20.0)
        assert mc.get_gauge("memory_count") == 20.0

    def test_gauge_zero_default(self):
        mc = MetricsCollector()
        assert mc.get_gauge("nonexistent") == 0.0


# ---------------------------------------------------------------------------
# Histogram tests
# ---------------------------------------------------------------------------


class TestHistograms:
    def test_observe(self):
        mc = MetricsCollector()
        mc.observe("recall_latency_ms", 12.5)
        mc.observe("recall_latency_ms", 25.0)
        assert mc.get_histogram("recall_latency_ms") == [12.5, 25.0]

    def test_histogram_empty(self):
        mc = MetricsCollector()
        assert mc.get_histogram("empty") == []

    def test_timer_records_ms(self):
        mc = MetricsCollector()
        with mc.timer("recall_latency_ms"):
            time.sleep(0.01)  # ~10ms
        values = mc.get_histogram("recall_latency_ms")
        assert len(values) == 1
        assert values[0] > 5.0  # at least 5ms


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


class TestConvenience:
    def test_record_recall(self):
        mc = MetricsCollector()
        mc.record_recall(15.0)
        assert mc.get_counter("total_recalls") == 1
        assert mc.get_histogram("recall_latency_ms") == [15.0]

    def test_record_remember(self):
        mc = MetricsCollector()
        mc.record_remember()
        mc.record_remember()
        assert mc.get_counter("total_remembers") == 2

    def test_record_error(self):
        mc = MetricsCollector()
        mc.record_error()
        assert mc.get_counter("error_count") == 1

    def test_update_cache_hit_rate(self):
        mc = MetricsCollector()
        mc.update_cache_hit_rate(0.85)
        assert mc.get_gauge("cache_hit_rate") == 0.85

    def test_cache_hit_rate_clamped(self):
        mc = MetricsCollector()
        mc.update_cache_hit_rate(1.5)
        assert mc.get_gauge("cache_hit_rate") == 1.0
        mc.update_cache_hit_rate(-0.1)
        assert mc.get_gauge("cache_hit_rate") == 0.0

    def test_update_memory_count(self):
        mc = MetricsCollector()
        mc.update_memory_count(1000)
        assert mc.get_gauge("memory_count") == 1000.0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_increments(self):
        mc = MetricsCollector()

        def worker():
            for _ in range(1000):
                mc.inc("total_recalls")

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert mc.get_counter("total_recalls") == 10_000


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_structure(self):
        mc = MetricsCollector()
        mc.inc("total_recalls")
        mc.set_gauge("memory_count", 50.0)
        mc.observe("recall_latency_ms", 10.0)
        snap = mc.snapshot()
        assert "counters" in snap
        assert "gauges" in snap
        assert "histograms" in snap
        assert "uptime_seconds" in snap
        assert snap["counters"]["total_recalls"] == 1
        assert snap["gauges"]["memory_count"] == 50.0
        assert snap["histograms"]["recall_latency_ms"]["count"] == 1

    def test_reset(self):
        mc = MetricsCollector()
        mc.inc("total_recalls")
        mc.set_gauge("memory_count", 50.0)
        mc.observe("recall_latency_ms", 10.0)
        mc.reset()
        assert mc.get_counter("total_recalls") == 0
        assert mc.get_gauge("memory_count") == 0.0
        assert mc.get_histogram("recall_latency_ms") == []


# ---------------------------------------------------------------------------
# Prometheus expose
# ---------------------------------------------------------------------------


class TestExpose:
    def test_expose_has_counter(self):
        mc = MetricsCollector()
        mc.inc("total_recalls")
        text = mc.expose()
        assert "neuralmem_total_recalls_total 1" in text
        assert "# TYPE neuralmem_total_recalls counter" in text

    def test_expose_has_gauge(self):
        mc = MetricsCollector()
        mc.set_gauge("memory_count", 42.0)
        text = mc.expose()
        assert "neuralmem_memory_count 42" in text
        assert "# TYPE neuralmem_memory_count gauge" in text

    def test_expose_has_histogram(self):
        mc = MetricsCollector()
        mc.observe("recall_latency_ms", 10.0)
        mc.observe("recall_latency_ms", 20.0)
        text = mc.expose()
        assert "neuralmem_recall_latency_ms_count 2" in text
        assert "neuralmem_recall_latency_ms_sum" in text
        assert '# TYPE neuralmem_recall_latency_ms histogram' in text
        assert 'le="5.0"' in text

    def test_expose_has_uptime(self):
        mc = MetricsCollector()
        text = mc.expose()
        assert "neuralmem_uptime_seconds" in text

    def test_expose_empty_is_valid(self):
        mc = MetricsCollector()
        text = mc.expose()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_expose_multiple_metrics(self):
        mc = MetricsCollector()
        mc.inc("total_recalls", 5)
        mc.inc("total_remembers", 3)
        mc.set_gauge("memory_count", 100.0)
        text = mc.expose()
        assert "total_recalls_total 5" in text
        assert "total_remembers_total 3" in text
        assert "memory_count 100" in text
