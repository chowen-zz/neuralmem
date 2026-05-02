"""Tests for neuralmem.core.metrics — MetricsCollector."""
from __future__ import annotations

import logging
import time

import pytest

from neuralmem.core.metrics import MetricsCollector

# ------------------------------------------------------------------
# Counter tests
# ------------------------------------------------------------------


class TestCounterRecording:
    def test_basic_counter(self) -> None:
        mc = MetricsCollector()
        mc.record_counter("hits")
        mc.record_counter("hits")
        assert mc._counters["hits"] == 2

    def test_counter_with_value(self) -> None:
        mc = MetricsCollector()
        mc.record_counter("bytes", 1024)
        assert mc._counters["bytes"] == 1024

    def test_counter_with_tags(self) -> None:
        mc = MetricsCollector()
        mc.record_counter("req", method="GET")
        mc.record_counter("req", method="POST")
        mc.record_counter("req", method="GET")
        assert mc._counters["req{method=GET}"] == 2
        assert mc._counters["req{method=POST}"] == 1


# ------------------------------------------------------------------
# Histogram tests
# ------------------------------------------------------------------


class TestHistogramRecording:
    def test_basic_histogram(self) -> None:
        mc = MetricsCollector()
        mc.record_histogram("latency", 0.1)
        mc.record_histogram("latency", 0.2)
        assert mc._histograms["latency"] == [0.1, 0.2]

    def test_histogram_with_tags(self) -> None:
        mc = MetricsCollector()
        mc.record_histogram("latency", 0.5, endpoint="/api")
        mc.record_histogram("latency", 1.5, endpoint="/api")
        key = "latency{endpoint=/api}"
        assert len(mc._histograms[key]) == 2


# ------------------------------------------------------------------
# Timer tests
# ------------------------------------------------------------------


class TestTimer:
    def test_timer_records_duration(self) -> None:
        mc = MetricsCollector()
        with mc.timer("sleep_test"):
            time.sleep(0.01)
        key = "sleep_test.duration"
        assert key in mc._histograms
        assert len(mc._histograms[key]) == 1
        assert mc._histograms[key][0] >= 0.005  # at least 5ms

    def test_timer_records_on_exception(self) -> None:
        mc = MetricsCollector()
        try:
            with mc.timer("boom"):
                raise ValueError("test error")
        except ValueError:
            pass
        key = "boom.duration"
        assert key in mc._histograms
        assert len(mc._histograms[key]) == 1

    def test_timer_with_tags(self) -> None:
        mc = MetricsCollector()
        with mc.timer("op", op="fetch"):
            pass
        assert "op.duration{op=fetch}" in mc._histograms


# ------------------------------------------------------------------
# get_metrics tests
# ------------------------------------------------------------------


class TestGetMetrics:
    def test_empty_metrics(self) -> None:
        mc = MetricsCollector()
        result = mc.get_metrics()
        assert result == {"counters": {}, "histograms": {}}

    def test_metrics_structure(self) -> None:
        mc = MetricsCollector()
        mc.record_counter("hits", 5)
        mc.record_histogram("latency", 0.1)
        mc.record_histogram("latency", 0.3)
        mc.record_histogram("latency", 0.2)

        result = mc.get_metrics()

        assert result["counters"]["hits"] == 5
        hist = result["histograms"]["latency"]
        assert hist["count"] == 3
        assert hist["min"] == 0.1
        assert hist["max"] == 0.3
        assert hist["mean"] == pytest.approx(0.2)
        # sorted: [0.1, 0.2, 0.3] -> p50 index 1 = 0.2
        assert hist["p50"] == pytest.approx(0.2)
        assert "p95" in hist
        assert "p99" in hist

    def test_single_value_histogram(self) -> None:
        mc = MetricsCollector()
        mc.record_histogram("x", 1.0)
        result = mc.get_metrics()
        hist = result["histograms"]["x"]
        assert hist["count"] == 1
        assert hist["p50"] == 1.0
        assert hist["p95"] == 1.0
        assert hist["p99"] == 1.0


# ------------------------------------------------------------------
# Reset tests
# ------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all(self) -> None:
        mc = MetricsCollector()
        mc.record_counter("a", 10)
        mc.record_histogram("b", 1.0)
        mc.reset()
        assert mc._counters == {}
        assert mc._histograms == {}


# ------------------------------------------------------------------
# _make_key tests
# ------------------------------------------------------------------


class TestMakeKey:
    def test_no_tags(self) -> None:
        assert MetricsCollector._make_key("foo", {}) == "foo"

    def test_single_tag(self) -> None:
        assert MetricsCollector._make_key("foo", {"a": "1"}) == "foo{a=1}"

    def test_multiple_tags_sorted(self) -> None:
        # Tags should be sorted by key
        result = MetricsCollector._make_key("foo", {"z": "last", "a": "first"})
        assert result == "foo{a=first,z=last}"


# ------------------------------------------------------------------
# Disabled mode
# ------------------------------------------------------------------


class TestDisabledMode:
    def test_timer_still_records_when_disabled(self) -> None:
        mc = MetricsCollector(enabled=False)
        with mc.timer("x"):
            pass
        assert "x.duration" in mc._histograms

    def test_no_logging_when_disabled(self, caplog) -> None:
        mc = MetricsCollector(enabled=False)
        with caplog.at_level(logging.INFO, logger="neuralmem.metrics"):
            with mc.timer("silent"):
                pass
        assert "silent" not in caplog.text
