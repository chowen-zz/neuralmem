"""Tests for NeuralMem V1.7 performance monitoring."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.observability.monitoring import PerformanceMonitoring, MetricType, AlertRule, Alert


class TestMetricCollection:
    def test_counter(self):
        mon = PerformanceMonitoring()
        mon.counter("requests", 1.0, {"endpoint": "remember"})
        metrics = mon.get_metrics("requests")
        assert len(metrics) == 1
        assert metrics[0].metric_type == MetricType.COUNTER

    def test_histogram(self):
        mon = PerformanceMonitoring()
        mon.histogram("latency", 45.0)
        metrics = mon.get_metrics("latency", MetricType.HISTOGRAM)
        assert len(metrics) == 1
        assert metrics[0].value == 45.0

    def test_gauge_overwrite(self):
        mon = PerformanceMonitoring()
        mon.gauge("memory_usage", 50.0)
        mon.gauge("memory_usage", 60.0)
        metrics = mon.get_metrics("memory_usage", MetricType.GAUGE)
        assert len(metrics) == 1
        assert metrics[0].value == 60.0

    def test_get_percentile(self):
        mon = PerformanceMonitoring()
        for v in [10, 20, 30, 40, 50]:
            mon.histogram("latency", float(v))
        p99 = mon.get_percentile("latency", 99)
        assert p99 == 50.0


class TestAlerting:
    def test_alert_rule_evaluate(self):
        rule = AlertRule("high_latency", "latency", ">", 100.0)
        assert rule.evaluate(150.0) is True
        assert rule.evaluate(50.0) is False

    def test_check_alerts(self):
        mon = PerformanceMonitoring()
        rule = AlertRule("high_latency", "latency", ">", 100.0, duration_sec=0.0)
        mon.add_alert_rule(rule)
        mon.histogram("latency", 150.0)
        alerts = mon.check_alerts()
        assert len(alerts) == 1
        assert alerts[0].rule.name == "high_latency"

    def test_alert_handler_called(self):
        handler = MagicMock()
        mon = PerformanceMonitoring(alert_handler=handler)
        rule = AlertRule("high_latency", "latency", ">", 100.0, duration_sec=0.0)
        mon.add_alert_rule(rule)
        mon.histogram("latency", 150.0)
        mon.check_alerts()
        handler.assert_called_once()

    def test_resolve_alert(self):
        mon = PerformanceMonitoring()
        rule = AlertRule("high_latency", "latency", ">", 100.0, duration_sec=0.0)
        mon.add_alert_rule(rule)
        mon.histogram("latency", 150.0)
        alerts = mon.check_alerts()
        mon.resolve_alert(alerts[0])
        assert len(mon.get_active_alerts()) == 0


class TestPrometheusExport:
    def test_export_prometheus(self):
        mon = PerformanceMonitoring()
        mon.counter("requests", 1.0, {"endpoint": "remember"})
        mon.gauge("memory_usage", 60.0)
        output = mon.export_prometheus()
        assert "requests" in output
        assert "memory_usage" in output

    def test_grafana_template(self):
        mon = PerformanceMonitoring()
        tpl = mon.get_grafana_template()
        assert "dashboard" in tpl
        assert len(tpl["dashboard"]["panels"]) == 3

    def test_reset(self):
        mon = PerformanceMonitoring()
        mon.counter("requests", 1.0)
        mon.reset()
        assert len(mon.get_metrics()) == 0
