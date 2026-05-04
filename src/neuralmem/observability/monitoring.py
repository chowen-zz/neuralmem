"""Prometheus-style performance monitoring for NeuralMem."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable


class MetricType(Enum):
    COUNTER = auto()
    HISTOGRAM = auto()
    GAUGE = auto()


@dataclass
class MetricValue:
    name: str
    metric_type: MetricType
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AlertRule:
    name: str
    metric_name: str
    condition: str  # ">", "<", "==", ">=", "<="
    threshold: float
    duration_sec: float = 60.0
    severity: str = "warning"
    _fired_at: float | None = None

    def evaluate(self, value: float) -> bool:
        ops = {
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
        }
        return ops.get(self.condition, lambda a, b: False)(value, self.threshold)


@dataclass
class Alert:
    rule: AlertRule
    value: float
    fired_at: float
    resolved: bool = False


class PerformanceMonitoring:
    """Prometheus-style metrics with alerting."""

    def __init__(self, alert_handler: Callable | None = None) -> None:
        self._metrics: list[MetricValue] = []
        self._rules: list[AlertRule] = []
        self._alerts: list[Alert] = []
        self._alert_handler = alert_handler

    def counter(self, name: str, value: float = 1.0, labels: dict | None = None) -> None:
        self._metrics.append(MetricValue(name, MetricType.COUNTER, value, labels or {}))

    def histogram(self, name: str, value: float, labels: dict | None = None) -> None:
        self._metrics.append(MetricValue(name, MetricType.HISTOGRAM, value, labels or {}))

    def gauge(self, name: str, value: float, labels: dict | None = None) -> None:
        # overwrite latest gauge for same name+labels
        labels = labels or {}
        key = tuple(sorted(labels.items()))
        self._metrics = [m for m in self._metrics if not (
            m.name == name and m.metric_type == MetricType.GAUGE and tuple(sorted(m.labels.items())) == key
        )]
        self._metrics.append(MetricValue(name, MetricType.GAUGE, value, labels))

    def get_metrics(self, name: str | None = None, metric_type: MetricType | None = None) -> list[MetricValue]:
        out = self._metrics
        if name:
            out = [m for m in out if m.name == name]
        if metric_type:
            out = [m for m in out if m.metric_type == metric_type]
        return out

    def get_percentile(self, name: str, p: float) -> float | None:
        vals = [m.value for m in self._metrics if m.name == name and m.metric_type == MetricType.HISTOGRAM]
        if not vals:
            return None
        s = sorted(vals)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    def add_alert_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def check_alerts(self) -> list[Alert]:
        now = time.time()
        new_alerts: list[Alert] = []
        for rule in self._rules:
            vals = [m.value for m in self._metrics if m.name == rule.metric_name]
            if not vals:
                continue
            latest = vals[-1]
            if rule.evaluate(latest):
                if rule._fired_at is None:
                    rule._fired_at = now
                if now - rule._fired_at >= rule.duration_sec:
                    alert = Alert(rule, latest, now)
                    self._alerts.append(alert)
                    new_alerts.append(alert)
                    if self._alert_handler:
                        self._alert_handler(alert)
            else:
                rule._fired_at = None
        return new_alerts

    def get_active_alerts(self) -> list[Alert]:
        return [a for a in self._alerts if not a.resolved]

    def resolve_alert(self, alert: Alert) -> None:
        alert.resolved = True

    def export_prometheus(self) -> str:
        lines: list[str] = []
        for m in self._metrics:
            label_str = ",".join(f'{k}="{v}"' for k, v in m.labels.items())
            if label_str:
                lines.append(f'{m.name}{{{label_str}}} {m.value}')
            else:
                lines.append(f'{m.name} {m.value}')
        return "\n".join(lines)

    def get_grafana_template(self) -> dict:
        return {
            "dashboard": {
                "title": "NeuralMem Performance",
                "panels": [
                    {"title": "Recall P99", "type": "graph", "targets": [{"expr": "recall_latency_p99"}]},
                    {"title": "Memory Count", "type": "stat", "targets": [{"expr": "memory_count"}]},
                    {"title": "Cache Hit Rate", "type": "gauge", "targets": [{"expr": "cache_hit_rate"}]},
                ],
            }
        }

    def reset(self) -> None:
        self._metrics.clear()
        self._rules.clear()
        self._alerts.clear()
