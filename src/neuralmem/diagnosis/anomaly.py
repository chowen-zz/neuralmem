"""Anomaly detection for memory access patterns and vector drift."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable
import time
import statistics


class AnomalyType(Enum):
    LATENCY_SPIKE = auto()
    VECTOR_DRIFT = auto()
    ACCESS_PATTERN = auto()
    CAPACITY = auto()


@dataclass
class Anomaly:
    anomaly_type: AnomalyType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    metric_value: float
    expected_range: tuple[float, float]
    timestamp: float = field(default_factory=time.time)


class AnomalyDetectionEngine:
    """Detect anomalies in memory system behavior."""

    def __init__(self, alert_handler: Callable[[Anomaly], None] | None = None) -> None:
        self._alert_handler = alert_handler
        self._latency_history: list[float] = []
        self._access_history: list[dict] = []
        self._vector_history: list[list[float]] = []
        self._anomalies: list[Anomaly] = []

    def record_latency(self, latency_ms: float) -> None:
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > 1000:
            self._latency_history.pop(0)
        self._check_latency_anomaly()

    def _check_latency_anomaly(self) -> None:
        if len(self._latency_history) < 30:
            return
        recent = self._latency_history[-30:]
        mean = statistics.mean(recent)
        std = statistics.stdev(recent) if len(recent) > 1 else 0
        latest = recent[-1]
        if std > 0 and latest > mean + 3 * std:
            anomaly = Anomaly(
                anomaly_type=AnomalyType.LATENCY_SPIKE,
                severity="high",
                description=f"Latency spike: {latest:.1f}ms (mean: {mean:.1f}ms, std: {std:.1f}ms)",
                metric_value=latest,
                expected_range=(mean - 2 * std, mean + 2 * std),
            )
            self._anomalies.append(anomaly)
            if self._alert_handler:
                self._alert_handler(anomaly)
        elif std > 0 and latest > mean + 2 * std:
            # Record medium severity but don't alert to avoid noise
            pass

    def record_access(self, operation: str, key: str, latency_ms: float) -> None:
        self._access_history.append({
            "operation": operation,
            "key": key,
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        })

    def record_vector(self, vector: list[float]) -> None:
        self._vector_history.append(vector)
        if len(self._vector_history) > 100:
            self._vector_history.pop(0)
        self._check_vector_drift()

    def _check_vector_drift(self) -> None:
        if len(self._vector_history) < 20:
            return
        recent = self._vector_history[-10:]
        baseline = self._vector_history[-20:-10]
        if not baseline or not recent:
            return
        # Simple drift: compare mean norms
        recent_norm = statistics.mean(sum(v[i]**2 for i in range(len(v))) ** 0.5 for v in recent)
        baseline_norm = statistics.mean(sum(v[i]**2 for i in range(len(v))) ** 0.5 for v in baseline)
        if baseline_norm > 0 and abs(recent_norm - baseline_norm) / baseline_norm > 0.5:
            anomaly = Anomaly(
                anomaly_type=AnomalyType.VECTOR_DRIFT,
                severity="medium",
                description=f"Vector drift detected: norm changed by {abs(recent_norm - baseline_norm) / baseline_norm:.1%}",
                metric_value=recent_norm,
                expected_range=(baseline_norm * 0.8, baseline_norm * 1.2),
            )
            self._anomalies.append(anomaly)
            if self._alert_handler:
                self._alert_handler(anomaly)

    def detect_access_pattern_anomaly(self) -> list[Anomaly]:
        if len(self._access_history) < 50:
            return []
        recent = self._access_history[-50:]
        ops = [r["operation"] for r in recent]
        unique_ops = len(set(ops))
        if unique_ops == 1 and ops[0] == "failed":
            anomaly = Anomaly(
                anomaly_type=AnomalyType.ACCESS_PATTERN,
                severity="critical",
                description="All recent operations failed - possible system outage",
                metric_value=1.0,
                expected_range=(0.0, 0.2),
            )
            self._anomalies.append(anomaly)
            if self._alert_handler:
                self._alert_handler(anomaly)
            return [anomaly]
        return []

    def get_anomalies(self, severity: str | None = None) -> list[Anomaly]:
        out = self._anomalies
        if severity:
            out = [a for a in out if a.severity == severity]
        return out

    def reset(self) -> None:
        self._latency_history.clear()
        self._access_history.clear()
        self._vector_history.clear()
        self._anomalies.clear()
