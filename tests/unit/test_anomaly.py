"""Tests for NeuralMem V1.8 anomaly detection engine."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.diagnosis.anomaly import AnomalyDetectionEngine, AnomalyType, Anomaly


class TestLatencyAnomaly:
    def test_no_anomaly_with_few_samples(self):
        engine = AnomalyDetectionEngine()
        for _ in range(10):
            engine.record_latency(10.0)
        assert len(engine.get_anomalies()) == 0

    def test_detect_latency_spike(self):
        handler = MagicMock()
        engine = AnomalyDetectionEngine(alert_handler=handler)
        for _ in range(30):
            engine.record_latency(10.0)
        engine.record_latency(100.0)  # spike
        anomalies = engine.get_anomalies()
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.LATENCY_SPIKE
        handler.assert_called_once()

    def test_no_spike_within_normal_range(self):
        engine = AnomalyDetectionEngine()
        # Generate noisy data with significant std to avoid triggering on small variations
        import random
        random.seed(42)
        for _ in range(50):
            engine.record_latency(100.0 + random.gauss(0, 15.0))
        engine.record_latency(145.0)  # within 3 std of noisy data
        assert len(engine.get_anomalies()) == 0


class TestVectorDrift:
    def test_detect_drift(self):
        engine = AnomalyDetectionEngine()
        for _ in range(20):
            engine.record_vector([1.0, 0.0, 0.0])
        for _ in range(10):
            engine.record_vector([10.0, 0.0, 0.0])  # major drift
        anomalies = engine.get_anomalies()
        assert len(anomalies) >= 1
        assert anomalies[0].anomaly_type == AnomalyType.VECTOR_DRIFT

    def test_no_drift_with_similar_vectors(self):
        engine = AnomalyDetectionEngine()
        for _ in range(30):
            engine.record_vector([1.0, 0.0, 0.0])
        assert len(engine.get_anomalies()) == 0


class TestAccessPatternAnomaly:
    def test_detect_all_failed(self):
        engine = AnomalyDetectionEngine()
        for _ in range(50):
            engine.record_access("failed", "k1", 0.0)
        anomalies = engine.detect_access_pattern_anomaly()
        assert len(anomalies) == 1
        assert anomalies[0].severity == "critical"

    def test_no_anomaly_with_mixed(self):
        engine = AnomalyDetectionEngine()
        for i in range(50):
            op = "success" if i % 2 == 0 else "failed"
            engine.record_access(op, f"k{i}", 10.0)
        anomalies = engine.detect_access_pattern_anomaly()
        assert len(anomalies) == 0


class TestGetAnomalies:
    def test_filter_by_severity(self):
        engine = AnomalyDetectionEngine()
        for _ in range(30):
            engine.record_latency(10.0)
        engine.record_latency(100.0)
        high = engine.get_anomalies(severity="high")
        assert len(high) == 1

    def test_reset(self):
        engine = AnomalyDetectionEngine()
        for _ in range(30):
            engine.record_latency(10.0)
        engine.record_latency(100.0)
        engine.reset()
        assert len(engine.get_anomalies()) == 0
