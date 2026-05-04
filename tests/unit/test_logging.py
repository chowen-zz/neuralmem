"""Tests for NeuralMem V1.7 structured logging."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.observability.logging import StructuredLogging, LogLevel, LogEntry, ErrorCluster


class TestLogEntry:
    def test_create_entry(self):
        entry = LogEntry(message="test", level=LogLevel.INFO)
        assert entry.level == LogLevel.INFO
        assert entry.correlation_id is not None

    def test_to_json(self):
        entry = LogEntry(message="test", level=LogLevel.INFO)
        json_str = entry.to_json()
        assert "test" in json_str
        assert "INFO" in json_str


class TestStructuredLogging:
    def test_debug_log(self):
        log = StructuredLogging()
        entry = log.debug("debug msg", {"key": "val"})
        assert entry.level == LogLevel.DEBUG
        assert entry.metadata["key"] == "val"

    def test_info_log(self):
        log = StructuredLogging()
        entry = log.info("info msg")
        assert entry.level == LogLevel.INFO

    def test_error_log(self):
        log = StructuredLogging()
        entry = log.error("error msg", exc=ValueError("bad"))
        assert entry.level == LogLevel.ERROR
        assert entry.exception == "bad"

    def test_critical_log(self):
        log = StructuredLogging()
        entry = log.critical("critical msg")
        assert entry.level == LogLevel.CRITICAL

    def test_sink_called(self):
        sink = MagicMock()
        log = StructuredLogging(sink=sink)
        log.info("test")
        sink.assert_called_once()

    def test_correlation_id(self):
        log = StructuredLogging()
        log.set_correlation_id("abc123")
        entry = log.info("test")
        assert entry.correlation_id == "abc123"

    def test_error_clustering(self):
        log = StructuredLogging()
        log.error("DB connection failed: timeout")
        log.error("DB connection failed: timeout")
        log.error("DB connection failed: refused")
        clusters = log.get_error_clusters()
        assert len(clusters) >= 1
        assert clusters[0].count >= 2

    def test_query_by_level(self):
        log = StructuredLogging()
        log.debug("d1")
        log.info("i1")
        log.info("i2")
        results = log.query(level=LogLevel.INFO)
        assert len(results) == 2

    def test_query_by_correlation_id(self):
        log = StructuredLogging()
        log.set_correlation_id("cid1")
        log.info("msg1")
        log.set_correlation_id("cid2")
        log.info("msg2")
        results = log.query(correlation_id="cid1")
        assert len(results) == 1
        assert results[0].message == "msg1"

    def test_export_jsonl(self):
        log = StructuredLogging()
        log.info("msg1")
        log.info("msg2")
        output = log.export_jsonl()
        lines = output.strip().split("\n")
        assert len(lines) == 2

    def test_export_for_elk(self):
        log = StructuredLogging()
        log.info("msg")
        docs = log.export_for_elk()
        assert len(docs) == 1
        assert docs[0]["message"] == "msg"

    def test_reset(self):
        log = StructuredLogging()
        log.info("msg")
        log.reset()
        assert len(log.get_entries()) == 0
        assert len(log.get_error_clusters()) == 0
