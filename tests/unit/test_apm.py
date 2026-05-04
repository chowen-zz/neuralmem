"""Tests for NeuralMem V1.7 APM integration."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.observability.apm import APMIntegration, TraceSpan, TraceContext


class TestTraceSpan:
    def test_create_span(self):
        span = TraceSpan(name="test", span_id="s1", trace_id="t1")
        assert span.name == "test"
        assert span.status == "ok"
        assert span.end_time is None

    def test_finish_span(self):
        span = TraceSpan(name="test", span_id="s1", trace_id="t1")
        span.finish("error")
        assert span.end_time is not None
        assert span.status == "error"

    def test_add_event(self):
        span = TraceSpan(name="test", span_id="s1", trace_id="t1")
        span.add_event("db_query", {"sql": "SELECT * FROM memories"})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "db_query"

    def test_to_dict(self):
        span = TraceSpan(name="test", span_id="s1", trace_id="t1")
        span.finish()
        d = span.to_dict()
        assert d["name"] == "test"
        assert d["duration_ms"] is not None


class TestAPMIntegration:
    def test_start_trace(self):
        apm = APMIntegration()
        ctx = apm.start_trace("remember", {"user_id": "u1"})
        assert ctx.trace_id is not None
        assert len(ctx.spans) == 1
        assert ctx.spans[0].name == "remember"

    def test_start_span_with_parent(self):
        apm = APMIntegration()
        root = apm.start_trace("remember")
        child = apm.start_span("extract", parent_id=root.spans[0].span_id)
        assert child.parent_id == root.spans[0].span_id

    def test_finish_span(self):
        apm = APMIntegration()
        ctx = apm.start_trace("remember")
        span = ctx.spans[0]
        apm.finish_span(span)
        assert span.end_time is not None

    def test_inject_context(self):
        apm = APMIntegration()
        apm.start_trace("remember")
        carrier = {}
        apm.inject_context(carrier)
        assert "traceparent" in carrier

    def test_extract_context(self):
        apm = APMIntegration()
        ctx = apm.extract_context({"traceparent": "00-abc123-def456-01"})
        assert ctx["trace_id"] == "abc123"
        assert ctx["span_id"] == "def456"

    def test_export_trace(self):
        mock_exporter = MagicMock()
        apm = APMIntegration(exporter=mock_exporter)
        ctx = apm.start_trace("remember")
        payload = apm.export_trace(ctx)
        assert payload["service"] == "neuralmem"
        mock_exporter.assert_called_once()

    def test_get_all_spans(self):
        apm = APMIntegration()
        apm.start_trace("remember")
        apm.start_trace("recall")
        assert len(apm.get_all_spans()) == 2

    def test_reset(self):
        apm = APMIntegration()
        apm.start_trace("remember")
        apm.reset()
        assert len(apm.get_all_spans()) == 0
