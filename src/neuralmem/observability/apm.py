"""OpenTelemetry APM integration for NeuralMem."""
from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable

_current_trace: ContextVar[str | None] = ContextVar("trace_id", default=None)
_current_span: ContextVar[str | None] = ContextVar("span_id", default=None)


@dataclass
class TraceSpan:
    name: str
    span_id: str
    trace_id: str
    parent_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    status: str = "ok"

    def finish(self, status: str = "ok") -> None:
        self.end_time = time.time()
        self.status = status

    def add_event(self, name: str, attrs: dict | None = None) -> None:
        self.events.append({"name": name, "timestamp": time.time(), "attributes": attrs or {}})

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round((self.end_time - self.start_time) * 1000, 2) if self.end_time else None,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
        }


@dataclass
class TraceContext:
    trace_id: str
    spans: list[TraceSpan] = field(default_factory=list)
    baggage: dict[str, str] = field(default_factory=dict)

    def get_root_span(self) -> TraceSpan | None:
        return next((s for s in self.spans if s.parent_id is None), None)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "spans": [s.to_dict() for s in self.spans],
            "baggage": self.baggage,
        }


class APMIntegration:
    """OpenTelemetry-style APM with pluggable exporters."""

    def __init__(self, service_name: str = "neuralmem", exporter: Callable | None = None) -> None:
        self.service_name = service_name
        self._exporter = exporter
        self._spans: list[TraceSpan] = []
        self._active_spans: dict[str, TraceSpan] = {}

    def start_trace(self, name: str, attributes: dict | None = None) -> TraceContext:
        trace_id = str(uuid.uuid4())
        span = TraceSpan(name=name, span_id=str(uuid.uuid4())[:16], trace_id=trace_id, attributes=attributes or {})
        self._spans.append(span)
        self._active_spans[span.span_id] = span
        _current_trace.set(trace_id)
        _current_span.set(span.span_id)
        return TraceContext(trace_id=trace_id, spans=[span])

    def start_span(self, name: str, parent_id: str | None = None, attributes: dict | None = None) -> TraceSpan:
        trace_id = _current_trace.get() or str(uuid.uuid4())
        parent = parent_id or _current_span.get()
        span = TraceSpan(
            name=name,
            span_id=str(uuid.uuid4())[:16],
            trace_id=trace_id,
            parent_id=parent,
            attributes=attributes or {},
        )
        self._spans.append(span)
        self._active_spans[span.span_id] = span
        _current_span.set(span.span_id)
        return span

    def finish_span(self, span: TraceSpan, status: str = "ok") -> None:
        span.finish(status)
        self._active_spans.pop(span.span_id, None)
        if span.parent_id and span.parent_id in self._active_spans:
            _current_span.set(span.parent_id)

    def get_current_trace_id(self) -> str | None:
        return _current_trace.get()

    def get_current_span_id(self) -> str | None:
        return _current_span.get()

    def inject_context(self, carrier: dict) -> dict:
        trace_id = _current_trace.get()
        span_id = _current_span.get()
        if trace_id:
            carrier["traceparent"] = f"00-{trace_id.replace('-', '')}-{span_id or '0000000000000000'}-01"
        return carrier

    def extract_context(self, carrier: dict) -> dict:
        tp = carrier.get("traceparent", "")
        parts = tp.split("-")
        if len(parts) >= 3:
            return {"trace_id": parts[1], "span_id": parts[2]}
        return {}

    def export_trace(self, trace: TraceContext) -> dict:
        payload = {
            "service": self.service_name,
            **trace.to_dict(),
        }
        if self._exporter:
            self._exporter(payload)
        return payload

    def get_all_spans(self) -> list[TraceSpan]:
        return list(self._spans)

    def reset(self) -> None:
        self._spans.clear()
        self._active_spans.clear()
