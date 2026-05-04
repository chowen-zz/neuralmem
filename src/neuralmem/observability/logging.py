"""Structured logging with error clustering for NeuralMem."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable


class LogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class LogEntry:
    message: str
    level: LogLevel
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source: str = "neuralmem"
    metadata: dict[str, str] = field(default_factory=dict)
    exception: str | None = None

    def to_json(self) -> str:
        return json.dumps({
            "message": self.message,
            "level": self.level.name,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "source": self.source,
            "metadata": self.metadata,
            "exception": self.exception,
        }, default=str)


@dataclass
class ErrorCluster:
    signature: str
    count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    sample_message: str = ""


class StructuredLogging:
    """JSON structured logging with correlation IDs and error clustering."""

    def __init__(self, sink: Callable | None = None) -> None:
        self._entries: list[LogEntry] = []
        self._sink = sink
        self._error_clusters: dict[str, ErrorCluster] = {}
        self._correlation_id: str | None = None

    def set_correlation_id(self, cid: str) -> None:
        self._correlation_id = cid

    def _create_entry(self, message: str, level: LogLevel, metadata: dict | None = None, exc: Exception | None = None) -> LogEntry:
        entry = LogEntry(
            message=message,
            level=level,
            correlation_id=self._correlation_id or str(uuid.uuid4())[:8],
            metadata=metadata or {},
            exception=str(exc) if exc else None,
        )
        self._entries.append(entry)
        if self._sink:
            self._sink(entry.to_json())
        if level in (LogLevel.ERROR, LogLevel.CRITICAL):
            self._cluster_error(entry)
        return entry

    def debug(self, message: str, metadata: dict | None = None) -> LogEntry:
        return self._create_entry(message, LogLevel.DEBUG, metadata)

    def info(self, message: str, metadata: dict | None = None) -> LogEntry:
        return self._create_entry(message, LogLevel.INFO, metadata)

    def warning(self, message: str, metadata: dict | None = None) -> LogEntry:
        return self._create_entry(message, LogLevel.WARNING, metadata)

    def error(self, message: str, metadata: dict | None = None, exc: Exception | None = None) -> LogEntry:
        return self._create_entry(message, LogLevel.ERROR, metadata, exc)

    def critical(self, message: str, metadata: dict | None = None, exc: Exception | None = None) -> LogEntry:
        return self._create_entry(message, LogLevel.CRITICAL, metadata, exc)

    def _cluster_error(self, entry: LogEntry) -> None:
        sig = entry.message.split(":")[0].strip()
        if sig not in self._error_clusters:
            self._error_clusters[sig] = ErrorCluster(signature=sig, sample_message=entry.message)
        cluster = self._error_clusters[sig]
        cluster.count += 1
        cluster.last_seen = entry.timestamp

    def get_entries(self, level: LogLevel | None = None, limit: int = 100) -> list[LogEntry]:
        out = self._entries
        if level:
            out = [e for e in out if e.level == level]
        return out[-limit:]

    def get_error_clusters(self) -> list[ErrorCluster]:
        return sorted(self._error_clusters.values(), key=lambda c: c.count, reverse=True)

    def export_jsonl(self) -> str:
        return "\n".join(e.to_json() for e in self._entries)

    def export_for_elk(self) -> list[dict]:
        return [json.loads(e.to_json()) for e in self._entries]

    def query(self, level: LogLevel | None = None, source: str | None = None, correlation_id: str | None = None, limit: int = 100) -> list[LogEntry]:
        out = self._entries
        if level:
            out = [e for e in out if e.level == level]
        if source:
            out = [e for e in out if e.source == source]
        if correlation_id:
            out = [e for e in out if e.correlation_id == correlation_id]
        return out[-limit:]

    def reset(self) -> None:
        self._entries.clear()
        self._error_clusters.clear()
        self._correlation_id = None
