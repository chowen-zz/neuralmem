"""Audit logging for NeuralMem governance.

Thread-safe in-memory ring buffer with optional SQLite persistence.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from neuralmem.governance.risk import RiskLevel


@dataclass(frozen=True)
class AuditEvent:
    timestamp: datetime
    event_type: str  # REMEMBER, RECALL, FORGET, GOVERNANCE, STATE_CHANGE
    memory_id: str | None = None
    user_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel | None = None


class AuditLogger:
    """Thread-safe audit logger with in-memory ring buffer and optional SQLite persistence."""

    def __init__(
        self,
        max_events: int = 10_000,
        db_path: str | None = None,
    ) -> None:
        self._max_events = max_events
        self._buffer: deque[AuditEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._db_path = db_path

        if db_path:
            self._init_db(db_path)

    def _init_db(self, db_path: str) -> None:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    memory_id TEXT,
                    user_id TEXT,
                    details TEXT,
                    risk_level TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def log(self, event: AuditEvent) -> None:
        """Log an audit event (thread-safe)."""
        with self._lock:
            self._buffer.append(event)
            if self._db_path:
                self._persist(event)

    def _persist(self, event: AuditEvent) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            sql = (
                "INSERT INTO audit_events"
                " (timestamp, event_type, memory_id, user_id, details, risk_level)"
                " VALUES (?, ?, ?, ?, ?, ?)"
            )
            conn.execute(sql,
                (
                    event.timestamp.isoformat(),
                    event.event_type,
                    event.memory_id,
                    event.user_id,
                    json.dumps(event.details),
                    event.risk_level.value if event.risk_level else None,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def query(
        self,
        *,
        event_type: str | None = None,
        memory_id: str | None = None,
        user_id: str | None = None,
        risk_level: RiskLevel | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events with optional filters (thread-safe).

        Queries in-memory buffer. For large-scale persistence queries, use SQLite directly.
        """
        with self._lock:
            results: list[AuditEvent] = []
            for event in reversed(self._buffer):
                if event_type and event.event_type != event_type:
                    continue
                if memory_id and event.memory_id != memory_id:
                    continue
                if user_id and event.user_id != user_id:
                    continue
                if risk_level and event.risk_level != risk_level:
                    continue
                results.append(event)
                if len(results) >= limit:
                    break
            return results

    @property
    def count(self) -> int:
        """Current number of events in the buffer."""
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        """Clear the in-memory buffer."""
        with self._lock:
            self._buffer.clear()
