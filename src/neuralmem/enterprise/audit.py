"""Audit logger — compliance-grade audit trail (GDPR / SOC2).

Thread-safe in-memory ring buffer.  No external database dependency.
All events are kept in memory with configurable retention.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ComplianceStandard(str, Enum):
    """Supported compliance frameworks."""

    GDPR = "gdpr"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    CCPA = "ccpa"
    ISO27001 = "iso27001"


class AuditEventType(str, Enum):
    """Categories of auditable events."""

    MEMORY_CREATE = "memory_create"
    MEMORY_READ = "memory_read"
    MEMORY_UPDATE = "memory_update"
    MEMORY_DELETE = "memory_delete"
    MEMORY_EXPORT = "memory_export"
    TENANT_CREATE = "tenant_create"
    TENANT_DELETE = "tenant_delete"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGE = "permission_change"
    DATA_PURGE = "data_purge"
    LOGIN = "login"
    LOGOUT = "logout"


@dataclass(frozen=True)
class AuditEvent:
    """A single auditable event.

    Fields are intentionally broad to cover GDPR Article 30 (record of processing)
    and SOC2 CC6.1 / CC7.2 (logical access / system operations).
    """

    timestamp: datetime
    event_type: AuditEventType
    tenant_id: str | None = None
    user_id: str | None = None
    memory_id: str | None = None
    action: str = ""
    resource: str = ""
    success: bool = True
    details: dict[str, Any] = field(default_factory=dict)
    compliance_tags: tuple[str, ...] = field(default_factory=tuple)
    ip_address: str | None = None
    session_id: str | None = None


class AuditLogger:
    """Thread-safe compliance audit logger with in-memory ring buffer.

    Parameters
    ----------
    max_events
        Maximum number of events to retain in memory.  Older events are
        discarded when the buffer is full (ring buffer behaviour).
    compliance_standards
        Which standards to tag events for automatically.
    """

    def __init__(
        self,
        max_events: int = 50_000,
        compliance_standards: tuple[ComplianceStandard, ...] = (
            ComplianceStandard.GDPR,
            ComplianceStandard.SOC2,
        ),
    ) -> None:
        self._max_events = max_events
        self._standards = compliance_standards
        self._buffer: deque[AuditEvent] = deque(maxlen=max_events)
        self._lock = threading.RLock()
        self._event_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log(
        self,
        event_type: AuditEventType,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        memory_id: str | None = None,
        action: str = "",
        resource: str = "",
        success: bool = True,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
        session_id: str | None = None,
    ) -> AuditEvent:
        """Log an audit event and return it (thread-safe)."""
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            tenant_id=tenant_id,
            user_id=user_id,
            memory_id=memory_id,
            action=action,
            resource=resource,
            success=success,
            details=details or {},
            compliance_tags=tuple(s.value for s in self._standards),
            ip_address=ip_address,
            session_id=session_id,
        )
        with self._lock:
            self._buffer.append(event)
            self._event_counts[event_type.value] = (
                self._event_counts.get(event_type.value, 0) + 1
            )
        return event

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def log_memory_access(
        self,
        event_type: AuditEventType,
        tenant_id: str,
        user_id: str,
        memory_id: str,
        success: bool = True,
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Log a memory CRUD event."""
        return self.log(
            event_type=event_type,
            tenant_id=tenant_id,
            user_id=user_id,
            memory_id=memory_id,
            action=event_type.value,
            resource="memory",
            success=success,
            details=details,
        )

    def log_tenant_event(
        self,
        event_type: AuditEventType,
        tenant_id: str,
        user_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Log a tenant lifecycle event."""
        return self.log(
            event_type=event_type,
            tenant_id=tenant_id,
            user_id=user_id,
            action=event_type.value,
            resource="tenant",
            details=details,
        )

    def log_access_denied(
        self,
        tenant_id: str | None,
        user_id: str | None,
        action: str,
        resource: str,
        reason: str = "",
    ) -> AuditEvent:
        """Log an access-denied event (important for SOC2 CC6.1)."""
        return self.log(
            event_type=AuditEventType.ACCESS_DENIED,
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            resource=resource,
            success=False,
            details={"reason": reason},
        )

    def log_data_purge(
        self,
        tenant_id: str,
        user_id: str,
        memory_ids: list[str],
        reason: str = "gdpr_request",
    ) -> AuditEvent:
        """Log a data-purge / right-to-erasure event (GDPR Article 17)."""
        return self.log(
            event_type=AuditEventType.DATA_PURGE,
            tenant_id=tenant_id,
            user_id=user_id,
            action="purge",
            resource="memory",
            details={"memory_ids": memory_ids, "reason": reason},
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        event_type: AuditEventType | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        memory_id: str | None = None,
        success: bool | None = None,
        limit: int = 100,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[AuditEvent]:
        """Query audit events with optional filters (thread-safe).

        Results are returned newest-first.
        """
        with self._lock:
            results: list[AuditEvent] = []
            for event in reversed(self._buffer):
                if event_type and event.event_type != event_type:
                    continue
                if tenant_id is not None and event.tenant_id != tenant_id:
                    continue
                if user_id is not None and event.user_id != user_id:
                    continue
                if memory_id is not None and event.memory_id != memory_id:
                    continue
                if success is not None and event.success != success:
                    continue
                if since is not None and event.timestamp < since:
                    continue
                if until is not None and event.timestamp > until:
                    continue
                results.append(event)
                if len(results) >= limit:
                    break
            return results

    def query_by_compliance(
        self,
        standard: ComplianceStandard,
        *,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Return events tagged for a specific compliance standard."""
        with self._lock:
            results: list[AuditEvent] = []
            for event in reversed(self._buffer):
                if standard.value in event.compliance_tags:
                    results.append(event)
                    if len(results) >= limit:
                        break
            return results

    # ------------------------------------------------------------------
    # Aggregation / reporting
    # ------------------------------------------------------------------

    def get_event_counts(self) -> dict[str, int]:
        """Return a mapping of event-type -> count."""
        with self._lock:
            return dict(self._event_counts)

    @property
    def count(self) -> int:
        """Current number of events in the buffer."""
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        """Clear the in-memory buffer and counters."""
        with self._lock:
            self._buffer.clear()
            self._event_counts.clear()

    # ------------------------------------------------------------------
    # GDPR / SOC2 helpers
    # ------------------------------------------------------------------

    def get_user_activity(
        self,
        user_id: str,
        *,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Return all activity for a specific user (GDPR Article 15 — right of access)."""
        return self.query(
            user_id=user_id,
            tenant_id=tenant_id,
            limit=limit,
        )

    def get_tenant_activity(
        self,
        tenant_id: str,
        *,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Return all activity within a tenant (SOC2 CC7.2 — monitoring)."""
        return self.query(
            tenant_id=tenant_id,
            limit=limit,
        )

    def export_for_compliance(
        self,
        standard: ComplianceStandard,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Export audit events as plain dicts suitable for compliance reporting."""
        events = self.query(
            tenant_id=tenant_id,
            user_id=user_id,
            limit=self._max_events,
        )
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type.value,
                "tenant_id": e.tenant_id,
                "user_id": e.user_id,
                "memory_id": e.memory_id,
                "action": e.action,
                "resource": e.resource,
                "success": e.success,
                "details": e.details,
                "compliance_standard": standard.value,
            }
            for e in events
            if standard.value in e.compliance_tags
        ]
