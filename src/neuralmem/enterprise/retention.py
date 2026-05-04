"""Data Retention Enforcer — GDPR deletion automation, tiered retention,
compliance verification.

NeuralMem Enterprise V1.7 compliance certification automation.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

from neuralmem.enterprise.audit import AuditEventType, AuditLogger, ComplianceStandard


# ------------------------------------------------------------------
# Retention tiers
# ------------------------------------------------------------------

class RetentionTier(str, Enum):
    """Data retention tier classification."""

    HOT = "hot"       # Active data, full access, no deletion
    WARM = "warm"     # Less active, restricted access
    COLD = "cold"     # Archived, minimal access
    ARCHIVE = "archive"  # Long-term archive, compliance hold


# ------------------------------------------------------------------
# Retention policy
# ------------------------------------------------------------------

@dataclass
class RetentionPolicy:
    """Retention policy for a data category.

    Attributes
    ----------
    category
        Data category (e.g. "memory", "audit", "user_profile").
    tier
        Retention tier (hot / warm / cold / archive).
    retention_days
        Number of days before data is eligible for deletion.
        None means indefinite (archive tier only).
    legal_hold
        If True, data cannot be deleted regardless of retention period.
    gdpr_applies
        If True, GDPR right-to-erasure applies to this category.
    auto_delete
        If True, data is automatically deleted after retention period.
    """

    category: str
    tier: RetentionTier
    retention_days: int | None
    legal_hold: bool = False
    gdpr_applies: bool = True
    auto_delete: bool = False


@dataclass
class DataRecord:
    """A tracked data record for retention management."""

    record_id: str
    category: str
    tenant_id: str
    user_id: str | None
    created_at: datetime
    tier: RetentionTier
    metadata: dict[str, Any] = field(default_factory=dict)
    deleted: bool = False
    deleted_at: datetime | None = None
    deletion_reason: str = ""


@dataclass
class DeletionRequest:
    """GDPR Article 17 — Right to erasure request."""

    request_id: str
    user_id: str
    tenant_id: str
    requested_at: datetime
    status: str = "pending"  # pending, in_progress, completed, failed
    categories: list[str] = field(default_factory=list)
    records_affected: int = 0
    completed_at: datetime | None = None


# ------------------------------------------------------------------
# Data Retention Enforcer
# ------------------------------------------------------------------

class DataRetentionEnforcer:
    """GDPR deletion automation with tiered retention and compliance verification.

    Features
    --------
    * Tiered retention: hot / warm / cold / archive
    * GDPR right-to-erasure automation
    * Automatic deletion after retention period
    * Compliance verification and reporting
    * Thread-safe operation
    """

    def __init__(
        self,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.audit = audit_logger or AuditLogger()
        self._policies: dict[str, RetentionPolicy] = {}
        self._records: dict[str, DataRecord] = {}
        self._deletion_requests: dict[str, DeletionRequest] = {}
        self._lock = threading.RLock()
        self._request_counter = 0
        self._setup_default_policies()

    # ------------------------------------------------------------------
    # Default retention policies
    # ------------------------------------------------------------------

    def _setup_default_policies(self) -> None:
        """Install default retention policies aligned with GDPR / SOC2."""
        defaults = [
            RetentionPolicy(
                category="memory",
                tier=RetentionTier.HOT,
                retention_days=365,
                gdpr_applies=True,
                auto_delete=False,
            ),
            RetentionPolicy(
                category="audit",
                tier=RetentionTier.ARCHIVE,
                retention_days=2555,  # 7 years for SOC2
                gdpr_applies=False,
                auto_delete=False,
            ),
            RetentionPolicy(
                category="user_profile",
                tier=RetentionTier.HOT,
                retention_days=365,
                gdpr_applies=True,
                auto_delete=True,
            ),
            RetentionPolicy(
                category="session",
                tier=RetentionTier.WARM,
                retention_days=30,
                gdpr_applies=True,
                auto_delete=True,
            ),
            RetentionPolicy(
                category="backup",
                tier=RetentionTier.COLD,
                retention_days=90,
                gdpr_applies=False,
                auto_delete=True,
            ),
            RetentionPolicy(
                category="analytics",
                tier=RetentionTier.WARM,
                retention_days=180,
                gdpr_applies=True,
                auto_delete=True,
            ),
        ]
        for policy in defaults:
            self.set_policy(policy)

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def set_policy(self, policy: RetentionPolicy) -> None:
        with self._lock:
            self._policies[policy.category] = policy

    def get_policy(self, category: str) -> RetentionPolicy | None:
        with self._lock:
            return self._policies.get(category)

    def remove_policy(self, category: str) -> bool:
        with self._lock:
            return self._policies.pop(category, None) is not None

    def list_policies(self) -> list[RetentionPolicy]:
        with self._lock:
            return list(self._policies.values())

    # ------------------------------------------------------------------
    # Record tracking
    # ------------------------------------------------------------------

    def track_record(
        self,
        record_id: str,
        category: str,
        tenant_id: str,
        user_id: str | None = None,
        created_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DataRecord:
        """Register a data record for retention tracking."""
        with self._lock:
            policy = self._policies.get(category)
            tier = policy.tier if policy else RetentionTier.HOT
            record = DataRecord(
                record_id=record_id,
                category=category,
                tenant_id=tenant_id,
                user_id=user_id,
                created_at=created_at or datetime.now(timezone.utc),
                tier=tier,
                metadata=metadata or {},
            )
            self._records[record_id] = record
            return record

    def get_record(self, record_id: str) -> DataRecord | None:
        with self._lock:
            return self._records.get(record_id)

    def list_records(
        self,
        category: str | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        tier: RetentionTier | None = None,
    ) -> list[DataRecord]:
        with self._lock:
            results = list(self._records.values())
            if category:
                results = [r for r in results if r.category == category]
            if tenant_id:
                results = [r for r in results if r.tenant_id == tenant_id]
            if user_id:
                results = [r for r in results if r.user_id == user_id]
            if tier:
                results = [r for r in results if r.tier == tier]
            return results

    # ------------------------------------------------------------------
    # Tiered retention transitions
    # ------------------------------------------------------------------

    def transition_tier(self, record_id: str, new_tier: RetentionTier) -> bool:
        """Move a record to a different retention tier."""
        with self._lock:
            record = self._records.get(record_id)
            if record is None or record.deleted:
                return False
            old_tier = record.tier
            record.tier = new_tier
            self.audit.log(
                event_type=AuditEventType.MEMORY_UPDATE,
                action="tier_transition",
                resource="retention",
                details={
                    "record_id": record_id,
                    "old_tier": old_tier.value,
                    "new_tier": new_tier.value,
                },
            )
            return True

    def auto_transition_tiers(self) -> list[tuple[str, RetentionTier, RetentionTier]]:
        """Automatically transition records based on age and policy.

        Returns list of (record_id, old_tier, new_tier) tuples.
        """
        with self._lock:
            transitions: list[tuple[str, RetentionTier, RetentionTier]] = []
            now = datetime.now(timezone.utc)
            for record in self._records.values():
                if record.deleted:
                    continue
                policy = self._policies.get(record.category)
                if policy is None:
                    continue
                age_days = (now - record.created_at).days
                old_tier = record.tier
                new_tier = old_tier

                # Hot -> Warm after 30 days
                if old_tier == RetentionTier.HOT and age_days > 30:
                    new_tier = RetentionTier.WARM
                # Warm -> Cold after 90 days
                elif old_tier == RetentionTier.WARM and age_days > 90:
                    new_tier = RetentionTier.COLD
                # Cold -> Archive after retention period
                elif old_tier == RetentionTier.COLD and policy.retention_days and age_days > policy.retention_days:
                    new_tier = RetentionTier.ARCHIVE

                if new_tier != old_tier:
                    record.tier = new_tier
                    transitions.append((record.record_id, old_tier, new_tier))
                    self.audit.log(
                        event_type=AuditEventType.MEMORY_UPDATE,
                        action="auto_tier_transition",
                        resource="retention",
                        details={
                            "record_id": record.record_id,
                            "old_tier": old_tier.value,
                            "new_tier": new_tier.value,
                            "age_days": age_days,
                        },
                    )
            return transitions

    # ------------------------------------------------------------------
    # GDPR right-to-erasure (Article 17)
    # ------------------------------------------------------------------

    def submit_erasure_request(
        self,
        user_id: str,
        tenant_id: str,
        categories: list[str] | None = None,
    ) -> DeletionRequest:
        """Submit a GDPR right-to-erasure request."""
        with self._lock:
            self._request_counter += 1
            request_id = f"GDPR-DEL-{self._request_counter:06d}"
            request = DeletionRequest(
                request_id=request_id,
                user_id=user_id,
                tenant_id=tenant_id,
                requested_at=datetime.now(timezone.utc),
                status="pending",
                categories=categories or [],
            )
            self._deletion_requests[request_id] = request
            self.audit.log(
                event_type=AuditEventType.DATA_PURGE,
                action="erasure_request",
                resource="gdpr",
                details={
                    "request_id": request_id,
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "categories": categories,
                },
            )
            return request

    def process_erasure_request(self, request_id: str) -> DeletionRequest:
        """Process a pending erasure request.

        Deletes all records for the user in applicable categories,
        respecting legal holds and non-GDPR categories.
        """
        with self._lock:
            request = self._deletion_requests.get(request_id)
            if request is None:
                raise ValueError(f"Deletion request {request_id} not found")
            if request.status != "pending":
                return request

            request.status = "in_progress"
            records_affected = 0
            now = datetime.now(timezone.utc)

            for record in list(self._records.values()):
                if record.deleted:
                    continue
                if record.user_id != request.user_id:
                    continue
                if record.tenant_id != request.tenant_id:
                    continue

                policy = self._policies.get(record.category)
                if policy is None:
                    continue

                # Skip if categories are specified and this isn't one
                if request.categories and record.category not in request.categories:
                    continue

                # Skip if legal hold
                if policy.legal_hold:
                    continue

                # Skip if GDPR doesn't apply and no explicit categories given
                if not policy.gdpr_applies and not request.categories:
                    continue

                record.deleted = True
                record.deleted_at = now
                record.deletion_reason = f"gdpr_erasure_{request_id}"
                records_affected += 1
                self.audit.log(
                    event_type=AuditEventType.DATA_PURGE,
                    action="record_deleted",
                    resource="gdpr",
                    details={
                        "record_id": record.record_id,
                        "request_id": request_id,
                        "category": record.category,
                    },
                )

            request.records_affected = records_affected
            request.status = "completed" if records_affected >= 0 else "failed"
            request.completed_at = now

            self.audit.log(
                event_type=AuditEventType.DATA_PURGE,
                action="erasure_complete",
                resource="gdpr",
                details={
                    "request_id": request_id,
                    "records_affected": records_affected,
                    "status": request.status,
                },
            )
            return request

    def get_erasure_request(self, request_id: str) -> DeletionRequest | None:
        with self._lock:
            return self._deletion_requests.get(request_id)

    def list_erasure_requests(
        self,
        user_id: str | None = None,
        status: str | None = None,
    ) -> list[DeletionRequest]:
        with self._lock:
            results = list(self._deletion_requests.values())
            if user_id:
                results = [r for r in results if r.user_id == user_id]
            if status:
                results = [r for r in results if r.status == status]
            return results

    # ------------------------------------------------------------------
    # Automatic deletion after retention period
    # ------------------------------------------------------------------

    def enforce_retention(self) -> list[DataRecord]:
        """Enforce retention policies: delete records past their retention period.

        Returns list of deleted records.
        """
        with self._lock:
            deleted: list[DataRecord] = []
            now = datetime.now(timezone.utc)
            for record in list(self._records.values()):
                if record.deleted:
                    continue
                policy = self._policies.get(record.category)
                if policy is None:
                    continue
                if not policy.auto_delete:
                    continue
                if policy.legal_hold:
                    continue
                if policy.retention_days is None:
                    continue

                age_days = (now - record.created_at).days
                if age_days > policy.retention_days:
                    record.deleted = True
                    record.deleted_at = now
                    record.deletion_reason = "retention_policy"
                    deleted.append(record)
                    self.audit.log(
                        event_type=AuditEventType.DATA_PURGE,
                        action="retention_deletion",
                        resource="retention",
                        details={
                            "record_id": record.record_id,
                            "category": record.category,
                            "age_days": age_days,
                            "retention_days": policy.retention_days,
                        },
                    )
            return deleted

    # ------------------------------------------------------------------
    # Compliance verification
    # ------------------------------------------------------------------

    def verify_compliance(self) -> dict[str, Any]:
        """Verify retention compliance across all policies and records.

        Returns a compliance report dict.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            issues: list[dict[str, Any]] = []
            compliant_records = 0
            non_compliant_records = 0

            for record in self._records.values():
                if record.deleted:
                    continue
                policy = self._policies.get(record.category)
                if policy is None:
                    issues.append({
                        "record_id": record.record_id,
                        "issue": "no_policy",
                        "message": f"No retention policy for category {record.category}",
                    })
                    non_compliant_records += 1
                    continue

                age_days = (now - record.created_at).days
                if policy.retention_days and age_days > policy.retention_days:
                    if not policy.legal_hold:
                        issues.append({
                            "record_id": record.record_id,
                            "issue": "retention_exceeded",
                            "message": (
                                f"Record age ({age_days}d) exceeds retention "
                                f"({policy.retention_days}d)"
                            ),
                            "category": record.category,
                        })
                        non_compliant_records += 1
                        continue

                compliant_records += 1

            total_active = sum(1 for r in self._records.values() if not r.deleted)
            compliance_rate = (
                round((compliant_records / total_active) * 100, 2)
                if total_active > 0 else 100.0
            )

            return {
                "verified_at": now.isoformat(),
                "total_records": len(self._records),
                "active_records": total_active,
                "deleted_records": sum(1 for r in self._records.values() if r.deleted),
                "compliant_records": compliant_records,
                "non_compliant_records": non_compliant_records,
                "compliance_rate": compliance_rate,
                "policies_defined": len(self._policies),
                "issues": issues,
                "gdpr_requests_pending": sum(
                    1 for r in self._deletion_requests.values() if r.status == "pending"
                ),
                "gdpr_requests_completed": sum(
                    1 for r in self._deletion_requests.values() if r.status == "completed"
                ),
            }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_retention_report(self) -> dict[str, Any]:
        """Export a comprehensive retention report."""
        with self._lock:
            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "policies": [
                    {
                        "category": p.category,
                        "tier": p.tier.value,
                        "retention_days": p.retention_days,
                        "legal_hold": p.legal_hold,
                        "gdpr_applies": p.gdpr_applies,
                        "auto_delete": p.auto_delete,
                    }
                    for p in self._policies.values()
                ],
                "records_by_tier": {
                    tier.value: len([
                        r for r in self._records.values()
                        if r.tier == tier and not r.deleted
                    ])
                    for tier in RetentionTier
                },
                "gdpr_requests": [
                    {
                        "request_id": r.request_id,
                        "user_id": r.user_id,
                        "status": r.status,
                        "records_affected": r.records_affected,
                        "requested_at": r.requested_at.isoformat(),
                        "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                    }
                    for r in self._deletion_requests.values()
                ],
                "compliance_summary": self.verify_compliance(),
            }

    def to_json(self) -> str:
        """Export retention report as JSON."""
        return json.dumps(self.export_retention_report(), indent=2, default=str)
