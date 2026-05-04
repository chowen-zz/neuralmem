"""SOC 2 Evidence Collector — automated control testing, evidence snapshots,
timestamp chains, and auditor-ready PDF report generation.

NeuralMem Enterprise V1.7 compliance certification automation.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from neuralmem.enterprise.audit import AuditEventType, AuditLogger, ComplianceStandard


# ------------------------------------------------------------------
# Control test schedule
# ------------------------------------------------------------------

class ControlFrequency(str, Enum):
    """Frequency for automated control tests."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


# ------------------------------------------------------------------
# Evidence snapshot with cryptographic timestamp chain
# ------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceSnapshot:
    """Immutable evidence snapshot with hash chain integrity.

    Each snapshot links to the previous snapshot's hash, forming a
    tamper-evident chain suitable for auditor review.
    """

    id: str
    control_id: str
    control_name: str
    timestamp: datetime
    test_result: bool
    details: dict[str, Any]
    previous_hash: str
    snapshot_hash: str
    standard: ComplianceStandard


@dataclass
class ControlDefinition:
    """Definition of a SOC 2 / compliance control."""

    control_id: str
    name: str
    description: str
    standard: ComplianceStandard
    frequency: ControlFrequency
    category: str  # e.g. "security", "availability", "confidentiality"
    test_fn_name: str  # name of the test method to run
    auto_remediate: bool = False


@dataclass
class ControlTestResult:
    """Result of a single control test execution."""

    control_id: str
    control_name: str
    passed: bool
    timestamp: datetime
    evidence: EvidenceSnapshot
    remediation_suggestion: str = ""
    severity: str = "info"  # info, warning, critical


# ------------------------------------------------------------------
# SOC2 Evidence Collector
# ------------------------------------------------------------------

class SOC2EvidenceCollector:
    """Automated SOC 2 evidence collection with cryptographic timestamp chains.

    Features
    --------
    * Daily / weekly automated control tests
    * Evidence snapshots with SHA-256 hash chains
    * Auditor-ready PDF report generation (mock-based)
    * Thread-safe operation
    """

    def __init__(
        self,
        audit_logger: AuditLogger | None = None,
        compliance_standard: ComplianceStandard = ComplianceStandard.SOC2,
    ) -> None:
        self.audit = audit_logger or AuditLogger()
        self.standard = compliance_standard
        self._controls: list[ControlDefinition] = []
        self._snapshots: list[EvidenceSnapshot] = []
        self._test_results: list[ControlTestResult] = []
        self._lock = threading.RLock()
        self._last_hash: str = "0" * 64
        self._setup_default_controls()

    # ------------------------------------------------------------------
    # Default SOC 2 controls
    # ------------------------------------------------------------------

    def _setup_default_controls(self) -> None:
        """Install default SOC 2 Type II controls."""
        defaults = [
            ControlDefinition(
                control_id="CC6.1",
                name="Logical Access Security",
                description="Verify access controls enforce logical separation.",
                standard=self.standard,
                frequency=ControlFrequency.DAILY,
                category="security",
                test_fn_name="_test_logical_access",
            ),
            ControlDefinition(
                control_id="CC6.2",
                name="Access Removal",
                description="Verify terminated user access is revoked within 24h.",
                standard=self.standard,
                frequency=ControlFrequency.DAILY,
                category="security",
                test_fn_name="_test_access_removal",
            ),
            ControlDefinition(
                control_id="CC7.1",
                name="System Operations Monitoring",
                description="Verify system monitoring detects anomalies.",
                standard=self.standard,
                frequency=ControlFrequency.DAILY,
                category="availability",
                test_fn_name="_test_system_monitoring",
            ),
            ControlDefinition(
                control_id="CC7.2",
                name="Security Incident Detection",
                description="Verify security incidents are detected and logged.",
                standard=self.standard,
                frequency=ControlFrequency.DAILY,
                category="security",
                test_fn_name="_test_incident_detection",
            ),
            ControlDefinition(
                control_id="CC8.1",
                name="Change Management",
                description="Verify changes are authorized and documented.",
                standard=self.standard,
                frequency=ControlFrequency.WEEKLY,
                category="security",
                test_fn_name="_test_change_management",
            ),
            ControlDefinition(
                control_id="A1.1",
                name="Data Backup",
                description="Verify backups are performed and restorable.",
                standard=self.standard,
                frequency=ControlFrequency.WEEKLY,
                category="availability",
                test_fn_name="_test_data_backup",
            ),
            ControlDefinition(
                control_id="A1.2",
                name="Disaster Recovery",
                description="Verify DR plan is tested and documented.",
                standard=self.standard,
                frequency=ControlFrequency.WEEKLY,
                category="availability",
                test_fn_name="_test_disaster_recovery",
            ),
            ControlDefinition(
                control_id="C1.1",
                name="Data Encryption at Rest",
                description="Verify sensitive data is encrypted at rest.",
                standard=self.standard,
                frequency=ControlFrequency.DAILY,
                category="confidentiality",
                test_fn_name="_test_encryption_at_rest",
            ),
            ControlDefinition(
                control_id="C1.2",
                name="Data Encryption in Transit",
                description="Verify data is encrypted in transit.",
                standard=self.standard,
                frequency=ControlFrequency.DAILY,
                category="confidentiality",
                test_fn_name="_test_encryption_in_transit",
            ),
        ]
        for ctrl in defaults:
            self.add_control(ctrl)

    # ------------------------------------------------------------------
    # Control management
    # ------------------------------------------------------------------

    def add_control(self, control: ControlDefinition) -> None:
        with self._lock:
            self._controls.append(control)

    def remove_control(self, control_id: str) -> bool:
        with self._lock:
            original = len(self._controls)
            self._controls = [c for c in self._controls if c.control_id != control_id]
            return len(self._controls) < original

    def list_controls(
        self,
        category: str | None = None,
        frequency: ControlFrequency | None = None,
    ) -> list[ControlDefinition]:
        with self._lock:
            results = list(self._controls)
            if category:
                results = [c for c in results if c.category == category]
            if frequency:
                results = [c for c in results if c.frequency == frequency]
            return results

    # ------------------------------------------------------------------
    # Built-in control tests (mock-based, overridable)
    # ------------------------------------------------------------------

    def _test_logical_access(self) -> tuple[bool, dict[str, Any], str]:
        """Mock test: verify logical access controls."""
        passed = True
        details = {"access_policies_active": True, "rbac_enforced": True}
        return passed, details, ""

    def _test_access_removal(self) -> tuple[bool, dict[str, Any], str]:
        """Mock test: verify terminated user access removal."""
        passed = True
        details = {"terminated_users": 0, "orphan_accounts": 0}
        return passed, details, ""

    def _test_system_monitoring(self) -> tuple[bool, dict[str, Any], str]:
        """Mock test: verify system monitoring."""
        passed = True
        details = {"monitoring_active": True, "alert_rules": 12}
        return passed, details, ""

    def _test_incident_detection(self) -> tuple[bool, dict[str, Any], str]:
        """Mock test: verify incident detection."""
        passed = True
        details = {"incidents_last_24h": 0, "siem_connected": True}
        return passed, details, ""

    def _test_change_management(self) -> tuple[bool, dict[str, Any], str]:
        """Mock test: verify change management."""
        passed = True
        details = {"pending_changes": 2, "approved_changes": 5}
        return passed, details, ""

    def _test_data_backup(self) -> tuple[bool, dict[str, Any], str]:
        """Mock test: verify data backup."""
        passed = True
        details = {"last_backup": datetime.now(timezone.utc).isoformat(), "restorable": True}
        return passed, details, ""

    def _test_disaster_recovery(self) -> tuple[bool, dict[str, Any], str]:
        """Mock test: verify disaster recovery."""
        passed = True
        details = {"dr_plan_documented": True, "last_dr_test": "2024-01-15"}
        return passed, details, ""

    def _test_encryption_at_rest(self) -> tuple[bool, dict[str, Any], str]:
        """Mock test: verify encryption at rest."""
        passed = True
        details = {"encryption_algorithm": "AES-256-GCM", "key_rotation_days": 90}
        return passed, details, ""

    def _test_encryption_in_transit(self) -> tuple[bool, dict[str, Any], str]:
        """Mock test: verify encryption in transit."""
        passed = True
        details = {"tls_version": "1.3", "cipher_suites": ["TLS_AES_256_GCM_SHA384"]}
        return passed, details, ""

    # ------------------------------------------------------------------
    # Evidence snapshot / hash chain
    # ------------------------------------------------------------------

    def _compute_hash(self, data: dict[str, Any], previous_hash: str) -> str:
        """Compute SHA-256 hash of snapshot data + previous hash."""
        payload = json.dumps(data, sort_keys=True, default=str) + previous_hash
        return hashlib.sha256(payload.encode()).hexdigest()

    def create_snapshot(
        self,
        control_id: str,
        control_name: str,
        test_result: bool,
        details: dict[str, Any],
    ) -> EvidenceSnapshot:
        """Create an immutable evidence snapshot with hash chain."""
        with self._lock:
            snapshot_id = f"EVID-{len(self._snapshots)+1:06d}"
            timestamp = datetime.now(timezone.utc)
            data = {
                "id": snapshot_id,
                "control_id": control_id,
                "control_name": control_name,
                "timestamp": timestamp.isoformat(),
                "test_result": test_result,
                "details": details,
                "previous_hash": self._last_hash,
            }
            snapshot_hash = self._compute_hash(data, self._last_hash)
            snapshot = EvidenceSnapshot(
                id=snapshot_id,
                control_id=control_id,
                control_name=control_name,
                timestamp=timestamp,
                test_result=test_result,
                details=details,
                previous_hash=self._last_hash,
                snapshot_hash=snapshot_hash,
                standard=self.standard,
            )
            self._snapshots.append(snapshot)
            self._last_hash = snapshot_hash
            return snapshot

    def verify_chain(self) -> list[tuple[str, bool, str]]:
        """Verify the integrity of the entire evidence chain.

        Returns a list of (snapshot_id, is_valid, message) tuples.
        """
        with self._lock:
            results: list[tuple[str, bool, str]] = []
            prev_hash = "0" * 64
            for snap in self._snapshots:
                data = {
                    "id": snap.id,
                    "control_id": snap.control_id,
                    "control_name": snap.control_name,
                    "timestamp": snap.timestamp.isoformat(),
                    "test_result": snap.test_result,
                    "details": snap.details,
                    "previous_hash": prev_hash,
                }
                expected = self._compute_hash(data, prev_hash)
                if snap.snapshot_hash != expected:
                    results.append((snap.id, False, f"hash mismatch: expected {expected}, got {snap.snapshot_hash}"))
                elif snap.previous_hash != prev_hash:
                    results.append((snap.id, False, f"chain break: previous hash mismatch"))
                else:
                    results.append((snap.id, True, "valid"))
                prev_hash = snap.snapshot_hash
            return results

    def get_snapshots(
        self,
        control_id: str | None = None,
        since: datetime | None = None,
    ) -> list[EvidenceSnapshot]:
        with self._lock:
            results = list(self._snapshots)
            if control_id:
                results = [s for s in results if s.control_id == control_id]
            if since:
                results = [s for s in results if s.timestamp >= since]
            return results

    # ------------------------------------------------------------------
    # Automated control testing
    # ------------------------------------------------------------------

    def run_control_test(self, control: ControlDefinition) -> ControlTestResult:
        """Execute a single control test and capture evidence."""
        test_fn = getattr(self, control.test_fn_name, None)
        if test_fn is None:
            passed = False
            details = {"error": f"test function {control.test_fn_name} not found"}
            suggestion = f"Implement test method {control.test_fn_name}"
            severity = "critical"
        else:
            passed, details, suggestion = test_fn()
            severity = "info" if passed else "warning"

        snapshot = self.create_snapshot(
            control_id=control.control_id,
            control_name=control.name,
            test_result=passed,
            details=details,
        )

        result = ControlTestResult(
            control_id=control.control_id,
            control_name=control.name,
            passed=passed,
            timestamp=snapshot.timestamp,
            evidence=snapshot,
            remediation_suggestion=suggestion,
            severity=severity,
        )
        with self._lock:
            self._test_results.append(result)

        self.audit.log(
            event_type=AuditEventType.MEMORY_READ,
            action="control_test",
            resource="compliance",
            details={
                "control_id": control.control_id,
                "passed": passed,
                "snapshot_id": snapshot.id,
            },
        )
        return result

    def run_all_tests(self) -> list[ControlTestResult]:
        """Run all control tests and return results."""
        with self._lock:
            controls = list(self._controls)
        results: list[ControlTestResult] = []
        for control in controls:
            results.append(self.run_control_test(control))
        return results

    def run_tests_by_frequency(self, frequency: ControlFrequency) -> list[ControlTestResult]:
        """Run tests filtered by frequency (daily / weekly / etc)."""
        controls = self.list_controls(frequency=frequency)
        return [self.run_control_test(c) for c in controls]

    def run_daily_tests(self) -> list[ControlTestResult]:
        """Run all daily control tests."""
        return self.run_tests_by_frequency(ControlFrequency.DAILY)

    def run_weekly_tests(self) -> list[ControlTestResult]:
        """Run all weekly control tests."""
        return self.run_tests_by_frequency(ControlFrequency.WEEKLY)

    # ------------------------------------------------------------------
    # Compliance score
    # ------------------------------------------------------------------

    def get_compliance_score(self) -> float:
        """Return compliance score (0-100) based on passed tests."""
        with self._lock:
            if not self._test_results:
                return 0.0
            passed = sum(1 for r in self._test_results if r.passed)
            return round((passed / len(self._test_results)) * 100, 2)

    def get_control_summary(self) -> dict[str, Any]:
        """Return summary of all controls and their latest test results."""
        with self._lock:
            summary: dict[str, Any] = {
                "total_controls": len(self._controls),
                "total_tests_run": len(self._test_results),
                "compliance_score": self.get_compliance_score(),
                "standard": self.standard.value,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "controls": [],
            }
            for control in self._controls:
                latest = None
                for r in reversed(self._test_results):
                    if r.control_id == control.control_id:
                        latest = r
                        break
                summary["controls"].append({
                    "control_id": control.control_id,
                    "name": control.name,
                    "category": control.category,
                    "frequency": control.frequency.value,
                    "latest_result": {
                        "passed": latest.passed if latest else None,
                        "timestamp": latest.timestamp.isoformat() if latest else None,
                        "snapshot_id": latest.evidence.id if latest else None,
                    },
                })
            return summary

    # ------------------------------------------------------------------
    # Auditor-ready PDF report generation (mock-based)
    # ------------------------------------------------------------------

    def generate_pdf_report(
        self,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate an auditor-ready PDF report structure (mock-based).

        Returns a dict with report metadata and content.  Actual PDF
        generation is mocked — no real PDF library required.
        """
        start = start_date or (datetime.now(timezone.utc) - timedelta(days=30))
        end = end_date or datetime.now(timezone.utc)

        with self._lock:
            snapshots_in_range = [
                s for s in self._snapshots
                if start <= s.timestamp <= end
            ]
            results_in_range = [
                r for r in self._test_results
                if start <= r.timestamp <= end
            ]

        passed = sum(1 for r in results_in_range if r.passed)
        failed = len(results_in_range) - passed
        chain_verification = self.verify_chain()
        chain_valid = all(v for _, v, _ in chain_verification)

        report = {
            "report_type": "SOC2_Evidence_Report",
            "standard": self.standard.value,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period": {
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "tenant_id": tenant_id,
            "executive_summary": {
                "total_controls_tested": len({r.control_id for r in results_in_range}),
                "total_tests_run": len(results_in_range),
                "tests_passed": passed,
                "tests_failed": failed,
                "compliance_score": self.get_compliance_score(),
                "evidence_chain_valid": chain_valid,
            },
            "evidence_chain": {
                "total_snapshots": len(snapshots_in_range),
                "chain_integrity": "VALID" if chain_valid else "COMPROMISED",
                "verification_results": [
                    {"snapshot_id": sid, "valid": valid, "message": msg}
                    for sid, valid, msg in chain_verification
                ],
            },
            "control_details": [
                {
                    "control_id": r.control_id,
                    "control_name": r.control_name,
                    "passed": r.passed,
                    "timestamp": r.timestamp.isoformat(),
                    "snapshot_id": r.evidence.id,
                    "snapshot_hash": r.evidence.snapshot_hash,
                    "details": r.evidence.details,
                    "remediation": r.remediation_suggestion,
                    "severity": r.severity,
                }
                for r in results_in_range
            ],
            "audit_events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type.value,
                    "details": e.details,
                }
                for e in self.audit.query(since=start, until=end, limit=1000)
            ],
            "pdf_metadata": {
                "title": f"SOC 2 Evidence Report — {self.standard.value.upper()}",
                "author": "NeuralMem Compliance Engine",
                "pages": len(results_in_range) + 3,
                "mock_pdf": True,
            },
        }
        return report

    def export_report_json(self, report: dict[str, Any]) -> str:
        """Export a report dict to a JSON string."""
        return json.dumps(report, indent=2, default=str)
