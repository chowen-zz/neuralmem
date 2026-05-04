"""NeuralMem Enterprise Compliance V1.4 — SOC 2 / HIPAA / GDPR framework.

Data encryption at rest, access control policies, risk assessment,
compliance report generation, and audit trail integration.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Callable

from neuralmem.enterprise.audit import AuditEventType, AuditLogger, ComplianceStandard


# ------------------------------------------------------------------
# Data encryption (AES-256-GCM via cryptography)
# ------------------------------------------------------------------

class DataEncryption:
    """AES-256-GCM encryption for sensitive fields at rest.

    Uses the ``cryptography`` library if available; otherwise falls back
    to a no-op pass-through (development / test mode).
    """

    def __init__(self, master_key: bytes | None = None) -> None:
        self._master_key = master_key or self._generate_key()
        self._lock = threading.RLock()

    @staticmethod
    def _generate_key() -> bytes:
        return secrets.token_bytes(32)

    def _get_backend(self) -> Any:
        cls = self._aesgcm_class()
        if cls is None:
            return None
        try:
            return cls(self._master_key)
        except Exception:  # pragma: no cover
            return None

    @staticmethod
    def _aesgcm_class() -> Any:
        """Return the AESGCM class (or None) for mocking in tests."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            return AESGCM
        except Exception:
            return None

    def encrypt(self, plaintext: str | bytes) -> dict[str, str]:
        """Encrypt *plaintext* and return a dict with ciphertext, nonce, tag."""
        data = plaintext.encode() if isinstance(plaintext, str) else plaintext
        nonce = secrets.token_bytes(12)
        backend = self._get_backend()
        with self._lock:
            if backend is None:
                # Fallback: base64 encode (not secure — dev/test only)
                import base64
                return {
                    "ciphertext": base64.b64encode(data).decode(),
                    "nonce": base64.b64encode(nonce).decode(),
                    "tag": "",
                    "mode": "fallback",
                }
            ciphertext = backend.encrypt(nonce, data, None)
            return {
                "ciphertext": ciphertext.hex(),
                "nonce": nonce.hex(),
                "tag": "",
                "mode": "aes-256-gcm",
            }

    def decrypt(self, encrypted: dict[str, str]) -> str:
        """Decrypt an encrypted dict back to a UTF-8 string."""
        with self._lock:
            if encrypted.get("mode") == "fallback":
                import base64
                return base64.b64decode(encrypted["ciphertext"]).decode()
            backend = self._get_backend()
            if backend is None:
                raise RuntimeError("cryptography backend unavailable")
            ciphertext = bytes.fromhex(encrypted["ciphertext"])
            nonce = bytes.fromhex(encrypted["nonce"])
            plaintext = backend.decrypt(nonce, ciphertext, None)
            return plaintext.decode()

    def _encrypt_for_test(self, plaintext: str | bytes) -> dict[str, str]:
        """Test-only encrypt that always uses fallback mode (no real crypto)."""
        data = plaintext.encode() if isinstance(plaintext, str) else plaintext
        nonce = secrets.token_bytes(12)
        import base64
        return {
            "ciphertext": base64.b64encode(data).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "tag": "",
            "mode": "fallback",
        }

    def _decrypt_for_test(self, encrypted: dict[str, str]) -> str:
        """Test-only decrypt that handles fallback mode."""
        if encrypted.get("mode") == "fallback":
            import base64
            return base64.b64decode(encrypted["ciphertext"]).decode()
        raise RuntimeError("test decrypt only supports fallback mode")

    def encrypt_field(self, field_name: str, value: Any) -> dict[str, str]:
        """Encrypt a single field and tag it with the field name."""
        payload = json.dumps({"field": field_name, "value": value})
        result = self.encrypt(payload)
        result["field"] = field_name
        return result

    def decrypt_field(self, encrypted: dict[str, str]) -> Any:
        """Decrypt a field-encrypted dict and return the original value."""
        payload = self.decrypt(encrypted)
        obj = json.loads(payload)
        return obj["value"]


# ------------------------------------------------------------------
# Access control policy engine
# ------------------------------------------------------------------

class PolicyDecision(IntEnum):
    """Result of a policy evaluation."""

    ALLOW = 1
    DENY = 0
    DEFER = -1


@dataclass
class AccessControlPolicy:
    """A single access control rule.

    Attributes
    ----------
    name
        Human-readable rule name.
    resource_type
        e.g. ``memory``, ``tenant``, ``audit``.
    action
        e.g. ``read``, ``write``, ``delete``.
    condition
        Callable ``(context) -> bool`` that decides if the rule applies.
    effect
        ``ALLOW`` or ``DENY``.
    standards
        Compliance standards this rule enforces.
    """

    name: str
    resource_type: str
    action: str
    condition: Callable[[dict[str, Any]], bool]
    effect: PolicyDecision = PolicyDecision.ALLOW
    standards: tuple[ComplianceStandard, ...] = field(default_factory=tuple)
    priority: int = 0


class PolicyEngine:
    """Evaluate ordered access-control policies for a request context."""

    def __init__(self) -> None:
        self._policies: list[AccessControlPolicy] = []
        self._lock = threading.RLock()

    def add_policy(self, policy: AccessControlPolicy) -> None:
        with self._lock:
            self._policies.append(policy)
            self._policies.sort(key=lambda p: p.priority, reverse=True)

    def remove_policy(self, name: str) -> bool:
        with self._lock:
            original = len(self._policies)
            self._policies = [p for p in self._policies if p.name != name]
            return len(self._policies) < original

    def evaluate(
        self,
        resource_type: str,
        action: str,
        context: dict[str, Any],
    ) -> PolicyDecision:
        """Evaluate all matching policies and return the decisive result.

        DENY overrides ALLOW.  If no policy matches, returns DEFER.
        """
        with self._lock:
            matched = [
                p for p in self._policies
                if p.resource_type == resource_type
                and p.action == action
                and p.condition(context)
            ]
            if not matched:
                return PolicyDecision.DEFER
            for policy in matched:
                if policy.effect == PolicyDecision.DENY:
                    return PolicyDecision.DENY
            return PolicyDecision.ALLOW

    def list_policies(
        self,
        resource_type: str | None = None,
        standard: ComplianceStandard | None = None,
    ) -> list[AccessControlPolicy]:
        with self._lock:
            results = list(self._policies)
            if resource_type:
                results = [p for p in results if p.resource_type == resource_type]
            if standard:
                results = [p for p in results if standard in p.standards]
            return results


# ------------------------------------------------------------------
# Risk assessment
# ------------------------------------------------------------------

class RiskLevel(IntEnum):
    """Risk severity levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RiskFinding:
    """A single risk assessment finding."""

    id: str
    category: str
    description: str
    level: RiskLevel
    standards: tuple[ComplianceStandard, ...]
    remediation: str = ""
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RiskAssessor:
    """Score and track compliance risks.

    Provides built-in checks for common SOC2 / HIPAA / GDPR / ISO27001 gaps.
    """

    def __init__(self) -> None:
        self._findings: list[RiskFinding] = []
        self._lock = threading.RLock()
        self._counter = 0

    def _next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"RISK-{self._counter:04d}"

    def add_finding(
        self,
        category: str,
        description: str,
        level: RiskLevel,
        standards: tuple[ComplianceStandard, ...],
        remediation: str = "",
    ) -> RiskFinding:
        finding = RiskFinding(
            id=self._next_id(),
            category=category,
            description=description,
            level=level,
            standards=standards,
            remediation=remediation,
        )
        with self._lock:
            self._findings.append(finding)
        return finding

    def assess_encryption_at_rest(
        self,
        encryption_enabled: bool,
        standards: tuple[ComplianceStandard, ...] = (
            ComplianceStandard.SOC2,
            ComplianceStandard.HIPAA,
            ComplianceStandard.GDPR,
            ComplianceStandard.ISO27001,
        ),
    ) -> RiskFinding | None:
        if not encryption_enabled:
            return self.add_finding(
                category="encryption",
                description="Data encryption at rest is not enabled.",
                level=RiskLevel.CRITICAL,
                standards=standards,
                remediation="Enable AES-256-GCM encryption for all sensitive fields.",
            )
        return None

    def assess_access_logging(
        self,
        audit_logger: AuditLogger | None,
        standards: tuple[ComplianceStandard, ...] = (
            ComplianceStandard.SOC2,
            ComplianceStandard.HIPAA,
            ComplianceStandard.ISO27001,
        ),
    ) -> RiskFinding | None:
        if audit_logger is None or audit_logger.count == 0:
            return self.add_finding(
                category="logging",
                description="No audit events have been recorded.",
                level=RiskLevel.HIGH,
                standards=standards,
                remediation="Configure AuditLogger and ensure all access is logged.",
            )
        return None

    def assess_mfa_policy(
        self,
        mfa_required: bool,
        standards: tuple[ComplianceStandard, ...] = (
            ComplianceStandard.SOC2,
            ComplianceStandard.ISO27001,
        ),
    ) -> RiskFinding | None:
        if not mfa_required:
            return self.add_finding(
                category="authentication",
                description="Multi-factor authentication is not enforced.",
                level=RiskLevel.MEDIUM,
                standards=standards,
                remediation="Require MFA for all privileged accounts.",
            )
        return None

    def assess_data_retention(
        self,
        retention_days: int | None,
        standards: tuple[ComplianceStandard, ...] = (
            ComplianceStandard.GDPR,
            ComplianceStandard.HIPAA,
        ),
    ) -> RiskFinding | None:
        if retention_days is None or retention_days > 2555:  # > 7 years
            return self.add_finding(
                category="retention",
                description="Data retention policy exceeds recommended limits or is undefined.",
                level=RiskLevel.MEDIUM,
                standards=standards,
                remediation="Define and enforce a data retention schedule (GDPR Art. 5).",
            )
        return None

    def get_findings(
        self,
        level: RiskLevel | None = None,
        standard: ComplianceStandard | None = None,
    ) -> list[RiskFinding]:
        with self._lock:
            results = list(self._findings)
            if level is not None:
                results = [f for f in results if f.level == level]
            if standard is not None:
                results = [f for f in results if standard in f.standards]
            return results

    def get_risk_score(self) -> int:
        """Return aggregate risk score (sum of level values)."""
        with self._lock:
            return sum(f.level.value for f in self._findings)

    def clear(self) -> None:
        with self._lock:
            self._findings.clear()
            self._counter = 0


# ------------------------------------------------------------------
# Compliance report generation
# ------------------------------------------------------------------

@dataclass
class ComplianceReport:
    """Generated compliance report."""

    standard: ComplianceStandard
    generated_at: datetime
    findings: list[RiskFinding]
    audit_event_count: int
    risk_score: int
    policy_count: int
    encryption_enabled: bool
    summary: dict[str, Any] = field(default_factory=dict)


class ComplianceReportGenerator:
    """Generate structured compliance reports for auditors."""

    def __init__(
        self,
        audit_logger: AuditLogger,
        risk_assessor: RiskAssessor,
        policy_engine: PolicyEngine,
        encryption: DataEncryption,
    ) -> None:
        self._audit = audit_logger
        self._risk = risk_assessor
        self._policy = policy_engine
        self._encryption = encryption

    def generate(
        self,
        standard: ComplianceStandard,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> ComplianceReport:
        findings = self._risk.get_findings(standard=standard)
        audit_events = self._audit.export_for_compliance(
            standard, tenant_id=tenant_id, user_id=user_id
        )
        policies = self._policy.list_policies(standard=standard)
        encryption_enabled = self._encryption._get_backend() is not None

        summary: dict[str, Any] = {
            "standard": standard.value,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "total_findings": len(findings),
            "critical_findings": sum(1 for f in findings if f.level == RiskLevel.CRITICAL),
            "high_findings": sum(1 for f in findings if f.level == RiskLevel.HIGH),
            "audit_events_exported": len(audit_events),
            "policies_evaluated": len(policies),
            "encryption_at_rest": encryption_enabled,
        }

        return ComplianceReport(
            standard=standard,
            generated_at=datetime.now(timezone.utc),
            findings=findings,
            audit_event_count=len(audit_events),
            risk_score=self._risk.get_risk_score(),
            policy_count=len(policies),
            encryption_enabled=encryption_enabled,
            summary=summary,
        )

    def generate_all(self) -> dict[ComplianceStandard, ComplianceReport]:
        """Generate reports for all supported standards."""
        return {
            standard: self.generate(standard)
            for standard in ComplianceStandard
        }

    def to_dict(self, report: ComplianceReport) -> dict[str, Any]:
        return {
            "standard": report.standard.value,
            "generated_at": report.generated_at.isoformat(),
            "risk_score": report.risk_score,
            "audit_event_count": report.audit_event_count,
            "policy_count": report.policy_count,
            "encryption_enabled": report.encryption_enabled,
            "summary": report.summary,
            "findings": [
                {
                    "id": f.id,
                    "category": f.category,
                    "description": f.description,
                    "level": f.level.name,
                    "standards": [s.value for s in f.standards],
                    "remediation": f.remediation,
                    "detected_at": f.detected_at.isoformat(),
                }
                for f in report.findings
            ],
        }


# ------------------------------------------------------------------
# ComplianceManager — unified facade
# ------------------------------------------------------------------

class ComplianceManager:
    """Unified enterprise compliance manager.

    Orchestrates encryption, access policies, risk assessment, reporting,
    and audit logging for SOC2, HIPAA, GDPR, and ISO27001.
    """

    def __init__(
        self,
        audit_logger: AuditLogger | None = None,
        master_key: bytes | None = None,
    ) -> None:
        self.audit = audit_logger or AuditLogger()
        self.encryption = DataEncryption(master_key=master_key)
        self.policies = PolicyEngine()
        self.risk = RiskAssessor()
        self.reports = ComplianceReportGenerator(
            audit_logger=self.audit,
            risk_assessor=self.risk,
            policy_engine=self.policies,
            encryption=self.encryption,
        )
        self._setup_default_policies()

    def _setup_default_policies(self) -> None:
        """Install default compliance policies."""
        # SOC2 / ISO27001: deny access outside business hours for admin actions
        def _outside_business_hours(ctx: dict[str, Any]) -> bool:
            hour = ctx.get("hour", 12)
            return 0 <= hour < 6 or 22 <= hour <= 23

        self.policies.add_policy(
            AccessControlPolicy(
                name="soc2_admin_hours",
                resource_type="memory",
                action="delete",
                condition=_outside_business_hours,
                effect=PolicyDecision.DENY,
                standards=(ComplianceStandard.SOC2, ComplianceStandard.ISO27001),
                priority=10,
            )
        )

        # GDPR: deny if data subject has withdrawn consent
        def _consent_withdrawn(ctx: dict[str, Any]) -> bool:
            return ctx.get("consent", True) is False

        self.policies.add_policy(
            AccessControlPolicy(
                name="gdpr_consent_withdrawn",
                resource_type="memory",
                action="read",
                condition=_consent_withdrawn,
                effect=PolicyDecision.DENY,
                standards=(ComplianceStandard.GDPR,),
                priority=100,
            )
        )

        # HIPAA: deny if not authenticated
        def _not_authenticated(ctx: dict[str, Any]) -> bool:
            return ctx.get("authenticated", True) is False

        self.policies.add_policy(
            AccessControlPolicy(
                name="hipaa_auth_required",
                resource_type="memory",
                action="read",
                condition=_not_authenticated,
                effect=PolicyDecision.DENY,
                standards=(ComplianceStandard.HIPAA,),
                priority=100,
            )
        )

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def encrypt_sensitive(self, data: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
        """Encrypt specified fields in a data dict."""
        result = dict(data)
        for f in fields:
            if f in result:
                result[f] = self.encryption.encrypt_field(f, result[f])
        self.audit.log(
            AuditEventType.MEMORY_UPDATE,
            action="encrypt_fields",
            resource="memory",
            details={"fields": list(fields)},
        )
        return result

    def decrypt_sensitive(self, data: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
        """Decrypt specified fields in a data dict."""
        result = dict(data)
        for f in fields:
            if f in result and isinstance(result[f], dict):
                result[f] = self.encryption.decrypt_field(result[f])
        return result

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def check_access(
        self,
        resource_type: str,
        action: str,
        context: dict[str, Any],
    ) -> bool:
        decision = self.policies.evaluate(resource_type, action, context)
        if decision == PolicyDecision.DENY:
            self.audit.log_access_denied(
                tenant_id=context.get("tenant_id"),
                user_id=context.get("user_id"),
                action=action,
                resource=resource_type,
                reason="policy_denied",
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------

    def run_risk_assessment(
        self,
        encryption_enabled: bool | None = None,
        mfa_required: bool = False,
        retention_days: int | None = None,
    ) -> list[RiskFinding]:
        if encryption_enabled is None:
            encryption_enabled = self.encryption._get_backend() is not None
        self.risk.assess_encryption_at_rest(encryption_enabled)
        self.risk.assess_access_logging(self.audit)
        self.risk.assess_mfa_policy(mfa_required)
        self.risk.assess_data_retention(retention_days)
        return self.risk.get_findings()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(
        self,
        standard: ComplianceStandard,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> ComplianceReport:
        return self.reports.generate(standard, tenant_id=tenant_id, user_id=user_id)

    def generate_all_reports(self) -> dict[ComplianceStandard, ComplianceReport]:
        return self.reports.generate_all()

    def export_report(self, report: ComplianceReport) -> dict[str, Any]:
        return self.reports.to_dict(report)
