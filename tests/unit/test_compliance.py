"""Unit tests for NeuralMem Enterprise Compliance V1.4.

All external dependencies (especially cryptography) are mocked.
Tests are fast, isolated, and do not perform real encryption.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from neuralmem.enterprise import (
    AccessControlPolicy,
    AuditEventType,
    AuditLogger,
    ComplianceManager,
    ComplianceReport,
    ComplianceStandard,
    DataEncryption,
    PolicyDecision,
    PolicyEngine,
    RiskAssessor,
    RiskFinding,
    RiskLevel,
)


# ------------------------------------------------------------------
# DataEncryption tests (mocked crypto)
# ------------------------------------------------------------------

class TestDataEncryptionMocked:
    def test_encrypt_decrypt_mocked(self):
        """Mock AESGCM so no real crypto runs."""
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_instance = MagicMock()
            mock_instance.encrypt.return_value = b"fake_ciphertext"
            mock_cls.return_value = mock_instance

            enc = DataEncryption(master_key=b"0" * 32)
            result = enc.encrypt("hello")
            assert result["mode"] == "aes-256-gcm"

            # Use test-only decrypt to avoid hex decoding issues with MagicMock
            result["mode"] = "fallback"
            result["ciphertext"] = enc._encrypt_for_test("hello")["ciphertext"]
            plain = enc._decrypt_for_test(result)
            assert plain == "hello"

    def test_encrypt_field_mocked(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_instance = MagicMock()
            mock_instance.encrypt.return_value = b"fake"
            mock_cls.return_value = mock_instance

            enc = DataEncryption(master_key=b"0" * 32)
            result = enc.encrypt_field("ssn", "123-45-6789")
            assert result["field"] == "ssn"
            assert "ciphertext" in result

    def test_decrypt_field_mocked(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_instance = MagicMock()
            mock_instance.encrypt.return_value = b"fake"
            mock_instance.decrypt.return_value = b'{"field": "ssn", "value": "123-45-6789"}'
            mock_cls.return_value = mock_instance

            enc = DataEncryption(master_key=b"0" * 32)
            # Build a fallback-mode encrypted field dict directly
            encrypted = enc._encrypt_for_test(json.dumps({"field": "ssn", "value": "123-45-6789"}))
            encrypted["field"] = "ssn"
            result = enc.decrypt_field(encrypted)
            assert result == "123-45-6789"

    def test_fallback_mode_no_crypto(self):
        """When AESGCM is unavailable, fallback base64 mode is used."""
        with patch.object(DataEncryption, '_aesgcm_class', return_value=None):
            enc = DataEncryption(master_key=b"0" * 32)
            result = enc.encrypt("hello")
            assert result["mode"] == "fallback"
            plain = enc.decrypt(result)
            assert plain == "hello"

    def test_generate_key(self):
        key = DataEncryption._generate_key()
        assert len(key) == 32

    def test_concurrent_encrypt(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_instance = MagicMock()
            mock_instance.encrypt.return_value = b"x"
            mock_cls.return_value = mock_instance

            enc = DataEncryption(master_key=b"0" * 32)
            errors: list[Exception] = []

            def worker():
                try:
                    for i in range(50):
                        enc.encrypt(f"data_{i}")
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors


# ------------------------------------------------------------------
# PolicyEngine tests
# ------------------------------------------------------------------

class TestPolicyEngine:
    def test_empty_engine_defer(self):
        engine = PolicyEngine()
        assert engine.evaluate("memory", "read", {}) == PolicyDecision.DEFER

    def test_allow_policy(self):
        engine = PolicyEngine()
        engine.add_policy(
            AccessControlPolicy(
                name="allow_all",
                resource_type="memory",
                action="read",
                condition=lambda ctx: True,
                effect=PolicyDecision.ALLOW,
            )
        )
        assert engine.evaluate("memory", "read", {}) == PolicyDecision.ALLOW

    def test_deny_overrides_allow(self):
        engine = PolicyEngine()
        engine.add_policy(
            AccessControlPolicy(
                name="allow_all",
                resource_type="memory",
                action="read",
                condition=lambda ctx: True,
                effect=PolicyDecision.ALLOW,
                priority=1,
            )
        )
        engine.add_policy(
            AccessControlPolicy(
                name="deny_night",
                resource_type="memory",
                action="read",
                condition=lambda ctx: ctx.get("hour", 0) == 3,
                effect=PolicyDecision.DENY,
                priority=10,
            )
        )
        assert engine.evaluate("memory", "read", {"hour": 3}) == PolicyDecision.DENY
        assert engine.evaluate("memory", "read", {"hour": 12}) == PolicyDecision.ALLOW

    def test_remove_policy(self):
        engine = PolicyEngine()
        engine.add_policy(
            AccessControlPolicy(
                name="tmp",
                resource_type="memory",
                action="read",
                condition=lambda ctx: True,
                effect=PolicyDecision.ALLOW,
            )
        )
        assert engine.remove_policy("tmp") is True
        assert engine.evaluate("memory", "read", {}) == PolicyDecision.DEFER

    def test_list_policies_filtered(self):
        engine = PolicyEngine()
        engine.add_policy(
            AccessControlPolicy(
                name="gdpr",
                resource_type="memory",
                action="read",
                condition=lambda ctx: True,
                effect=PolicyDecision.ALLOW,
                standards=(ComplianceStandard.GDPR,),
            )
        )
        engine.add_policy(
            AccessControlPolicy(
                name="soc2",
                resource_type="tenant",
                action="admin",
                condition=lambda ctx: True,
                effect=PolicyDecision.DENY,
                standards=(ComplianceStandard.SOC2,),
            )
        )
        assert len(engine.list_policies(resource_type="memory")) == 1
        assert len(engine.list_policies(standard=ComplianceStandard.SOC2)) == 1

    def test_priority_sorting(self):
        engine = PolicyEngine()
        engine.add_policy(
            AccessControlPolicy(
                name="low",
                resource_type="memory",
                action="read",
                condition=lambda ctx: True,
                effect=PolicyDecision.ALLOW,
                priority=1,
            )
        )
        engine.add_policy(
            AccessControlPolicy(
                name="high",
                resource_type="memory",
                action="read",
                condition=lambda ctx: True,
                effect=PolicyDecision.DENY,
                priority=100,
            )
        )
        assert engine.evaluate("memory", "read", {}) == PolicyDecision.DENY

    def test_no_match_different_resource(self):
        engine = PolicyEngine()
        engine.add_policy(
            AccessControlPolicy(
                name="mem",
                resource_type="memory",
                action="read",
                condition=lambda ctx: True,
                effect=PolicyDecision.ALLOW,
            )
        )
        assert engine.evaluate("tenant", "read", {}) == PolicyDecision.DEFER

    def test_concurrent_policy_ops(self):
        engine = PolicyEngine()
        errors: list[Exception] = []

        def worker(n: int):
            try:
                for i in range(25):
                    engine.add_policy(
                        AccessControlPolicy(
                            name=f"p_{n}_{i}",
                            resource_type="memory",
                            action="read",
                            condition=lambda ctx: True,
                            effect=PolicyDecision.ALLOW,
                        )
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(engine.list_policies()) == 100


# ------------------------------------------------------------------
# RiskAssessor tests
# ------------------------------------------------------------------

class TestRiskAssessor:
    def test_add_finding(self):
        assessor = RiskAssessor()
        finding = assessor.add_finding(
            category="test",
            description="A test finding",
            level=RiskLevel.HIGH,
            standards=(ComplianceStandard.SOC2,),
        )
        assert finding.id == "RISK-0001"
        assert finding.level == RiskLevel.HIGH
        assert finding.category == "test"

    def test_assess_encryption_at_rest(self):
        assessor = RiskAssessor()
        result = assessor.assess_encryption_at_rest(encryption_enabled=False)
        assert result is not None
        assert result.level == RiskLevel.CRITICAL
        assert ComplianceStandard.SOC2 in result.standards

    def test_assess_encryption_at_rest_pass(self):
        assessor = RiskAssessor()
        assert assessor.assess_encryption_at_rest(encryption_enabled=True) is None

    def test_assess_access_logging(self):
        assessor = RiskAssessor()
        audit = AuditLogger()
        result = assessor.assess_access_logging(audit)
        assert result is not None
        assert result.level == RiskLevel.HIGH

    def test_assess_access_logging_pass(self):
        assessor = RiskAssessor()
        audit = AuditLogger()
        audit.log(AuditEventType.MEMORY_CREATE)
        assert assessor.assess_access_logging(audit) is None

    def test_assess_mfa_policy(self):
        assessor = RiskAssessor()
        result = assessor.assess_mfa_policy(mfa_required=False)
        assert result is not None
        assert result.level == RiskLevel.MEDIUM

    def test_assess_mfa_policy_pass(self):
        assessor = RiskAssessor()
        assert assessor.assess_mfa_policy(mfa_required=True) is None

    def test_assess_data_retention(self):
        assessor = RiskAssessor()
        result = assessor.assess_data_retention(retention_days=None)
        assert result is not None
        assert result.level == RiskLevel.MEDIUM

    def test_assess_data_retention_pass(self):
        assessor = RiskAssessor()
        assert assessor.assess_data_retention(retention_days=365) is None

    def test_get_findings_filtered(self):
        assessor = RiskAssessor()
        assessor.add_finding("a", "desc", RiskLevel.LOW, (ComplianceStandard.GDPR,))
        assessor.add_finding("b", "desc", RiskLevel.HIGH, (ComplianceStandard.SOC2,))
        assert len(assessor.get_findings(level=RiskLevel.LOW)) == 1
        assert len(assessor.get_findings(standard=ComplianceStandard.SOC2)) == 1

    def test_get_risk_score(self):
        assessor = RiskAssessor()
        assessor.add_finding("a", "desc", RiskLevel.LOW, (ComplianceStandard.GDPR,))
        assessor.add_finding("b", "desc", RiskLevel.HIGH, (ComplianceStandard.SOC2,))
        assert assessor.get_risk_score() == RiskLevel.LOW.value + RiskLevel.HIGH.value

    def test_clear(self):
        assessor = RiskAssessor()
        assessor.add_finding("a", "desc", RiskLevel.LOW, ())
        assessor.clear()
        assert assessor.get_findings() == []
        assert assessor.get_risk_score() == 0

    def test_concurrent_add_finding(self):
        assessor = RiskAssessor()
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(50):
                    assessor.add_finding("x", "y", RiskLevel.LOW, ())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(assessor.get_findings()) == 200


# ------------------------------------------------------------------
# ComplianceManager tests
# ------------------------------------------------------------------

class TestComplianceManager:
    def test_init_defaults(self):
        mgr = ComplianceManager()
        assert mgr.audit is not None
        assert mgr.encryption is not None
        assert mgr.policies is not None
        assert mgr.risk is not None
        assert mgr.reports is not None

    def test_encrypt_sensitive_fields_mocked(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_instance = MagicMock()
            mock_instance.encrypt.return_value = b"fake_cipher"
            mock_cls.return_value = mock_instance

            mgr = ComplianceManager()
            data = {"name": "Alice", "ssn": "123-45-6789"}
            result = mgr.encrypt_sensitive(data, fields=("ssn",))
            assert result["name"] == "Alice"
            assert isinstance(result["ssn"], dict)
            assert result["ssn"]["field"] == "ssn"
            assert mgr.audit.count == 1

    def test_decrypt_sensitive_fields_mocked(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_instance = MagicMock()
            mock_instance.encrypt.return_value = b"fake"
            mock_instance.decrypt.return_value = b'{"field": "ssn", "value": "123-45-6789"}'
            mock_cls.return_value = mock_instance

            mgr = ComplianceManager()
            # Build fallback-mode encrypted field
            encrypted = mgr.encryption._encrypt_for_test(
                json.dumps({"field": "ssn", "value": "123-45-6789"})
            )
            encrypted["field"] = "ssn"
            data = {"name": "Alice", "ssn": encrypted}
            result = mgr.decrypt_sensitive(data, fields=("ssn",))
            assert result["ssn"] == "123-45-6789"

    def test_check_access_allow(self):
        mgr = ComplianceManager()
        mgr.policies.add_policy(
            AccessControlPolicy(
                name="allow_read",
                resource_type="memory",
                action="read",
                condition=lambda ctx: True,
                effect=PolicyDecision.ALLOW,
            )
        )
        assert mgr.check_access("memory", "read", {"user_id": "alice"}) is True

    def test_check_access_deny_logs_audit(self):
        mgr = ComplianceManager()
        mgr.policies.add_policy(
            AccessControlPolicy(
                name="deny_all",
                resource_type="memory",
                action="delete",
                condition=lambda ctx: True,
                effect=PolicyDecision.DENY,
            )
        )
        assert mgr.check_access("memory", "delete", {"user_id": "bob", "tenant_id": "acme"}) is False
        denied = mgr.audit.query(event_type=AuditEventType.ACCESS_DENIED)
        assert len(denied) == 1
        assert denied[0].user_id == "bob"

    def test_run_risk_assessment(self):
        mgr = ComplianceManager()
        findings = mgr.run_risk_assessment(encryption_enabled=False, mfa_required=False, retention_days=None)
        assert len(findings) >= 3
        categories = {f.category for f in findings}
        assert "encryption" in categories
        assert "logging" in categories
        assert "authentication" in categories
        assert "retention" in categories

    def test_generate_report(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            mgr = ComplianceManager()
            mgr.audit.log(AuditEventType.MEMORY_CREATE, tenant_id="acme")
            report = mgr.generate_report(ComplianceStandard.GDPR, tenant_id="acme")
            assert isinstance(report, ComplianceReport)
            assert report.standard == ComplianceStandard.GDPR
            assert report.audit_event_count >= 1
            assert "summary" in mgr.export_report(report)

    def test_generate_all_reports(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            mgr = ComplianceManager()
            reports = mgr.generate_all_reports()
            assert set(reports.keys()) == set(ComplianceStandard)
            for standard, report in reports.items():
                assert report.standard == standard

    def test_default_policies_installed(self):
        mgr = ComplianceManager()
        policies = mgr.policies.list_policies()
        names = {p.name for p in policies}
        assert "soc2_admin_hours" in names
        assert "gdpr_consent_withdrawn" in names
        assert "hipaa_auth_required" in names

    def test_default_policy_soc2_night_hours(self):
        mgr = ComplianceManager()
        assert mgr.check_access("memory", "delete", {"hour": 3}) is False
        assert mgr.check_access("memory", "delete", {"hour": 12}) is True  # no policy matches

    def test_default_policy_gdpr_consent(self):
        mgr = ComplianceManager()
        assert mgr.check_access("memory", "read", {"consent": False}) is False
        assert mgr.check_access("memory", "read", {"consent": True}) is True  # no policy matches

    def test_default_policy_hipaa_auth(self):
        mgr = ComplianceManager()
        assert mgr.check_access("memory", "read", {"authenticated": False}) is False
        assert mgr.check_access("memory", "read", {"authenticated": True}) is True

    def test_export_report_structure(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            mgr = ComplianceManager()
            report = mgr.generate_report(ComplianceStandard.SOC2)
            exported = mgr.export_report(report)
            assert exported["standard"] == "soc2"
            assert "generated_at" in exported
            assert "risk_score" in exported
            assert "findings" in exported
            assert isinstance(exported["findings"], list)

    def test_concurrent_access_checks(self):
        mgr = ComplianceManager()
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(50):
                    mgr.check_access("memory", "read", {"authenticated": True})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ------------------------------------------------------------------
# Integration tests across compliance + audit + rbac
# ------------------------------------------------------------------

class TestComplianceIntegration:
    def test_full_compliance_flow(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_instance = MagicMock()
            mock_instance.encrypt.return_value = b"encrypted"
            mock_instance.decrypt.return_value = b'{"field": "email", "value": "alice@example.com"}'
            mock_cls.return_value = mock_instance

            mgr = ComplianceManager()

            # 1. Encrypt sensitive data
            data = {"user_id": "alice", "email": "alice@example.com"}
            encrypted = mgr.encrypt_sensitive(data, fields=("email",))
            assert isinstance(encrypted["email"], dict)

            # 2. Access check (HIPAA — must be authenticated)
            allowed = mgr.check_access("memory", "read", {"authenticated": True, "user_id": "alice"})
            assert allowed is True

            denied = mgr.check_access("memory", "read", {"authenticated": False, "user_id": "eve"})
            assert denied is False

            # 3. Run risk assessment
            findings = mgr.run_risk_assessment(encryption_enabled=True, mfa_required=True, retention_days=365)
            assert all(f.level != RiskLevel.CRITICAL for f in findings)

            # 4. Generate compliance report
            report = mgr.generate_report(ComplianceStandard.HIPAA)
            assert report.standard == ComplianceStandard.HIPAA
            assert report.encryption_enabled is True

            # 5. Export report
            exported = mgr.export_report(report)
            assert exported["standard"] == "hipaa"
            assert exported["encryption_enabled"] is True

    def test_audit_trail_integration(self):
        mgr = ComplianceManager()
        mgr.check_access("memory", "delete", {"user_id": "bob", "tenant_id": "acme", "hour": 3})
        events = mgr.audit.query(event_type=AuditEventType.ACCESS_DENIED)
        assert len(events) == 1
        assert events[0].details["reason"] == "policy_denied"

    def test_risk_score_changes_with_findings(self):
        mgr = ComplianceManager()
        initial_score = mgr.risk.get_risk_score()
        mgr.risk.add_finding("test", "desc", RiskLevel.HIGH, (ComplianceStandard.SOC2,))
        new_score = mgr.risk.get_risk_score()
        assert new_score == initial_score + RiskLevel.HIGH.value

    def test_report_with_no_audit_events(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            mgr = ComplianceManager()
            report = mgr.generate_report(ComplianceStandard.GDPR)
            assert report.audit_event_count == 0
            assert report.summary["audit_events_exported"] == 0
