"""Unit tests for DataRetentionEnforcer — mock-based.

NeuralMem Enterprise V1.7 compliance certification automation.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from neuralmem.enterprise.audit import AuditEventType, AuditLogger, ComplianceStandard
from neuralmem.enterprise.retention import (
    DataRecord,
    DataRetentionEnforcer,
    DeletionRequest,
    RetentionPolicy,
    RetentionTier,
)


# ------------------------------------------------------------------
# Constructor / defaults
# ------------------------------------------------------------------

class TestDataRetentionEnforcerInit:
    def test_init_defaults(self):
        enforcer = DataRetentionEnforcer()
        assert enforcer.audit is not None
        assert len(enforcer.list_policies()) > 0

    def test_init_with_audit(self):
        audit = AuditLogger()
        enforcer = DataRetentionEnforcer(audit_logger=audit)
        assert enforcer.audit is audit


# ------------------------------------------------------------------
# Policy management
# ------------------------------------------------------------------

class TestPolicyManagement:
    def test_set_and_get_policy(self):
        enforcer = DataRetentionEnforcer()
        policy = RetentionPolicy(
            category="test",
            tier=RetentionTier.HOT,
            retention_days=30,
        )
        enforcer.set_policy(policy)
        got = enforcer.get_policy("test")
        assert got is not None
        assert got.category == "test"
        assert got.retention_days == 30

    def test_remove_policy(self):
        enforcer = DataRetentionEnforcer()
        assert enforcer.remove_policy("memory") is True
        assert enforcer.remove_policy("memory") is False
        assert enforcer.get_policy("memory") is None

    def test_list_policies(self):
        enforcer = DataRetentionEnforcer()
        policies = enforcer.list_policies()
        assert len(policies) >= 6  # defaults
        categories = {p.category for p in policies}
        assert "memory" in categories
        assert "audit" in categories

    def test_default_policy_memory(self):
        enforcer = DataRetentionEnforcer()
        policy = enforcer.get_policy("memory")
        assert policy is not None
        assert policy.tier == RetentionTier.HOT
        assert policy.retention_days == 365
        assert policy.gdpr_applies is True
        assert policy.auto_delete is False

    def test_default_policy_audit(self):
        enforcer = DataRetentionEnforcer()
        policy = enforcer.get_policy("audit")
        assert policy is not None
        assert policy.tier == RetentionTier.ARCHIVE
        assert policy.retention_days == 2555  # 7 years
        assert policy.gdpr_applies is False


# ------------------------------------------------------------------
# Record tracking
# ------------------------------------------------------------------

class TestRecordTracking:
    def test_track_record(self):
        enforcer = DataRetentionEnforcer()
        record = enforcer.track_record("r1", "memory", "tenant1", "user1")
        assert record.record_id == "r1"
        assert record.category == "memory"
        assert record.tenant_id == "tenant1"
        assert record.user_id == "user1"
        assert record.tier == RetentionTier.HOT
        assert record.deleted is False

    def test_get_record(self):
        enforcer = DataRetentionEnforcer()
        enforcer.track_record("r1", "memory", "tenant1")
        got = enforcer.get_record("r1")
        assert got is not None
        assert got.record_id == "r1"

    def test_list_records_filtered(self):
        enforcer = DataRetentionEnforcer()
        enforcer.track_record("r1", "memory", "t1", "u1")
        enforcer.track_record("r2", "session", "t1", "u1")
        enforcer.track_record("r3", "memory", "t2", "u2")
        assert len(enforcer.list_records(category="memory")) == 2  # r1 + r3
        assert len(enforcer.list_records(tenant_id="t1")) == 2
        assert len(enforcer.list_records(user_id="u1")) == 2
        assert len(enforcer.list_records(tier=RetentionTier.HOT)) == 2


# ------------------------------------------------------------------
# Tier transitions
# ------------------------------------------------------------------

class TestTierTransitions:
    def test_transition_tier(self):
        enforcer = DataRetentionEnforcer()
        enforcer.track_record("r1", "memory", "t1")
        result = enforcer.transition_tier("r1", RetentionTier.COLD)
        assert result is True
        record = enforcer.get_record("r1")
        assert record.tier == RetentionTier.COLD

    def test_transition_tier_missing(self):
        enforcer = DataRetentionEnforcer()
        assert enforcer.transition_tier("missing", RetentionTier.COLD) is False

    def test_transition_tier_deleted(self):
        enforcer = DataRetentionEnforcer()
        enforcer.track_record("r1", "memory", "t1")
        record = enforcer.get_record("r1")
        record.deleted = True
        assert enforcer.transition_tier("r1", RetentionTier.COLD) is False

    def test_auto_transition_hot_to_warm(self):
        enforcer = DataRetentionEnforcer()
        old_time = datetime.now(timezone.utc) - timedelta(days=31)
        enforcer.track_record("r1", "memory", "t1", created_at=old_time)
        transitions = enforcer.auto_transition_tiers()
        assert len(transitions) > 0
        assert transitions[0][1] == RetentionTier.HOT
        assert transitions[0][2] == RetentionTier.WARM

    def test_auto_transition_warm_to_cold(self):
        enforcer = DataRetentionEnforcer()
        old_time = datetime.now(timezone.utc) - timedelta(days=91)
        record = enforcer.track_record("r1", "memory", "t1", created_at=old_time)
        record.tier = RetentionTier.WARM
        transitions = enforcer.auto_transition_tiers()
        assert any(t[2] == RetentionTier.COLD for t in transitions)

    def test_auto_transition_no_change_recent(self):
        enforcer = DataRetentionEnforcer()
        enforcer.track_record("r1", "memory", "t1")
        transitions = enforcer.auto_transition_tiers()
        assert len(transitions) == 0


# ------------------------------------------------------------------
# GDPR erasure
# ------------------------------------------------------------------

class TestGDPRerasure:
    def test_submit_erasure_request(self):
        enforcer = DataRetentionEnforcer()
        req = enforcer.submit_erasure_request("user1", "tenant1")
        assert req.request_id.startswith("GDPR-DEL-")
        assert req.user_id == "user1"
        assert req.tenant_id == "tenant1"
        assert req.status == "pending"

    def test_submit_erasure_with_categories(self):
        enforcer = DataRetentionEnforcer()
        req = enforcer.submit_erasure_request("user1", "tenant1", categories=["memory"])
        assert req.categories == ["memory"]

    def test_process_erasure_request(self):
        enforcer = DataRetentionEnforcer()
        enforcer.track_record("r1", "memory", "tenant1", "user1")
        enforcer.track_record("r2", "memory", "tenant1", "user1")
        enforcer.track_record("r3", "memory", "tenant1", "user2")
        req = enforcer.submit_erasure_request("user1", "tenant1")
        result = enforcer.process_erasure_request(req.request_id)
        assert result.status == "completed"
        assert result.records_affected == 2
        assert enforcer.get_record("r1").deleted is True
        assert enforcer.get_record("r2").deleted is True
        assert enforcer.get_record("r3").deleted is False

    def test_process_erasure_respects_legal_hold(self):
        enforcer = DataRetentionEnforcer()
        policy = enforcer.get_policy("memory")
        policy.legal_hold = True
        enforcer.track_record("r1", "memory", "tenant1", "user1")
        req = enforcer.submit_erasure_request("user1", "tenant1")
        result = enforcer.process_erasure_request(req.request_id)
        assert result.records_affected == 0
        assert enforcer.get_record("r1").deleted is False

    def test_process_erasure_respects_categories(self):
        enforcer = DataRetentionEnforcer()
        enforcer.track_record("r1", "memory", "tenant1", "user1")
        enforcer.track_record("r2", "session", "tenant1", "user1")
        req = enforcer.submit_erasure_request("user1", "tenant1", categories=["memory"])
        result = enforcer.process_erasure_request(req.request_id)
        assert result.records_affected == 1
        assert enforcer.get_record("r1").deleted is True
        assert enforcer.get_record("r2").deleted is False

    def test_process_erasure_non_gdpr_skipped(self):
        enforcer = DataRetentionEnforcer()
        enforcer.track_record("r1", "audit", "tenant1", "user1")
        req = enforcer.submit_erasure_request("user1", "tenant1")
        result = enforcer.process_erasure_request(req.request_id)
        # audit has gdpr_applies=False, so skipped unless explicit categories
        assert result.records_affected == 0

    def test_process_erasure_not_found(self):
        enforcer = DataRetentionEnforcer()
        with pytest.raises(ValueError):
            enforcer.process_erasure_request("nonexistent")

    def test_process_erasure_already_processed(self):
        enforcer = DataRetentionEnforcer()
        req = enforcer.submit_erasure_request("user1", "tenant1")
        enforcer.process_erasure_request(req.request_id)
        result2 = enforcer.process_erasure_request(req.request_id)
        assert result2.status == "completed"

    def test_get_erasure_request(self):
        enforcer = DataRetentionEnforcer()
        req = enforcer.submit_erasure_request("user1", "tenant1")
        got = enforcer.get_erasure_request(req.request_id)
        assert got is not None
        assert got.request_id == req.request_id

    def test_list_erasure_requests(self):
        enforcer = DataRetentionEnforcer()
        enforcer.submit_erasure_request("user1", "tenant1")
        enforcer.submit_erasure_request("user2", "tenant1")
        assert len(enforcer.list_erasure_requests()) == 2
        assert len(enforcer.list_erasure_requests(user_id="user1")) == 1


# ------------------------------------------------------------------
# Automatic retention enforcement
# ------------------------------------------------------------------

class TestRetentionEnforcement:
    def test_enforce_retention_deletes_expired(self):
        enforcer = DataRetentionEnforcer()
        old_time = datetime.now(timezone.utc) - timedelta(days=400)
        enforcer.track_record("r1", "user_profile", "t1", created_at=old_time)
        deleted = enforcer.enforce_retention()
        assert len(deleted) == 1
        assert deleted[0].record_id == "r1"
        assert deleted[0].deleted is True

    def test_enforce_retention_skips_legal_hold(self):
        enforcer = DataRetentionEnforcer()
        policy = enforcer.get_policy("user_profile")
        policy.legal_hold = True
        old_time = datetime.now(timezone.utc) - timedelta(days=400)
        enforcer.track_record("r1", "user_profile", "t1", created_at=old_time)
        deleted = enforcer.enforce_retention()
        assert len(deleted) == 0

    def test_enforce_retention_skips_no_auto_delete(self):
        enforcer = DataRetentionEnforcer()
        # memory has auto_delete=False
        old_time = datetime.now(timezone.utc) - timedelta(days=400)
        enforcer.track_record("r1", "memory", "t1", created_at=old_time)
        deleted = enforcer.enforce_retention()
        assert len(deleted) == 0

    def test_enforce_retention_skips_recent(self):
        enforcer = DataRetentionEnforcer()
        enforcer.track_record("r1", "user_profile", "t1")
        deleted = enforcer.enforce_retention()
        assert len(deleted) == 0

    def test_enforce_retention_skips_archive_no_days(self):
        enforcer = DataRetentionEnforcer()
        # audit has retention_days=2555, not None
        old_time = datetime.now(timezone.utc) - timedelta(days=3000)
        enforcer.track_record("r1", "audit", "t1", created_at=old_time)
        deleted = enforcer.enforce_retention()
        # audit has auto_delete=False
        assert len(deleted) == 0


# ------------------------------------------------------------------
# Compliance verification
# ------------------------------------------------------------------

class TestComplianceVerification:
    def test_verify_compliance_all_good(self):
        enforcer = DataRetentionEnforcer()
        enforcer.track_record("r1", "memory", "t1")
        result = enforcer.verify_compliance()
        assert result["compliance_rate"] == 100.0
        assert result["non_compliant_records"] == 0
        assert result["issues"] == []

    def test_verify_compliance_retention_exceeded(self):
        enforcer = DataRetentionEnforcer()
        old_time = datetime.now(timezone.utc) - timedelta(days=400)
        enforcer.track_record("r1", "memory", "t1", created_at=old_time)
        result = enforcer.verify_compliance()
        assert result["non_compliant_records"] == 1
        assert len(result["issues"]) == 1
        assert result["issues"][0]["issue"] == "retention_exceeded"

    def test_verify_compliance_no_policy(self):
        enforcer = DataRetentionEnforcer()
        enforcer.track_record("r1", "unknown_category", "t1")
        result = enforcer.verify_compliance()
        assert result["non_compliant_records"] == 1
        assert result["issues"][0]["issue"] == "no_policy"

    def test_verify_compliance_deleted_not_counted(self):
        enforcer = DataRetentionEnforcer()
        enforcer.track_record("r1", "memory", "t1")
        enforcer.get_record("r1").deleted = True
        result = enforcer.verify_compliance()
        assert result["active_records"] == 0
        assert result["deleted_records"] == 1


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------

class TestRetentionExport:
    def test_export_retention_report(self):
        enforcer = DataRetentionEnforcer()
        enforcer.track_record("r1", "memory", "t1")
        report = enforcer.export_retention_report()
        assert "generated_at" in report
        assert "policies" in report
        assert "records_by_tier" in report
        assert "gdpr_requests" in report
        assert "compliance_summary" in report
        assert report["records_by_tier"]["hot"] >= 1

    def test_to_json(self):
        enforcer = DataRetentionEnforcer()
        json_str = enforcer.to_json()
        assert isinstance(json_str, str)
        assert "generated_at" in json_str


# ------------------------------------------------------------------
# Audit integration
# ------------------------------------------------------------------

class TestRetentionAudit:
    def test_erasure_logs_audit(self):
        enforcer = DataRetentionEnforcer()
        before = enforcer.audit.count
        enforcer.submit_erasure_request("user1", "tenant1")
        after = enforcer.audit.count
        assert after > before

    def test_retention_deletion_logs_audit(self):
        enforcer = DataRetentionEnforcer()
        old_time = datetime.now(timezone.utc) - timedelta(days=400)
        enforcer.track_record("r1", "user_profile", "t1", created_at=old_time)
        before = enforcer.audit.count
        enforcer.enforce_retention()
        after = enforcer.audit.count
        assert after > before

    def test_tier_transition_logs_audit(self):
        enforcer = DataRetentionEnforcer()
        enforcer.track_record("r1", "memory", "t1")
        before = enforcer.audit.count
        enforcer.transition_tier("r1", RetentionTier.COLD)
        after = enforcer.audit.count
        assert after > before


# ------------------------------------------------------------------
# Thread safety
# ------------------------------------------------------------------

class TestRetentionThreadSafety:
    def test_concurrent_track_records(self):
        enforcer = DataRetentionEnforcer()
        errors: list[Exception] = []

        def worker(n: int):
            try:
                for i in range(25):
                    enforcer.track_record(f"r{n}_{i}", "memory", f"t{n}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(enforcer._records) == 100

    def test_concurrent_erasure_requests(self):
        enforcer = DataRetentionEnforcer()
        errors: list[Exception] = []

        def worker(n: int):
            try:
                for i in range(10):
                    enforcer.submit_erasure_request(f"user{n}", f"tenant{n}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(enforcer._deletion_requests) == 40

    def test_concurrent_enforce_retention(self):
        enforcer = DataRetentionEnforcer()
        old_time = datetime.now(timezone.utc) - timedelta(days=400)
        for i in range(50):
            enforcer.track_record(f"r{i}", "user_profile", "t1", created_at=old_time)
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(5):
                    enforcer.enforce_retention()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # All records should be deleted after first enforcement
        assert all(r.deleted for r in enforcer._records.values())
