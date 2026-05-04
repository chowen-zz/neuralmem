"""Unit tests for SOC2EvidenceCollector — mock-based.

NeuralMem Enterprise V1.7 compliance certification automation.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from neuralmem.enterprise.audit import AuditEventType, AuditLogger, ComplianceStandard
from neuralmem.enterprise.soc2_evidence import (
    ControlDefinition,
    ControlFrequency,
    ControlTestResult,
    EvidenceSnapshot,
    SOC2EvidenceCollector,
)


# ------------------------------------------------------------------
# Constructor / defaults
# ------------------------------------------------------------------

class TestSOC2EvidenceCollectorInit:
    def test_init_defaults(self):
        collector = SOC2EvidenceCollector()
        assert collector.audit is not None
        assert collector.standard == ComplianceStandard.SOC2
        assert len(collector.list_controls()) > 0

    def test_init_with_audit(self):
        audit = AuditLogger()
        collector = SOC2EvidenceCollector(audit_logger=audit)
        assert collector.audit is audit

    def test_init_standard_override(self):
        collector = SOC2EvidenceCollector(compliance_standard=ComplianceStandard.GDPR)
        assert collector.standard == ComplianceStandard.GDPR


# ------------------------------------------------------------------
# Control management
# ------------------------------------------------------------------

class TestControlManagement:
    def test_add_and_list_controls(self):
        collector = SOC2EvidenceCollector()
        ctrl = ControlDefinition(
            control_id="TEST-1",
            name="Test Control",
            description="desc",
            standard=ComplianceStandard.SOC2,
            frequency=ControlFrequency.DAILY,
            category="security",
            test_fn_name="_test_logical_access",
        )
        collector.add_control(ctrl)
        assert len(collector.list_controls()) == 10  # 9 defaults + 1

    def test_list_controls_filtered(self):
        collector = SOC2EvidenceCollector()
        security = collector.list_controls(category="security")
        daily = collector.list_controls(frequency=ControlFrequency.DAILY)
        assert all(c.category == "security" for c in security)
        assert all(c.frequency == ControlFrequency.DAILY for c in daily)

    def test_remove_control(self):
        collector = SOC2EvidenceCollector()
        assert collector.remove_control("CC6.1") is True
        assert collector.remove_control("CC6.1") is False
        ids = {c.control_id for c in collector.list_controls()}
        assert "CC6.1" not in ids


# ------------------------------------------------------------------
# Evidence snapshots and hash chain
# ------------------------------------------------------------------

class TestEvidenceSnapshots:
    def test_create_snapshot(self):
        collector = SOC2EvidenceCollector()
        snap = collector.create_snapshot(
            control_id="CC6.1",
            control_name="Logical Access",
            test_result=True,
            details={"ok": True},
        )
        assert isinstance(snap, EvidenceSnapshot)
        assert snap.control_id == "CC6.1"
        assert snap.test_result is True
        assert snap.snapshot_hash != "0" * 64
        assert snap.previous_hash == "0" * 64

    def test_snapshot_chain_links(self):
        collector = SOC2EvidenceCollector()
        snap1 = collector.create_snapshot("C1", "Name1", True, {})
        snap2 = collector.create_snapshot("C2", "Name2", False, {})
        assert snap2.previous_hash == snap1.snapshot_hash

    def test_verify_chain_valid(self):
        collector = SOC2EvidenceCollector()
        collector.create_snapshot("C1", "Name1", True, {})
        collector.create_snapshot("C2", "Name2", True, {})
        results = collector.verify_chain()
        assert all(valid for _, valid, _ in results)

    def test_verify_chain_tampered(self):
        collector = SOC2EvidenceCollector()
        snap = collector.create_snapshot("C1", "Name1", True, {})
        # Tamper with internal state (simulate corruption)
        with collector._lock:
            # Replace the snapshot with a corrupted one
            corrupted = EvidenceSnapshot(
                id=snap.id,
                control_id=snap.control_id,
                control_name=snap.control_name,
                timestamp=snap.timestamp,
                test_result=False,  # tampered
                details=snap.details,
                previous_hash=snap.previous_hash,
                snapshot_hash=snap.snapshot_hash,  # old hash
                standard=snap.standard,
            )
            collector._snapshots[-1] = corrupted
        results = collector.verify_chain()
        assert any(not valid for _, valid, _ in results)

    def test_get_snapshots_filtered(self):
        collector = SOC2EvidenceCollector()
        collector.create_snapshot("C1", "Name1", True, {})
        collector.create_snapshot("C2", "Name2", False, {})
        c1_snaps = collector.get_snapshots(control_id="C1")
        assert len(c1_snaps) == 1
        assert c1_snaps[0].control_id == "C1"


# ------------------------------------------------------------------
# Control test execution
# ------------------------------------------------------------------

class TestControlTests:
    def test_run_control_test_builtin(self):
        collector = SOC2EvidenceCollector()
        ctrl = [c for c in collector.list_controls() if c.control_id == "CC6.1"][0]
        result = collector.run_control_test(ctrl)
        assert isinstance(result, ControlTestResult)
        assert result.control_id == "CC6.1"
        assert result.passed is True
        assert result.evidence is not None

    def test_run_control_test_missing_fn(self):
        collector = SOC2EvidenceCollector()
        ctrl = ControlDefinition(
            control_id="BAD",
            name="Bad",
            description="desc",
            standard=ComplianceStandard.SOC2,
            frequency=ControlFrequency.DAILY,
            category="security",
            test_fn_name="_nonexistent_method",
        )
        collector.add_control(ctrl)
        result = collector.run_control_test(ctrl)
        assert result.passed is False
        assert result.severity == "critical"

    def test_run_all_tests(self):
        collector = SOC2EvidenceCollector()
        results = collector.run_all_tests()
        assert len(results) == 9
        assert all(r.passed for r in results)

    def test_run_daily_tests(self):
        collector = SOC2EvidenceCollector()
        results = collector.run_daily_tests()
        assert all(r.evidence.standard == ComplianceStandard.SOC2 for r in results)

    def test_run_weekly_tests(self):
        collector = SOC2EvidenceCollector()
        results = collector.run_weekly_tests()
        daily = {c.control_id for c in collector.list_controls(frequency=ControlFrequency.DAILY)}
        weekly = {c.control_id for c in collector.list_controls(frequency=ControlFrequency.WEEKLY)}
        assert len(results) == len(weekly)
        assert all(r.control_id in weekly for r in results)
        assert all(r.control_id not in daily for r in results)

    def test_run_tests_by_frequency(self):
        collector = SOC2EvidenceCollector()
        results = collector.run_tests_by_frequency(ControlFrequency.MONTHLY)
        assert results == []


# ------------------------------------------------------------------
# Compliance score
# ------------------------------------------------------------------

class TestComplianceScore:
    def test_score_zero_before_tests(self):
        collector = SOC2EvidenceCollector()
        assert collector.get_compliance_score() == 0.0

    def test_score_after_tests(self):
        collector = SOC2EvidenceCollector()
        collector.run_all_tests()
        score = collector.get_compliance_score()
        assert score == 100.0

    def test_score_with_failure(self):
        collector = SOC2EvidenceCollector()
        collector.run_all_tests()
        # Add a failing test result manually
        snap = collector.create_snapshot("FAIL", "Fail", False, {})
        from neuralmem.enterprise.soc2_evidence import ControlTestResult
        result = ControlTestResult(
            control_id="FAIL",
            control_name="Fail",
            passed=False,
            timestamp=snap.timestamp,
            evidence=snap,
        )
        with collector._lock:
            collector._test_results.append(result)
        score = collector.get_compliance_score()
        assert score < 100.0


# ------------------------------------------------------------------
# Control summary
# ------------------------------------------------------------------

class TestControlSummary:
    def test_summary_structure(self):
        collector = SOC2EvidenceCollector()
        collector.run_all_tests()
        summary = collector.get_control_summary()
        assert "total_controls" in summary
        assert "compliance_score" in summary
        assert "controls" in summary
        assert summary["total_controls"] == 9
        assert summary["standard"] == "soc2"

    def test_summary_control_latest_result(self):
        collector = SOC2EvidenceCollector()
        collector.run_all_tests()
        summary = collector.get_control_summary()
        for ctrl in summary["controls"]:
            assert ctrl["latest_result"]["passed"] is True
            assert ctrl["latest_result"]["snapshot_id"] is not None


# ------------------------------------------------------------------
# PDF report generation (mock-based)
# ------------------------------------------------------------------

class TestPDFReport:
    def test_generate_pdf_report_structure(self):
        collector = SOC2EvidenceCollector()
        collector.run_all_tests()
        report = collector.generate_pdf_report()
        assert report["report_type"] == "SOC2_Evidence_Report"
        assert "executive_summary" in report
        assert "evidence_chain" in report
        assert "control_details" in report
        assert "pdf_metadata" in report
        assert report["pdf_metadata"]["mock_pdf"] is True

    def test_pdf_report_period_filter(self):
        collector = SOC2EvidenceCollector()
        collector.run_all_tests()
        start = datetime.now(timezone.utc) - timedelta(days=1)
        end = datetime.now(timezone.utc) + timedelta(days=1)
        report = collector.generate_pdf_report(start_date=start, end_date=end)
        assert len(report["control_details"]) > 0

    def test_pdf_report_empty_period(self):
        collector = SOC2EvidenceCollector()
        start = datetime(2000, 1, 1, tzinfo=timezone.utc)
        end = datetime(2000, 1, 2, tzinfo=timezone.utc)
        report = collector.generate_pdf_report(start_date=start, end_date=end)
        assert report["executive_summary"]["total_tests_run"] == 0

    def test_export_report_json(self):
        collector = SOC2EvidenceCollector()
        collector.run_all_tests()
        report = collector.generate_pdf_report()
        json_str = collector.export_report_json(report)
        assert isinstance(json_str, str)
        assert "SOC2_Evidence_Report" in json_str

    def test_pdf_chain_integrity(self):
        collector = SOC2EvidenceCollector()
        collector.run_all_tests()
        report = collector.generate_pdf_report()
        assert report["evidence_chain"]["chain_integrity"] == "VALID"


# ------------------------------------------------------------------
# Audit integration
# ------------------------------------------------------------------

class TestAuditIntegration:
    def test_test_run_logs_audit(self):
        collector = SOC2EvidenceCollector()
        before = collector.audit.count
        collector.run_all_tests()
        after = collector.audit.count
        assert after > before

    def test_audit_event_details(self):
        collector = SOC2EvidenceCollector()
        collector.run_all_tests()
        events = collector.audit.query()
        control_events = [e for e in events if e.action == "control_test"]
        assert len(control_events) > 0
        assert all(e.details.get("control_id") is not None for e in control_events)


# ------------------------------------------------------------------
# Thread safety
# ------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_snapshots(self):
        collector = SOC2EvidenceCollector()
        errors: list[Exception] = []

        def worker(n: int):
            try:
                for i in range(25):
                    collector.create_snapshot(
                        control_id=f"C{n}",
                        control_name=f"Control{n}",
                        test_result=True,
                        details={"i": i},
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(collector._snapshots) == 100

    def test_concurrent_control_tests(self):
        collector = SOC2EvidenceCollector()
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(10):
                    collector.run_all_tests()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(collector._test_results) == 270  # 9 controls * 10 * 3
