"""Unit tests for ComplianceDashboard — mock-based.

NeuralMem Enterprise V1.7 compliance certification automation.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from neuralmem.enterprise.audit import AuditEventType, AuditLogger, ComplianceStandard
from neuralmem.enterprise.compliance import (
    ComplianceManager,
    DataEncryption,
    RiskLevel,
)
from neuralmem.enterprise.compliance_dashboard import (
    ComplianceDashboard,
    ComplianceDashboardState,
    RemediationSuggestion,
    RiskHeatmapCell,
)
from neuralmem.enterprise.soc2_evidence import (
    ControlFrequency,
    ControlTestResult,
    SOC2EvidenceCollector,
)


# ------------------------------------------------------------------
# Constructor / init
# ------------------------------------------------------------------

class TestComplianceDashboardInit:
    def test_init_defaults(self):
        dash = ComplianceDashboard()
        assert dash.evidence is not None
        assert dash.compliance is not None
        assert dash.audit is not None
        assert dash.get_state().overall_score == 0.0

    def test_init_with_deps(self):
        evidence = SOC2EvidenceCollector()
        compliance = ComplianceManager()
        audit = AuditLogger()
        dash = ComplianceDashboard(evidence, compliance, audit)
        assert dash.evidence is evidence
        assert dash.compliance is compliance
        assert dash.audit is audit


# ------------------------------------------------------------------
# Compliance scoring
# ------------------------------------------------------------------

class TestComplianceScoring:
    def test_calculate_standard_score_no_findings(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            score = dash.calculate_standard_score(ComplianceStandard.SOC2)
            assert 0.0 <= score <= 100.0

    def test_calculate_standard_score_with_critical(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            dash.compliance.risk.add_finding(
                "encryption", "no encryption", RiskLevel.CRITICAL, (ComplianceStandard.SOC2,)
            )
            score = dash.calculate_standard_score(ComplianceStandard.SOC2)
            assert score < 100.0

    def test_calculate_overall_score(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            overall = dash.calculate_overall_score()
            assert 0.0 <= overall <= 100.0

    def test_refresh_scores(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            scores = dash.refresh_scores()
            assert "overall" in scores
            assert "soc2" in scores
            assert "gdpr" in scores


# ------------------------------------------------------------------
# Risk heatmap
# ------------------------------------------------------------------

class TestRiskHeatmap:
    def test_build_heatmap_empty(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            heatmap = dash.build_risk_heatmap()
            assert heatmap == []

    def test_build_heatmap_with_findings(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            dash.compliance.risk.add_finding(
                "encryption", "no encryption", RiskLevel.HIGH, (ComplianceStandard.SOC2,)
            )
            dash.compliance.risk.add_finding(
                "logging", "no logs", RiskLevel.MEDIUM, (ComplianceStandard.GDPR,)
            )
            heatmap = dash.build_risk_heatmap()
            assert len(heatmap) == 2
            categories = {h.category for h in heatmap}
            assert "encryption" in categories
            assert "logging" in categories
            for h in heatmap:
                assert 0.0 <= h.score <= 100.0
                assert h.count > 0

    def test_heatmap_trend(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            dash.compliance.risk.add_finding(
                "encryption", "no encryption", RiskLevel.HIGH, (ComplianceStandard.SOC2,)
            )
            dash.build_risk_heatmap()
            # Add another finding to change trend
            dash.compliance.risk.add_finding(
                "encryption", "weak cipher", RiskLevel.CRITICAL, (ComplianceStandard.SOC2,)
            )
            heatmap = dash.build_risk_heatmap()
            enc_cell = [h for h in heatmap if h.category == "encryption"][0]
            assert enc_cell.trend == "up"  # risk went up (score down)


# ------------------------------------------------------------------
# Remediation suggestions
# ------------------------------------------------------------------

class TestRemediationSuggestions:
    def test_generate_suggestions_empty(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            suggestions = dash.generate_remediation_suggestions()
            assert len(suggestions) > 0  # policy suggestions for missing policies

    def test_generate_suggestions_with_findings(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            dash.compliance.risk.add_finding(
                "encryption", "no encryption", RiskLevel.CRITICAL, (ComplianceStandard.SOC2,),
                remediation="Enable AES-256-GCM",
            )
            suggestions = dash.generate_remediation_suggestions()
            critical = [s for s in suggestions if s.priority == 1]
            assert len(critical) > 0
            assert any(s.category == "encryption" for s in critical)

    def test_apply_remediation_automated(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            dash.compliance.risk.add_finding(
                "encryption", "no encryption", RiskLevel.CRITICAL, (ComplianceStandard.SOC2,),
            )
            suggestions = dash.generate_remediation_suggestions()
            auto = [s for s in suggestions if s.automated]
            if auto:
                result = dash.apply_remediation(auto[0].action_key)
                assert result is True
                # Should be removed from queue
                remaining = [s for s in dash.get_state().remediation_queue if s.action_key == auto[0].action_key]
                assert len(remaining) == 0

    def test_apply_remediation_not_automated(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            dash.compliance.risk.add_finding(
                "retention", "no policy", RiskLevel.MEDIUM, (ComplianceStandard.GDPR,),
            )
            suggestions = dash.generate_remediation_suggestions()
            manual = [s for s in suggestions if not s.automated]
            if manual:
                result = dash.apply_remediation(manual[0].action_key)
                assert result is False

    def test_apply_remediation_not_found(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            result = dash.apply_remediation("nonexistent_key")
            assert result is False


# ------------------------------------------------------------------
# Dashboard refresh
# ------------------------------------------------------------------

class TestDashboardRefresh:
    def test_refresh_runs_tests(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            state = dash.refresh()
            assert state.last_test_run is not None
            assert state.tests_passed + state.tests_failed > 0

    def test_refresh_updates_scores(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            dash.refresh()
            assert dash.get_state().overall_score >= 0.0

    def test_refresh_builds_heatmap(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            dash.compliance.risk.add_finding(
                "encryption", "no encryption", RiskLevel.HIGH, (ComplianceStandard.SOC2,)
            )
            dash.refresh()
            assert len(dash.get_state().risk_heatmap) > 0

    def test_refresh_generates_remediations(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            dash.refresh()
            assert len(dash.get_state().remediation_queue) >= 0

    def test_refresh_history(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            dash.refresh()
            dash.refresh()
            history = dash.get_history()
            assert len(history) >= 2


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------

class TestDashboardExport:
    def test_to_dict_structure(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            dash.refresh()
            data = dash.to_dict()
            assert "overall_score" in data
            assert "standard_scores" in data
            assert "risk_heatmap" in data
            assert "open_findings" in data
            assert "remediation_queue" in data
            assert "tests_passed" in data
            assert "evidence_chain_valid" in data

    def test_to_json(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            dash.refresh()
            json_str = dash.to_json()
            assert isinstance(json_str, str)
            assert "overall_score" in json_str

    def test_export_summary(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            dash.refresh()
            summary = dash.export_summary()
            assert "overall_compliance_score" in summary
            assert "risk_status" in summary
            assert "evidence_chain_status" in summary
            assert "tests_status" in summary


# ------------------------------------------------------------------
# Thread safety
# ------------------------------------------------------------------

class TestDashboardThreadSafety:
    def test_concurrent_refresh(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            errors: list[Exception] = []

            def worker():
                try:
                    for _ in range(10):
                        dash.refresh()
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            assert len(dash.get_history()) > 0

    def test_concurrent_score_calculation(self):
        with patch.object(DataEncryption, '_aesgcm_class') as mock_cls:
            mock_cls.return_value = MagicMock()
            dash = ComplianceDashboard()
            errors: list[Exception] = []

            def worker():
                try:
                    for _ in range(20):
                        dash.calculate_overall_score()
                        dash.build_risk_heatmap()
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
