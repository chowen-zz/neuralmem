"""Compliance Dashboard — real-time compliance score, risk heatmap,
auto-remediation suggestions.

NeuralMem Enterprise V1.7 compliance certification automation.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neuralmem.enterprise.audit import AuditEventType, AuditLogger, ComplianceStandard
from neuralmem.enterprise.compliance import (
    ComplianceManager,
    ComplianceReport,
    RiskFinding,
    RiskLevel,
)
from neuralmem.enterprise.soc2_evidence import (
    ControlFrequency,
    ControlTestResult,
    SOC2EvidenceCollector,
)


# ------------------------------------------------------------------
# Risk heatmap cell
# ------------------------------------------------------------------

@dataclass
class RiskHeatmapCell:
    """A single cell in the risk heatmap."""

    category: str
    level: RiskLevel
    count: int
    score: float  # 0.0 - 100.0
    trend: str = "stable"  # up, down, stable


@dataclass
class RemediationSuggestion:
    """Auto-generated remediation suggestion."""

    id: str
    category: str
    title: str
    description: str
    priority: int  # 1 = highest
    estimated_effort: str  # e.g. "1h", "1d", "1w"
    automated: bool  # can be auto-applied
    action_key: str  # key to reference the action


@dataclass
class ComplianceDashboardState:
    """Serializable dashboard state."""

    timestamp: datetime
    overall_score: float
    standard_scores: dict[str, float]
    risk_heatmap: list[RiskHeatmapCell]
    open_findings: int
    critical_findings: int
    remediation_queue: list[RemediationSuggestion]
    last_test_run: datetime | None
    tests_passed: int
    tests_failed: int
    evidence_chain_valid: bool


# ------------------------------------------------------------------
# Compliance Dashboard
# ------------------------------------------------------------------

class ComplianceDashboard:
    """Real-time compliance dashboard with scoring, heatmap, and auto-remediation.

    Features
    --------
    * Real-time compliance scoring (0-100) per standard
    * Risk heatmap by category
    * Auto-remediation suggestions with priority ranking
    * Thread-safe state updates
    * JSON export for external dashboards
    """

    def __init__(
        self,
        evidence_collector: SOC2EvidenceCollector | None = None,
        compliance_manager: ComplianceManager | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.evidence = evidence_collector or SOC2EvidenceCollector()
        self.compliance = compliance_manager or ComplianceManager()
        self.audit = audit_logger or AuditLogger()
        self._state: ComplianceDashboardState = self._empty_state()
        self._lock = threading.RLock()
        self._history: list[ComplianceDashboardState] = []
        self._remediation_counter = 0

    def _empty_state(self) -> ComplianceDashboardState:
        return ComplianceDashboardState(
            timestamp=datetime.now(timezone.utc),
            overall_score=0.0,
            standard_scores={},
            risk_heatmap=[],
            open_findings=0,
            critical_findings=0,
            remediation_queue=[],
            last_test_run=None,
            tests_passed=0,
            tests_failed=0,
            evidence_chain_valid=True,
        )

    # ------------------------------------------------------------------
    # Real-time compliance scoring
    # ------------------------------------------------------------------

    def calculate_standard_score(
        self,
        standard: ComplianceStandard,
    ) -> float:
        """Calculate compliance score (0-100) for a specific standard.

        Combines evidence collector score, risk findings, and policy coverage.
        """
        # Base score from evidence collector
        base_score = self.evidence.get_compliance_score()

        # Deduct for risk findings related to this standard
        risk_findings = self.compliance.risk.get_findings(standard=standard)
        deduction = 0.0
        for finding in risk_findings:
            if finding.level == RiskLevel.CRITICAL:
                deduction += 25.0
            elif finding.level == RiskLevel.HIGH:
                deduction += 15.0
            elif finding.level == RiskLevel.MEDIUM:
                deduction += 8.0
            elif finding.level == RiskLevel.LOW:
                deduction += 3.0

        # Deduct for missing policies
        policies = self.compliance.policies.list_policies(standard=standard)
        if not policies:
            deduction += 10.0

        score = max(0.0, min(100.0, base_score - deduction))
        return round(score, 2)

    def calculate_overall_score(self) -> float:
        """Calculate overall compliance score across all standards."""
        scores: list[float] = []
        for standard in ComplianceStandard:
            scores.append(self.calculate_standard_score(standard))
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 2)

    def refresh_scores(self) -> dict[str, float]:
        """Recalculate all scores and return them."""
        with self._lock:
            standard_scores = {
                s.value: self.calculate_standard_score(s)
                for s in ComplianceStandard
            }
            overall = self.calculate_overall_score()
            self._state.overall_score = overall
            self._state.standard_scores = standard_scores
            self._state.timestamp = datetime.now(timezone.utc)
            return {"overall": overall, **standard_scores}

    # ------------------------------------------------------------------
    # Risk heatmap by category
    # ------------------------------------------------------------------

    def build_risk_heatmap(self) -> list[RiskHeatmapCell]:
        """Build a risk heatmap organized by category.

        Categories: security, availability, confidentiality, retention,
        authentication, encryption, logging.
        """
        with self._lock:
            findings = self.compliance.risk.get_findings()
            categories: dict[str, list[RiskFinding]] = {}
            for f in findings:
                categories.setdefault(f.category, []).append(f)

            heatmap: list[RiskHeatmapCell] = []
            for category, cat_findings in categories.items():
                max_level = max((f.level for f in cat_findings), default=RiskLevel.LOW)
                count = len(cat_findings)
                # Score: 100 - weighted deductions
                score = max(0.0, 100.0 - sum(
                    25.0 if f.level == RiskLevel.CRITICAL else
                    15.0 if f.level == RiskLevel.HIGH else
                    8.0 if f.level == RiskLevel.MEDIUM else
                    3.0
                    for f in cat_findings
                ))
                # Simple trend: compare with previous state
                prev = None
                for h in reversed(self._state.risk_heatmap):
                    if h.category == category:
                        prev = h
                        break
                if prev is None:
                    trend = "stable"
                elif score < prev.score - 5:
                    trend = "up"  # risk went up (score down)
                elif score > prev.score + 5:
                    trend = "down"  # risk went down (score up)
                else:
                    trend = "stable"

                heatmap.append(RiskHeatmapCell(
                    category=category,
                    level=max_level,
                    count=count,
                    score=round(score, 2),
                    trend=trend,
                ))

            self._state.risk_heatmap = heatmap
            self._state.open_findings = len(findings)
            self._state.critical_findings = sum(
                1 for f in findings if f.level == RiskLevel.CRITICAL
            )
            return heatmap

    # ------------------------------------------------------------------
    # Auto-remediation suggestions
    # ------------------------------------------------------------------

    def _next_remediation_id(self) -> str:
        with self._lock:
            self._remediation_counter += 1
            return f"REM-{self._remediation_counter:04d}"

    def generate_remediation_suggestions(self) -> list[RemediationSuggestion]:
        """Generate prioritized auto-remediation suggestions based on findings."""
        with self._lock:
            suggestions: list[RemediationSuggestion] = []
            findings = self.compliance.risk.get_findings()

            for finding in findings:
                if finding.level == RiskLevel.CRITICAL:
                    priority = 1
                    effort = "1h"
                    automated = True
                elif finding.level == RiskLevel.HIGH:
                    priority = 2
                    effort = "1d"
                    automated = finding.category in ("encryption", "logging")
                elif finding.level == RiskLevel.MEDIUM:
                    priority = 3
                    effort = "1w"
                    automated = False
                else:
                    priority = 4
                    effort = "2w"
                    automated = False

                suggestions.append(RemediationSuggestion(
                    id=self._next_remediation_id(),
                    category=finding.category,
                    title=f"Fix {finding.category} issue: {finding.description[:50]}",
                    description=finding.remediation or f"Address {finding.category} finding {finding.id}",
                    priority=priority,
                    estimated_effort=effort,
                    automated=automated,
                    action_key=f"remediate_{finding.category}_{finding.id}",
                ))

            # Add suggestions for missing policies
            for standard in ComplianceStandard:
                policies = self.compliance.policies.list_policies(standard=standard)
                if not policies:
                    suggestions.append(RemediationSuggestion(
                        id=self._next_remediation_id(),
                        category="policy",
                        title=f"Add policies for {standard.value.upper()}",
                        description=f"No access control policies defined for {standard.value.upper()}.",
                        priority=2,
                        estimated_effort="1d",
                        automated=False,
                        action_key=f"add_policies_{standard.value}",
                    ))

            # Sort by priority
            suggestions.sort(key=lambda s: s.priority)
            self._state.remediation_queue = suggestions
            return suggestions

    def apply_remediation(self, action_key: str) -> bool:
        """Apply an automated remediation by action key.

        Returns True if applied, False if not automated or not found.
        """
        with self._lock:
            for suggestion in self._state.remediation_queue:
                if suggestion.action_key == action_key:
                    if not suggestion.automated:
                        return False
                    # Mock application: log and remove from queue
                    self.audit.log(
                        event_type=AuditEventType.PERMISSION_CHANGE,
                        action="auto_remediate",
                        resource="compliance",
                        details={
                            "remediation_id": suggestion.id,
                            "action_key": action_key,
                            "category": suggestion.category,
                        },
                    )
                    self._state.remediation_queue = [
                        s for s in self._state.remediation_queue
                        if s.action_key != action_key
                    ]
                    return True
            return False

    # ------------------------------------------------------------------
    # Dashboard refresh / state management
    # ------------------------------------------------------------------

    def refresh(self) -> ComplianceDashboardState:
        """Refresh the entire dashboard state.

        Runs control tests, recalculates scores, builds heatmap,
        and generates remediation suggestions.
        """
        with self._lock:
            # Run daily tests
            daily_results = self.evidence.run_daily_tests()
            weekly_results = self.evidence.run_weekly_tests()
            all_results = daily_results + weekly_results

            self._state.tests_passed = sum(1 for r in all_results if r.passed)
            self._state.tests_failed = len(all_results) - self._state.tests_passed
            self._state.last_test_run = datetime.now(timezone.utc)

            # Refresh scores
            self.refresh_scores()

            # Build heatmap
            self.build_risk_heatmap()

            # Generate remediation suggestions
            self.generate_remediation_suggestions()

            # Verify evidence chain
            chain_results = self.evidence.verify_chain()
            self._state.evidence_chain_valid = all(v for _, v, _ in chain_results)

            # Snapshot history
            self._history.append(self._state)
            if len(self._history) > 100:
                self._history.pop(0)

            return self._state

    def get_state(self) -> ComplianceDashboardState:
        """Return current dashboard state."""
        with self._lock:
            return self._state

    def get_history(self, limit: int = 24) -> list[ComplianceDashboardState]:
        """Return recent dashboard state history."""
        with self._lock:
            return list(self._history[-limit:])

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Export dashboard state as a plain dict."""
        with self._lock:
            state = self._state
            return {
                "timestamp": state.timestamp.isoformat(),
                "overall_score": state.overall_score,
                "standard_scores": state.standard_scores,
                "risk_heatmap": [
                    {
                        "category": h.category,
                        "level": h.level.name,
                        "count": h.count,
                        "score": h.score,
                        "trend": h.trend,
                    }
                    for h in state.risk_heatmap
                ],
                "open_findings": state.open_findings,
                "critical_findings": state.critical_findings,
                "remediation_queue": [
                    {
                        "id": r.id,
                        "category": r.category,
                        "title": r.title,
                        "priority": r.priority,
                        "estimated_effort": r.estimated_effort,
                        "automated": r.automated,
                        "action_key": r.action_key,
                    }
                    for r in state.remediation_queue
                ],
                "last_test_run": state.last_test_run.isoformat() if state.last_test_run else None,
                "tests_passed": state.tests_passed,
                "tests_failed": state.tests_failed,
                "evidence_chain_valid": state.evidence_chain_valid,
            }

    def to_json(self) -> str:
        """Export dashboard state as JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def export_summary(self) -> dict[str, Any]:
        """Export a concise executive summary."""
        with self._lock:
            state = self._state
            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "overall_compliance_score": state.overall_score,
                "risk_status": "CRITICAL" if state.critical_findings > 0 else (
                    "HIGH" if state.open_findings > 5 else (
                        "MEDIUM" if state.open_findings > 0 else "LOW"
                    )
                ),
                "active_remediations": len(state.remediation_queue),
                "automated_remediations_available": sum(
                    1 for r in state.remediation_queue if r.automated
                ),
                "evidence_chain_status": "VALID" if state.evidence_chain_valid else "COMPROMISED",
                "tests_status": f"{state.tests_passed}/{state.tests_passed + state.tests_failed} passed",
            }
