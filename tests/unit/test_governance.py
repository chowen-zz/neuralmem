"""Tests for NeuralMem governance module — risk, state, audit."""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from neuralmem.core.types import Memory
from neuralmem.governance.audit import AuditEvent, AuditLogger
from neuralmem.governance.risk import RiskLevel, scan_batch, scan_memory
from neuralmem.governance.state import (
    GovernanceState,
    MemoryState,
    StateTransitionError,
)

# ---------------------------------------------------------------------------
# Risk Scanner
# ---------------------------------------------------------------------------


class TestRiskScanner:
    def test_detect_email(self):
        mem = Memory(content="Contact user@example.com for details", user_id="u1")
        findings = scan_memory(mem)
        assert len(findings) >= 1
        pii = [f for f in findings if f.category == "PII" and "Email" in f.description]
        assert len(pii) == 1
        assert pii[0].level == RiskLevel.HIGH

    def test_detect_phone(self):
        mem = Memory(content="Call me at 555-123-4567", user_id="u1")
        findings = scan_memory(mem)
        phone = [f for f in findings if f.category == "PII" and "Phone" in f.description]
        assert len(phone) == 1
        assert phone[0].level == RiskLevel.MEDIUM

    def test_detect_ssn(self):
        mem = Memory(content="SSN: 123-45-6789", user_id="u1")
        findings = scan_memory(mem)
        ssn = [f for f in findings if f.category == "PII" and "SSN" in f.description]
        assert len(ssn) == 1
        assert ssn[0].level == RiskLevel.CRITICAL

    def test_detect_github_token(self):
        mem = Memory(content="token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij", user_id="u1")
        findings = scan_memory(mem)
        api = [f for f in findings if f.category == "API_KEYS"]
        assert len(api) >= 1
        assert api[0].level == RiskLevel.CRITICAL

    def test_detect_openai_key(self):
        mem = Memory(content="Use sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ for API", user_id="u1")
        findings = scan_memory(mem)
        api = [f for f in findings if f.category == "API_KEYS"]
        assert len(api) >= 1

    def test_detect_aws_key(self):
        mem = Memory(content="aws key: AKIAIOSFODNN7EXAMPLE", user_id="u1")
        findings = scan_memory(mem)
        api = [f for f in findings if f.category == "API_KEYS"]
        assert len(api) >= 1

    def test_detect_sensitive_word_default(self):
        mem = Memory(content="The password for the admin account is set", user_id="u1")
        findings = scan_memory(mem)
        sw = [f for f in findings if f.category == "SENSITIVE_WORDS"]
        assert len(sw) >= 1
        assert sw[0].level == RiskLevel.MEDIUM

    def test_detect_sensitive_word_custom(self):
        mem = Memory(content="Project Alpha is top secret", user_id="u1")
        findings = scan_memory(mem, sensitive_words=["alpha"])
        sw = [f for f in findings if f.category == "SENSITIVE_WORDS"]
        assert len(sw) == 1

    def test_no_sensitive_words_when_custom_empty(self):
        mem = Memory(content="The password is secret", user_id="u1")
        findings = scan_memory(mem, sensitive_words=[])
        sw = [f for f in findings if f.category == "SENSITIVE_WORDS"]
        assert len(sw) == 0

    def test_stale_expired_memory(self):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        mem = Memory(content="expired info", user_id="u1", expires_at=past)
        findings = scan_memory(mem)
        stale = [f for f in findings if f.category == "STALE_DATA"]
        assert len(stale) == 1
        assert stale[0].level == RiskLevel.HIGH

    def test_stale_old_high_importance_memory(self):
        old_date = datetime.now(timezone.utc) - timedelta(days=120)
        mem = Memory(
            content="old important memory",
            user_id="u1",
            importance=0.9,
            created_at=old_date,
        )
        findings = scan_memory(mem)
        stale = [f for f in findings if f.category == "STALE_DATA"]
        assert len(stale) == 1
        assert stale[0].level == RiskLevel.MEDIUM

    def test_clean_content_no_findings(self):
        mem = Memory(content="User likes Python programming", user_id="u1")
        findings = scan_memory(mem)
        assert len(findings) == 0

    def test_scan_batch(self):
        mems = [
            Memory(content="user@example.com", user_id="u1"),
            Memory(content="Clean memory", user_id="u1"),
            Memory(content="SSN: 987-65-4321", user_id="u1"),
        ]
        findings = scan_batch(mems)
        assert len(findings) >= 2  # email + SSN


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_draft_to_active(self):
        gs = GovernanceState()
        entry = gs.transition("m1", MemoryState.DRAFT, MemoryState.ACTIVE, "approved")
        assert entry.to_state == MemoryState.ACTIVE

    def test_active_to_review(self):
        gs = GovernanceState()
        entry = gs.transition("m1", MemoryState.ACTIVE, MemoryState.REVIEW, "flagged")
        assert entry.to_state == MemoryState.REVIEW

    def test_active_to_stale(self):
        gs = GovernanceState()
        entry = gs.transition("m1", MemoryState.ACTIVE, MemoryState.STALE, "not accessed")
        assert entry.to_state == MemoryState.STALE

    def test_active_to_superseded(self):
        gs = GovernanceState()
        entry = gs.transition("m1", MemoryState.ACTIVE, MemoryState.SUPERSEDED, "newer version")
        assert entry.to_state == MemoryState.SUPERSEDED

    def test_review_to_active(self):
        gs = GovernanceState()
        gs.transition("m1", MemoryState.ACTIVE, MemoryState.REVIEW, "flagged")
        entry = gs.transition("m1", MemoryState.REVIEW, MemoryState.ACTIVE, "cleared")
        assert entry.to_state == MemoryState.ACTIVE

    def test_review_to_archived(self):
        gs = GovernanceState()
        gs.transition("m1", MemoryState.ACTIVE, MemoryState.REVIEW, "flagged")
        entry = gs.transition("m1", MemoryState.REVIEW, MemoryState.ARCHIVED, "archived")
        assert entry.to_state == MemoryState.ARCHIVED

    def test_stale_to_active(self):
        gs = GovernanceState()
        gs.transition("m1", MemoryState.ACTIVE, MemoryState.STALE, "old")
        entry = gs.transition("m1", MemoryState.STALE, MemoryState.ACTIVE, "refreshed")
        assert entry.to_state == MemoryState.ACTIVE

    def test_stale_to_archived(self):
        gs = GovernanceState()
        gs.transition("m1", MemoryState.ACTIVE, MemoryState.STALE, "old")
        entry = gs.transition("m1", MemoryState.STALE, MemoryState.ARCHIVED, "archived")
        assert entry.to_state == MemoryState.ARCHIVED

    def test_superseded_is_terminal(self):
        gs = GovernanceState()
        gs.transition("m1", MemoryState.ACTIVE, MemoryState.SUPERSEDED, "replaced")
        with pytest.raises(StateTransitionError):
            gs.transition("m1", MemoryState.SUPERSEDED, MemoryState.ACTIVE, "un-replace")

    def test_archived_is_terminal(self):
        gs = GovernanceState()
        gs.transition("m1", MemoryState.ACTIVE, MemoryState.REVIEW, "review")
        gs.transition("m1", MemoryState.REVIEW, MemoryState.ARCHIVED, "done")
        with pytest.raises(StateTransitionError):
            gs.transition("m1", MemoryState.ARCHIVED, MemoryState.ACTIVE, "revive")

    def test_invalid_transition_draft_to_archived(self):
        gs = GovernanceState()
        with pytest.raises(StateTransitionError) as exc_info:
            gs.transition("m1", MemoryState.DRAFT, MemoryState.ARCHIVED, "skip")
        assert "Invalid transition" in str(exc_info.value)

    def test_infer_state_active(self):
        mem = Memory(content="Hello", user_id="u1", is_active=True)
        assert GovernanceState.infer_state(mem) == MemoryState.ACTIVE

    def test_infer_state_superseded_inactive(self):
        mem = Memory(content="Old", user_id="u1", is_active=False)
        assert GovernanceState.infer_state(mem) == MemoryState.SUPERSEDED

    def test_infer_state_superseded_by(self):
        mem = Memory(content="Old", user_id="u1", superseded_by="m2")
        assert GovernanceState.infer_state(mem) == MemoryState.SUPERSEDED

    def test_infer_state_stale_expired(self):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        mem = Memory(content="Expired", user_id="u1", expires_at=past)
        assert GovernanceState.infer_state(mem) == MemoryState.STALE

    def test_history_tracking(self):
        gs = GovernanceState()
        gs.transition("m1", MemoryState.DRAFT, MemoryState.ACTIVE, "created")
        gs.transition("m1", MemoryState.ACTIVE, MemoryState.REVIEW, "flagged")
        history = gs.get_history("m1")
        assert len(history) == 2
        assert history[0].from_state == MemoryState.DRAFT
        assert history[1].to_state == MemoryState.REVIEW

    def test_history_filter(self):
        gs = GovernanceState()
        gs.transition("m1", MemoryState.DRAFT, MemoryState.ACTIVE, "created")
        gs.transition("m2", MemoryState.DRAFT, MemoryState.ACTIVE, "created")
        assert len(gs.get_history("m1")) == 1

    def test_get_state_tracked(self):
        gs = GovernanceState()
        gs.track("m1", MemoryState.ACTIVE)
        assert gs.get_state("m1") == MemoryState.ACTIVE

    def test_get_state_untracked(self):
        gs = GovernanceState()
        assert gs.get_state("unknown") is None


# ---------------------------------------------------------------------------
# Audit Logger
# ---------------------------------------------------------------------------


class TestAuditLogger:
    def test_log_and_query(self):
        logger = AuditLogger()
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="REMEMBER",
            memory_id="m1",
            user_id="u1",
            details={"content": "test"},
        )
        logger.log(event)
        results = logger.query(memory_id="m1")
        assert len(results) == 1
        assert results[0].event_type == "REMEMBER"

    def test_query_filter_event_type(self):
        logger = AuditLogger()
        logger.log(AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="REMEMBER",
            memory_id="m1",
        ))
        logger.log(AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="FORGET",
            memory_id="m1",
        ))
        assert len(logger.query(event_type="REMEMBER")) == 1
        assert len(logger.query(event_type="FORGET")) == 1

    def test_query_filter_risk_level(self):
        logger = AuditLogger()
        logger.log(AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="GOVERNANCE",
            memory_id="m1",
            risk_level=RiskLevel.CRITICAL,
        ))
        logger.log(AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="GOVERNANCE",
            memory_id="m2",
            risk_level=RiskLevel.LOW,
        ))
        assert len(logger.query(risk_level=RiskLevel.CRITICAL)) == 1

    def test_ring_buffer_capacity(self):
        logger = AuditLogger(max_events=5)
        for i in range(10):
            logger.log(AuditEvent(
                timestamp=datetime.now(timezone.utc),
                event_type="REMEMBER",
                memory_id=f"m{i}",
            ))
        assert logger.count == 5
        # Oldest events (m0-m4) should be evicted
        results = logger.query(limit=10)
        ids = {r.memory_id for r in results}
        assert "m0" not in ids
        assert "m9" in ids

    def test_query_user_filter(self):
        logger = AuditLogger()
        logger.log(AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="RECALL",
            memory_id="m1",
            user_id="u1",
        ))
        logger.log(AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="RECALL",
            memory_id="m2",
            user_id="u2",
        ))
        assert len(logger.query(user_id="u1")) == 1

    def test_thread_safety(self):
        logger = AuditLogger(max_events=1000)
        errors: list[Exception] = []

        def worker(start: int, count: int):
            try:
                for i in range(start, start + count):
                    logger.log(AuditEvent(
                        timestamp=datetime.now(timezone.utc),
                        event_type="REMEMBER",
                        memory_id=f"m{i}",
                    ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i * 100, 100)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert logger.count == 1000

    def test_persistence_with_tmp_db(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        logger = AuditLogger(max_events=100, db_path=db_path)
        logger.log(AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="GOVERNANCE",
            memory_id="m1",
            details={"action": "test"},
            risk_level=RiskLevel.HIGH,
        ))

        # Verify via SQLite directly
        import sqlite3
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT event_type, memory_id, risk_level FROM audit_events").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0] == ("GOVERNANCE", "m1", "high")
