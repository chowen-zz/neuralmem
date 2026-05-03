"""Tests for the Evidence Ledger module (20+ tests)."""

from __future__ import annotations

import threading
import time

import pytest

from neuralmem.ledger import EvidenceLedger, FeedbackType, RetrievalRecord

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def mem_ledger() -> EvidenceLedger:
    """In-memory ledger (no db_path)."""
    return EvidenceLedger()


@pytest.fixture
def db_ledger(tmp_path) -> EvidenceLedger:
    """SQLite-backed ledger using a temporary directory."""
    return EvidenceLedger(db_path=str(tmp_path / "ledger.db"))


# ------------------------------------------------------------------ #
# RetrievalRecord / FeedbackType basics
# ------------------------------------------------------------------ #


class TestTypes:
    def test_record_creation(self):
        """record() returns RetrievalRecord with correct fields."""
        ledger = EvidenceLedger()
        rec = ledger.record(
            query="what is AI?",
            retrieved_ids=["id1", "id2"],
            retrieved_scores=[0.9, 0.7],
            composed_context="AI is …",
            latency_ms=12.5,
        )
        assert isinstance(rec, RetrievalRecord)
        assert rec.query == "what is AI?"
        assert rec.retrieved_ids == ["id1", "id2"]
        assert rec.retrieved_scores == [0.9, 0.7]
        assert rec.composed_context == "AI is …"
        assert rec.latency_ms == 12.5
        assert rec.feedback == FeedbackType.none
        assert rec.feedback_note == ""
        assert isinstance(rec.id, str) and len(rec.id) > 0
        assert rec.timestamp > 0

    def test_record_id_uniqueness(self, mem_ledger: EvidenceLedger):
        """All record IDs are unique even when many are created quickly."""
        ids = set()
        for i in range(200):
            rec = mem_ledger.record(
                query=f"q{i}",
                retrieved_ids=[f"id{i}"],
                retrieved_scores=[0.5],
            )
            ids.add(rec.id)
        assert len(ids) == 200

    def test_feedback_type_values(self):
        """FeedbackType exposes the expected string values."""
        assert FeedbackType.useful.value == "useful"
        assert FeedbackType.wrong.value == "wrong"
        assert FeedbackType.outdated.value == "outdated"
        assert FeedbackType.sensitive.value == "sensitive"
        assert FeedbackType.none.value == "none"


# ------------------------------------------------------------------ #
# In-memory mode
# ------------------------------------------------------------------ #


class TestInMemoryMode:
    def test_in_memory_mode(self, mem_ledger: EvidenceLedger):
        """No db_path works correctly — store and retrieve."""
        rec = mem_ledger.record(
            query="hello",
            retrieved_ids=["a"],
            retrieved_scores=[1.0],
        )
        assert mem_ledger.get_record(rec.id) is not None

    def test_record_persistence(self, mem_ledger: EvidenceLedger):
        """After record(), list_records() returns it."""
        rec = mem_ledger.record(
            query="persist me",
            retrieved_ids=["x"],
            retrieved_scores=[0.8],
        )
        records = mem_ledger.list_records()
        assert len(records) == 1
        assert records[0].id == rec.id


# ------------------------------------------------------------------ #
# Feedback
# ------------------------------------------------------------------ #


class TestFeedback:
    def test_feedback_update(self, mem_ledger: EvidenceLedger):
        """feedback() updates the record's feedback field."""
        rec = mem_ledger.record(
            query="q", retrieved_ids=["a"], retrieved_scores=[0.5]
        )
        mem_ledger.feedback(rec.id, FeedbackType.useful, note="helpful")
        updated = mem_ledger.get_record(rec.id)
        assert updated is not None
        assert updated.feedback == FeedbackType.useful
        assert updated.feedback_note == "helpful"

    def test_feedback_invalid_id(self, mem_ledger: EvidenceLedger):
        """feedback() with bad id raises ValueError."""
        with pytest.raises(ValueError, match="Record not found"):
            mem_ledger.feedback("nonexistent-id", FeedbackType.wrong)

    def test_feedback_note(self, mem_ledger: EvidenceLedger):
        """Feedback note is stored and retrievable."""
        rec = mem_ledger.record(
            query="note test", retrieved_ids=["a"], retrieved_scores=[0.5]
        )
        note_text = "This result was completely irrelevant to the query."
        mem_ledger.feedback(rec.id, FeedbackType.wrong, note=note_text)
        updated = mem_ledger.get_record(rec.id)
        assert updated is not None
        assert updated.feedback_note == note_text


# ------------------------------------------------------------------ #
# List / pagination
# ------------------------------------------------------------------ #


class TestListRecords:
    def test_list_records_order(self, mem_ledger: EvidenceLedger):
        """Records are ordered by timestamp descending."""
        r1 = mem_ledger.record(
            query="first", retrieved_ids=["a"], retrieved_scores=[0.5]
        )
        time.sleep(0.01)
        r2 = mem_ledger.record(
            query="second", retrieved_ids=["b"], retrieved_scores=[0.6]
        )
        records = mem_ledger.list_records()
        assert len(records) == 2
        # Most recent first
        assert records[0].id == r2.id
        assert records[1].id == r1.id

    def test_list_records_pagination(self, mem_ledger: EvidenceLedger):
        """limit/offset work correctly."""
        for i in range(10):
            mem_ledger.record(
                query=f"q{i}",
                retrieved_ids=[f"id{i}"],
                retrieved_scores=[float(i) / 10],
            )
        page1 = mem_ledger.list_records(limit=3, offset=0)
        assert len(page1) == 3
        page2 = mem_ledger.list_records(limit=3, offset=3)
        assert len(page2) == 3
        # No overlap
        p1_ids = {r.id for r in page1}
        p2_ids = {r.id for r in page2}
        assert p1_ids.isdisjoint(p2_ids)

    def test_list_records_empty(self, mem_ledger: EvidenceLedger):
        """Empty ledger returns empty list."""
        assert mem_ledger.list_records() == []


# ------------------------------------------------------------------ #
# Stats
# ------------------------------------------------------------------ #


class TestStats:
    def test_stats_empty(self, mem_ledger: EvidenceLedger):
        """stats() returns zeros for empty ledger."""
        s = mem_ledger.stats()
        assert s["total_records"] == 0
        assert s["avg_latency_ms"] == 0.0
        for ft in FeedbackType:
            assert s["feedback_counts"][ft] == 0

    def test_stats_with_records(self, mem_ledger: EvidenceLedger):
        """stats() counts correctly."""
        for i in range(5):
            mem_ledger.record(
                query=f"q{i}",
                retrieved_ids=[f"id{i}"],
                retrieved_scores=[0.5],
                latency_ms=10.0,
            )
        s = mem_ledger.stats()
        assert s["total_records"] == 5

    def test_stats_feedback_counts(self, mem_ledger: EvidenceLedger):
        """stats() groups by feedback type."""
        recs = []
        for i in range(4):
            rec = mem_ledger.record(
                query=f"q{i}",
                retrieved_ids=[f"id{i}"],
                retrieved_scores=[0.5],
            )
            recs.append(rec)
        mem_ledger.feedback(recs[0].id, FeedbackType.useful)
        mem_ledger.feedback(recs[1].id, FeedbackType.useful)
        mem_ledger.feedback(recs[2].id, FeedbackType.wrong)
        # recs[3] stays FeedbackType.none
        s = mem_ledger.stats()
        assert s["feedback_counts"][FeedbackType.useful] == 2
        assert s["feedback_counts"][FeedbackType.wrong] == 1
        assert s["feedback_counts"][FeedbackType.none] == 1
        assert s["feedback_counts"][FeedbackType.outdated] == 0

    def test_stats_avg_latency(self, mem_ledger: EvidenceLedger):
        """Average latency is calculated correctly."""
        mem_ledger.record(
            query="a", retrieved_ids=["1"], retrieved_scores=[1.0], latency_ms=10.0
        )
        mem_ledger.record(
            query="b", retrieved_ids=["2"], retrieved_scores=[1.0], latency_ms=20.0
        )
        mem_ledger.record(
            query="c", retrieved_ids=["3"], retrieved_scores=[1.0], latency_ms=30.0
        )
        s = mem_ledger.stats()
        assert abs(s["avg_latency_ms"] - 20.0) < 1e-9


# ------------------------------------------------------------------ #
# Field preservation
# ------------------------------------------------------------------ #


class TestFieldPreservation:
    def test_record_latency(self, mem_ledger: EvidenceLedger):
        """latency_ms field is preserved."""
        rec = mem_ledger.record(
            query="q",
            retrieved_ids=["a"],
            retrieved_scores=[0.5],
            latency_ms=42.7,
        )
        fetched = mem_ledger.get_record(rec.id)
        assert fetched is not None
        assert fetched.latency_ms == pytest.approx(42.7)

    def test_composed_context(self, mem_ledger: EvidenceLedger):
        """Long context stored correctly."""
        long_ctx = "x" * 10_000
        rec = mem_ledger.record(
            query="q",
            retrieved_ids=["a"],
            retrieved_scores=[0.5],
            composed_context=long_ctx,
        )
        fetched = mem_ledger.get_record(rec.id)
        assert fetched is not None
        assert fetched.composed_context == long_ctx

    def test_retrieved_scores_preserved(self, mem_ledger: EvidenceLedger):
        """Float scores round-trip correctly."""
        scores = [0.123456789, 0.999999999, 0.000000001, 1.0, 0.0]
        rec = mem_ledger.record(
            query="q",
            retrieved_ids=[f"id{i}" for i in range(len(scores))],
            retrieved_scores=scores,
        )
        fetched = mem_ledger.get_record(rec.id)
        assert fetched is not None
        for orig, got in zip(scores, fetched.retrieved_scores):
            assert got == pytest.approx(orig, abs=1e-9)

    def test_multiple_records(self, mem_ledger: EvidenceLedger):
        """Store and retrieve many records."""
        n = 50
        for i in range(n):
            mem_ledger.record(
                query=f"q{i}",
                retrieved_ids=[f"id{i}"],
                retrieved_scores=[float(i) / n],
                latency_ms=float(i),
            )
        all_recs = mem_ledger.list_records(limit=n)
        assert len(all_recs) == n


# ------------------------------------------------------------------ #
# SQLite persistence
# ------------------------------------------------------------------ #


class TestSQLitePersistence:
    def test_sqlite_persistence(self, tmp_path):
        """Records survive re-open."""
        db_file = str(tmp_path / "ledger.db")
        ledger = EvidenceLedger(db_path=db_file)
        rec = ledger.record(
            query="persist",
            retrieved_ids=["a", "b"],
            retrieved_scores=[0.9, 0.8],
            composed_context="some context",
            latency_ms=5.5,
        )
        rec_id = rec.id

        # Re-open the database
        ledger2 = EvidenceLedger(db_path=db_file)
        fetched = ledger2.get_record(rec_id)
        assert fetched is not None
        assert fetched.query == "persist"
        assert fetched.retrieved_ids == ["a", "b"]
        assert fetched.composed_context == "some context"
        assert fetched.latency_ms == pytest.approx(5.5)

    def test_sqlite_feedback_persists(self, tmp_path):
        """Feedback survives re-open."""
        db_file = str(tmp_path / "ledger.db")
        ledger = EvidenceLedger(db_path=db_file)
        rec = ledger.record(
            query="q", retrieved_ids=["a"], retrieved_scores=[0.5]
        )
        ledger.feedback(rec.id, FeedbackType.outdated, note="old data")
        del ledger

        ledger2 = EvidenceLedger(db_path=db_file)
        fetched = ledger2.get_record(rec.id)
        assert fetched is not None
        assert fetched.feedback == FeedbackType.outdated
        assert fetched.feedback_note == "old data"


# ------------------------------------------------------------------ #
# Concurrency
# ------------------------------------------------------------------ #


class TestConcurrency:
    def test_concurrent_records(self):
        """Thread safety: multiple threads writing simultaneously."""
        ledger = EvidenceLedger()  # in-memory
        n_threads = 10
        records_per_thread = 20
        barrier = threading.Barrier(n_threads)
        created: list[list[str]] = [[] for _ in range(n_threads)]

        def worker(tid: int) -> None:
            barrier.wait()
            for i in range(records_per_thread):
                rec = ledger.record(
                    query=f"t{tid}-q{i}",
                    retrieved_ids=[f"id-{tid}-{i}"],
                    retrieved_scores=[0.5],
                    latency_ms=float(tid + i),
                )
                created[tid].append(rec.id)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = n_threads * records_per_thread
        assert ledger.stats()["total_records"] == total

        # All IDs must be unique
        all_ids = [rid for group in created for rid in group]
        assert len(set(all_ids)) == total
