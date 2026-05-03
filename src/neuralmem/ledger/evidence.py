"""Evidence Ledger — retrieval audit trail with feedback mechanism."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from neuralmem.ledger.types import FeedbackType, RetrievalRecord


class EvidenceLedger:
    """Tracks every recall() call and supports user feedback.

    Parameters
    ----------
    db_path : str | None
        Path to a SQLite database file.  When *None* the ledger operates
        entirely in-memory (useful for tests and ephemeral workloads).
    """

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._records: list[RetrievalRecord] = []  # in-memory store

        if db_path is not None:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._create_table()
        else:
            self._conn = None  # type: ignore[assignment]

    def _create_table(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_records (
                id               TEXT PRIMARY KEY,
                timestamp        REAL,
                query            TEXT,
                retrieved_ids    TEXT,
                retrieved_scores TEXT,
                composed_context TEXT,
                feedback         TEXT,
                feedback_note    TEXT,
                latency_ms       REAL
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def record(
        self,
        query: str,
        retrieved_ids: list[str],
        retrieved_scores: list[float],
        composed_context: str = "",
        latency_ms: float = 0.0,
    ) -> RetrievalRecord:
        """Create a new retrieval record, store it, and return it."""
        rec = RetrievalRecord(
            query=query,
            retrieved_ids=list(retrieved_ids),
            retrieved_scores=list(retrieved_scores),
            composed_context=composed_context,
            latency_ms=latency_ms,
        )

        with self._lock:
            if self._conn is not None:
                self._conn.execute(
                    """
                    INSERT INTO retrieval_records
                        (id, timestamp, query, retrieved_ids, retrieved_scores,
                         composed_context, feedback, feedback_note, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rec.id,
                        rec.timestamp,
                        rec.query,
                        json.dumps(rec.retrieved_ids),
                        json.dumps(rec.retrieved_scores),
                        rec.composed_context,
                        rec.feedback.value,
                        rec.feedback_note,
                        rec.latency_ms,
                    ),
                )
                self._conn.commit()
            else:
                self._records.append(rec)

        return rec

    def feedback(
        self,
        record_id: str,
        feedback: FeedbackType,
        note: str = "",
    ) -> None:
        """Attach feedback to an existing record.

        Raises
        ------
        ValueError
            If *record_id* does not exist.
        """
        with self._lock:
            if self._conn is not None:
                cur = self._conn.execute(
                    "SELECT id FROM retrieval_records WHERE id = ?",
                    (record_id,),
                )
                if cur.fetchone() is None:
                    raise ValueError(f"Record not found: {record_id}")
                self._conn.execute(
                    """
                    UPDATE retrieval_records
                       SET feedback = ?, feedback_note = ?
                     WHERE id = ?
                    """,
                    (feedback.value, note, record_id),
                )
                self._conn.commit()
            else:
                for rec in self._records:
                    if rec.id == record_id:
                        rec.feedback = feedback
                        rec.feedback_note = note
                        return
                raise ValueError(f"Record not found: {record_id}")

    def get_record(self, record_id: str) -> RetrievalRecord | None:
        """Return a single record by ID, or *None*."""
        with self._lock:
            if self._conn is not None:
                cur = self._conn.execute(
                    """
                    SELECT id, timestamp, query, retrieved_ids, retrieved_scores,
                           composed_context, feedback, feedback_note, latency_ms
                      FROM retrieval_records
                     WHERE id = ?
                    """,
                    (record_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_record(row)
            else:
                for rec in self._records:
                    if rec.id == record_id:
                        return rec
                return None

    def list_records(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RetrievalRecord]:
        """Return records ordered by timestamp descending."""
        with self._lock:
            if self._conn is not None:
                cur = self._conn.execute(
                    """
                    SELECT id, timestamp, query, retrieved_ids, retrieved_scores,
                           composed_context, feedback, feedback_note, latency_ms
                      FROM retrieval_records
                     ORDER BY timestamp DESC
                     LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )
                return [self._row_to_record(row) for row in cur.fetchall()]
            else:
                sorted_recs = sorted(
                    self._records, key=lambda r: r.timestamp, reverse=True
                )
                return sorted_recs[offset : offset + limit]

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics for the ledger.

        Returns
        -------
        dict
            ``total_records``: int
            ``feedback_counts``: dict[FeedbackType, int]
            ``avg_latency_ms``: float
        """
        with self._lock:
            if self._conn is not None:
                cur = self._conn.execute(
                    "SELECT COUNT(*) FROM retrieval_records"
                )
                total = cur.fetchone()[0]

                cur = self._conn.execute(
                    """
                    SELECT feedback, COUNT(*)
                      FROM retrieval_records
                     GROUP BY feedback
                    """
                )
                fb_counts: dict[str, int] = {}
                for fb, cnt in cur.fetchall():
                    fb_counts[fb] = cnt

                cur = self._conn.execute(
                    "SELECT COALESCE(AVG(latency_ms), 0.0) FROM retrieval_records"
                )
                avg_latency = cur.fetchone()[0]
            else:
                total = len(self._records)
                fb_counts: dict[str, int] = {}  # type: ignore[no-redef]
                latency_sum = 0.0
                for rec in self._records:
                    key = rec.feedback.value
                    fb_counts[key] = fb_counts.get(key, 0) + 1
                    latency_sum += rec.latency_ms
                avg_latency = latency_sum / total if total else 0.0

            # Normalise keys to FeedbackType
            feedback_counts: dict[FeedbackType, int] = {}
            for ft in FeedbackType:
                feedback_counts[ft] = fb_counts.get(ft.value, 0)

            return {
                "total_records": total,
                "feedback_counts": feedback_counts,
                "avg_latency_ms": float(avg_latency),
            }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _row_to_record(row: tuple[Any, ...]) -> RetrievalRecord:
        """Convert a SQLite row tuple back to a RetrievalRecord."""
        return RetrievalRecord(
            id=row[0],
            timestamp=row[1],
            query=row[2],
            retrieved_ids=json.loads(row[3]),
            retrieved_scores=json.loads(row[4]),
            composed_context=row[5],
            feedback=FeedbackType(row[6]),
            feedback_note=row[7],
            latency_ms=row[8],
        )
