"""Tests for connection_pool module."""
from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from neuralmem.production.connection_pool import (
    ConnectionPool,
    SQLiteConnectionPool,
)

# ------------------------------------------------------------------ helpers
_factory_counter = 0


def _noop_factory():
    """Return a simple object that has a close() method."""
    global _factory_counter
    _factory_counter += 1

    class _Conn:
        def __init__(self, cid: int) -> None:
            self._cid = cid

        def close(self) -> None:
            pass

    return _Conn(_factory_counter)


def _counter_factory():
    """Factory that returns objects with an id."""
    _counter_factory.count += 1
    c = _noop_factory()
    c.id = _counter_factory.count
    return c


_counter_factory.count = 0


def _bad_factory():
    """Connection whose health_check always fails."""
    class _Bad:
        def close(self):
            pass

    return _Bad()


# ------------------------------------------------------------------- tests
class TestConnectionPool:
    def test_create_with_defaults(self):
        pool = ConnectionPool(_noop_factory)
        s = pool.stats()
        assert s["total"] == 2
        assert s["available"] == 2
        pool.close_all()

    def test_acquire_release(self):
        pool = ConnectionPool(_noop_factory, min_size=1, max_size=3)
        conn = pool.acquire()
        assert conn is not None
        s = pool.stats()
        assert s["in_use"] == 1
        pool.release(conn)
        s = pool.stats()
        assert s["in_use"] == 0
        pool.close_all()

    def test_acquire_creates_up_to_max(self):
        _counter = [0]

        def factory():
            _counter[0] += 1
            c = _noop_factory()
            c.uid = _counter[0]
            return c

        pool = ConnectionPool(
            factory, min_size=1, max_size=3
        )
        c1 = pool.acquire()
        c2 = pool.acquire()
        c3 = pool.acquire()
        assert c1 is not c2
        assert c2 is not c3
        s = pool.stats()
        assert s["total"] == 3
        assert s["in_use"] == 3
        pool.release(c1)
        pool.release(c2)
        pool.release(c3)
        pool.close_all()

    def test_acquire_timeout_on_exhaustion(self):
        pool = ConnectionPool(
            _noop_factory, min_size=0, max_size=1
        )
        pool.acquire()
        with pytest.raises(TimeoutError):
            pool.acquire(timeout=0.1)
        pool.close_all()

    def test_release_returns_conn_to_pool(self):
        pool = ConnectionPool(
            _noop_factory, min_size=0, max_size=1
        )
        conn = pool.acquire()
        pool.release(conn)
        conn2 = pool.acquire()
        assert conn is conn2
        pool.release(conn2)
        pool.close_all()

    def test_health_check_reports(self):
        def hc(conn):
            return True

        pool = ConnectionPool(
            _noop_factory,
            min_size=2,
            max_size=4,
            health_check=hc,
        )
        result = pool.health_check()
        assert result["healthy"] >= 2
        pool.close_all()

    def test_health_check_removes_dead(self):
        def hc(conn):
            return False

        pool = ConnectionPool(
            _noop_factory,
            min_size=1,
            max_size=3,
            health_check=hc,
        )
        result = pool.health_check()
        assert result["unhealthy"] >= 1
        # After health check, pool should recreate min_size
        s = pool.stats()
        assert s["available"] >= 1
        pool.close_all()

    def test_close_all(self):
        pool = ConnectionPool(_noop_factory, min_size=2)
        count = pool.close_all()
        assert count >= 2
        with pytest.raises(RuntimeError, match="closed"):
            pool.acquire()

    def test_stats_structure(self):
        pool = ConnectionPool(_noop_factory, min_size=1)
        s = pool.stats()
        for key in (
            "total",
            "available",
            "in_use",
            "created",
            "acquired",
            "released",
            "health_checks",
            "health_failures",
            "timeouts",
        ):
            assert key in s
        pool.close_all()

    def test_thread_safety_acquire_release(self):
        pool = ConnectionPool(
            _noop_factory, min_size=2, max_size=5
        )
        errors: list[Exception] = []

        def worker():
            try:
                conn = pool.acquire(timeout=5.0)
                time.sleep(0.01)
                pool.release(conn)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        pool.close_all()

    def test_min_size_validation(self):
        with pytest.raises(ValueError):
            ConnectionPool(_noop_factory, min_size=-1)

    def test_max_size_validation(self):
        with pytest.raises(ValueError):
            ConnectionPool(_noop_factory, max_size=0)

    def test_min_gt_max_validation(self):
        with pytest.raises(ValueError):
            ConnectionPool(
                _noop_factory, min_size=5, max_size=2
            )

    def test_idle_timeout_reaps(self):
        pool = ConnectionPool(
            _noop_factory,
            min_size=0,
            max_size=5,
            idle_timeout=0.05,
        )
        conn = pool.acquire()
        pool.release(conn)
        time.sleep(0.1)
        # acquire triggers reaping
        conn2 = pool.acquire()
        assert conn2 is not conn
        pool.release(conn2)
        pool.close_all()


class TestSQLiteConnectionPool:
    def test_basic_acquire_release(self):
        pool = SQLiteConnectionPool(":memory:")
        conn = pool.acquire()
        assert isinstance(conn, sqlite3.Connection)
        cursor = conn.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1
        pool.release(conn)
        pool.close_all()

    def test_wal_mode_enabled(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pool = SQLiteConnectionPool(db_path)
        conn = pool.acquire()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        pool.release(conn)
        pool.close_all()

    def test_busy_timeout_set(self):
        pool = SQLiteConnectionPool(
            ":memory:", busy_timeout=7000
        )
        conn = pool.acquire()
        val = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert val == 7000
        pool.release(conn)
        pool.close_all()

    def test_custom_pragmas(self):
        pool = SQLiteConnectionPool(
            ":memory:",
            pragmas={"cache_size": "2000"},
        )
        conn = pool.acquire()
        val = conn.execute("PRAGMA cache_size").fetchone()[0]
        assert val == 2000
        pool.release(conn)
        pool.close_all()
