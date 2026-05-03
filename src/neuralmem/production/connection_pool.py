"""Generic and SQLite-specific connection pools."""
from __future__ import annotations

import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class PoolStats:
    """Connection pool statistics."""

    created: int = 0
    acquired: int = 0
    released: int = 0
    health_checks: int = 0
    health_failures: int = 0
    timeouts: int = 0


@dataclass
class _PooledConn:
    """Wrapper around a raw connection with metadata."""

    conn: Any
    created_at: float
    last_used: float
    in_use: bool = False


class ConnectionPool:
    """Generic thread-safe connection pool.

    Parameters
    ----------
    factory : callable
        Zero-arg factory that returns a new connection object.
    min_size : int
        Minimum idle connections to maintain.
    max_size : int
        Maximum total connections.
    idle_timeout : float
        Seconds before an idle connection is reaped.
    max_lifetime : float
        Maximum age (seconds) of any connection before recycling.
    health_check : callable or None
        Optional one-arg callable(conn) -> bool to validate a conn.
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        min_size: int = 2,
        max_size: int = 10,
        idle_timeout: float = 300.0,
        max_lifetime: float = 3600.0,
        health_check: Callable[[Any], bool] | None = None,
    ) -> None:
        if min_size < 0:
            raise ValueError("min_size must be >= 0")
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if min_size > max_size:
            raise ValueError("min_size must be <= max_size")

        self._factory = factory
        self._min_size = min_size
        self._max_size = max_size
        self._idle_timeout = idle_timeout
        self._max_lifetime = max_lifetime
        self._health_check = health_check
        self._lock = threading.Lock()
        self._available: deque[_PooledConn] = deque()
        self._all: list[_PooledConn] = []
        self._stats = PoolStats()
        self._closed = False

        # Pre-create minimum connections
        for _ in range(min_size):
            self._create_conn()

    # -- internal helpers --------------------------------------------------

    def _create_conn(self) -> _PooledConn:
        conn = self._factory()
        now = time.monotonic()
        pooled = _PooledConn(
            conn=conn, created_at=now, last_used=now
        )
        self._all.append(pooled)
        self._available.append(pooled)
        self._stats.created += 1
        return pooled

    def _is_alive(self, pooled: _PooledConn) -> bool:
        if self._health_check is None:
            return True
        try:
            return bool(self._health_check(pooled.conn))
        except Exception:
            return False

    def _reap_idle(self, now: float) -> None:
        """Remove idle/expired connections below min_size."""
        kept: deque[_PooledConn] = deque()
        while self._available:
            pooled = self._available.popleft()
            age = now - pooled.created_at
            idle = now - pooled.last_used
            if (
                idle > self._idle_timeout
                or age > self._max_lifetime
            ):
                self._all.remove(pooled)
                self._close_single(pooled)
            else:
                kept.append(pooled)
        self._available = kept

    def _close_single(self, pooled: _PooledConn) -> None:
        try:
            pooled.conn.close()
        except Exception:
            pass

    # -- public API --------------------------------------------------------

    def acquire(self, timeout: float = 30.0) -> Any:
        """Acquire a connection, blocking up to *timeout* seconds."""
        if self._closed:
            raise RuntimeError("Pool is closed")
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                self._reap_idle(now)

                # Try to grab an idle connection
                while self._available:
                    pooled = self._available.popleft()
                    if self._is_alive(pooled):
                        pooled.in_use = True
                        pooled.last_used = now
                        self._stats.acquired += 1
                        return pooled.conn
                    # unhealthy — discard
                    self._all.remove(pooled)
                    self._close_single(pooled)

            # Create new if under max
            if len(self._all) < self._max_size:
                pooled = self._create_conn()
                self._available.remove(pooled)
                pooled.in_use = True
                self._stats.acquired += 1
                return pooled.conn

            # Pool exhausted — wait and retry
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._stats.timeouts += 1
                raise TimeoutError(
                    "Could not acquire connection within timeout"
                )
            time.sleep(min(0.05, remaining))

    def release(self, conn: Any) -> None:
        """Return a connection to the pool."""
        with self._lock:
            for pooled in self._all:
                if pooled.conn is conn:
                    pooled.in_use = False
                    pooled.last_used = time.monotonic()
                    self._available.append(pooled)
                    self._stats.released += 1
                    return
            # conn not found — already removed, ignore

    def health_check(self) -> dict[str, Any]:
        """Run health checks on all idle connections."""
        results = {"healthy": 0, "unhealthy": 0}
        with self._lock:
            now = time.monotonic()
            kept: deque[_PooledConn] = deque()
            while self._available:
                pooled = self._available.popleft()
                self._stats.health_checks += 1
                if (
                    self._is_alive(pooled)
                    and (now - pooled.created_at)
                    <= self._max_lifetime
                ):
                    kept.append(pooled)
                    results["healthy"] += 1
                else:
                    self._all.remove(pooled)
                    self._close_single(pooled)
                    self._stats.health_failures += 1
                    results["unhealthy"] += 1
            self._available = kept
            # Ensure min_size
            deficit = self._min_size - len(self._available)
            for _ in range(
                min(deficit, self._max_size - len(self._all))
            ):
                self._create_conn()
                results["healthy"] += 1
        return results

    def close_all(self) -> int:
        """Close every connection. Returns count closed."""
        with self._lock:
            count = len(self._all)
            for pooled in self._all:
                self._close_single(pooled)
            self._all.clear()
            self._available.clear()
            self._closed = True
            return count

    def stats(self) -> dict[str, int]:
        """Return a snapshot of pool statistics."""
        with self._lock:
            return {
                "total": len(self._all),
                "available": len(self._available),
                "in_use": len(self._all) - len(self._available),
                "created": self._stats.created,
                "acquired": self._stats.acquired,
                "released": self._stats.released,
                "health_checks": self._stats.health_checks,
                "health_failures": self._stats.health_failures,
                "timeouts": self._stats.timeouts,
            }


class SQLiteConnectionPool(ConnectionPool):
    """SQLite-specific connection pool with WAL mode and busy_timeout.

    Parameters
    ----------
    database : str
        Path to the SQLite database file (or ":memory:").
    min_size : int
        Minimum idle connections.
    max_size : int
        Maximum total connections.
    idle_timeout : float
        Seconds an idle connection survives.
    max_lifetime : float
        Maximum age of a connection.
    busy_timeout : int
        SQLite busy_timeout in milliseconds.
    pragmas : dict or None
        Additional PRAGMA statements to execute on each connection.
    """

    def __init__(
        self,
        database: str = ":memory:",
        *,
        min_size: int = 1,
        max_size: int = 5,
        idle_timeout: float = 300.0,
        max_lifetime: float = 3600.0,
        busy_timeout: int = 5000,
        pragmas: dict[str, str] | None = None,
    ) -> None:
        self._database = database
        self._busy_timeout = busy_timeout
        self._extra_pragmas = pragmas or {}

        def _factory() -> sqlite3.Connection:
            conn = sqlite3.connect(
                database, check_same_thread=False
            )
            conn.execute(
                f"PRAGMA busy_timeout = {busy_timeout}"
            )
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            for key, val in self._extra_pragmas.items():
                conn.execute(f"PRAGMA {key} = {val}")
            return conn

        def _hc(conn: sqlite3.Connection) -> bool:
            try:
                conn.execute("SELECT 1")
                return True
            except Exception:
                return False

        super().__init__(
            _factory,
            min_size=min_size,
            max_size=max_size,
            idle_timeout=idle_timeout,
            max_lifetime=max_lifetime,
            health_check=_hc,
        )
