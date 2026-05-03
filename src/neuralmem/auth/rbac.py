"""API Key authentication and role-based access control (RBAC)."""

from __future__ import annotations

import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from enum import IntEnum


class Role(IntEnum):
    """Permission hierarchy: reader < writer < admin."""
    READER = 1
    WRITER = 2
    ADMIN = 3


@dataclass
class APIKey:
    key: str
    role: Role
    user_id: str
    created_at: float
    expires_at: float | None = None
    is_active: bool = True


class AuthManager:
    """Thread-safe API key manager with SQLite persistence."""

    def __init__(self, db_path: str | None = None) -> None:
        self._lock = threading.Lock()
        self._in_memory: dict[str, APIKey] = {}
        self._use_db = db_path is not None
        if self._use_db:
            assert db_path is not None
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS api_keys (
                    key TEXT PRIMARY KEY,
                    role TEXT,
                    user_id TEXT,
                    created_at REAL,
                    expires_at REAL,
                    is_active INTEGER
                )"""
            )
            self._conn.commit()

    def create_key(
        self,
        role: Role,
        user_id: str = "",
        expires_in_seconds: float | None = None,
    ) -> APIKey:
        now = time.time()
        key_str = secrets.token_hex(16)
        expires_at = now + expires_in_seconds if expires_in_seconds is not None else None
        api_key = APIKey(
            key=key_str,
            role=role,
            user_id=user_id,
            created_at=now,
            expires_at=expires_at,
            is_active=True,
        )
        with self._lock:
            if self._use_db:
                self._conn.execute(
                    "INSERT INTO api_keys"
                    " (key, role, user_id, created_at, expires_at, is_active)"
                    " VALUES (?, ?, ?, ?, ?, 1)",
                    (key_str, role.name, user_id, now, expires_at),
                )
                self._conn.commit()
            else:
                self._in_memory[key_str] = api_key
        return api_key

    def validate_key(self, key: str) -> APIKey | None:
        with self._lock:
            if self._use_db:
                row = self._conn.execute(
                    "SELECT key, role, user_id, created_at,"
                    " expires_at, is_active"
                    " FROM api_keys WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    return None
                api_key = APIKey(
                    key=row[0],
                    role=Role[row[1]],
                    user_id=row[2],
                    created_at=row[3],
                    expires_at=row[4],
                    is_active=bool(row[5]),
                )
            else:
                api_key = self._in_memory.get(key)
                if api_key is None:
                    return None

        if not api_key.is_active:
            return None
        if api_key.expires_at is not None and time.time() > api_key.expires_at:
            return None
        return api_key

    def check_permission(self, key: str, required_role: Role) -> bool:
        api_key = self.validate_key(key)
        if api_key is None:
            return False
        return api_key.role >= required_role

    def revoke_key(self, key: str) -> bool:
        with self._lock:
            if self._use_db:
                cur = self._conn.execute(
                    "UPDATE api_keys SET is_active = 0 WHERE key = ? AND is_active = 1",
                    (key,),
                )
                self._conn.commit()
                return cur.rowcount > 0
            else:
                api_key = self._in_memory.get(key)
                if api_key is None or not api_key.is_active:
                    return False
                api_key.is_active = False
                return True

    def list_keys(self) -> list[APIKey]:
        with self._lock:
            if self._use_db:
                rows = self._conn.execute(
                    "SELECT key, role, user_id, created_at, expires_at, is_active FROM api_keys"
                ).fetchall()
                return [
                    APIKey(
                        key=r[0],
                        role=Role[r[1]],
                        user_id=r[2],
                        created_at=r[3],
                        expires_at=r[4],
                        is_active=bool(r[5]),
                    )
                    for r in rows
                ]
            else:
                return list(self._in_memory.values())

    def rotate_key(self, old_key: str) -> APIKey | None:
        with self._lock:
            if self._use_db:
                row = self._conn.execute(
                    "SELECT role, user_id, expires_at, is_active FROM api_keys WHERE key = ?",
                    (old_key,),
                ).fetchone()
                if row is None:
                    return None
                role = Role[row[0]]
                user_id = row[1]
                expires_at = row[2]
                # Compute remaining time
                expires_in = expires_at - time.time() if expires_at is not None else None
            else:
                old = self._in_memory.get(old_key)
                if old is None:
                    return None
                role = old.role
                user_id = old.user_id
                expires_in = old.expires_at - time.time() if old.expires_at is not None else None

        # Revoke old key
        self.revoke_key(old_key)
        # Create new key (don't pass negative expiry)
        return self.create_key(role, user_id, expires_in if expires_in and expires_in > 0 else None)

    def cleanup_expired(self) -> int:
        now = time.time()
        with self._lock:
            if self._use_db:
                cur = self._conn.execute(
                    "UPDATE api_keys SET is_active = 0"
                    " WHERE expires_at IS NOT NULL"
                    " AND expires_at < ? AND is_active = 1",
                    (now,),
                )
                self._conn.commit()
                return cur.rowcount
            else:
                count = 0
                for api_key in self._in_memory.values():
                    if (
                        api_key.is_active
                        and api_key.expires_at is not None
                        and now > api_key.expires_at
                    ):
                        api_key.is_active = False
                        count += 1
                return count
