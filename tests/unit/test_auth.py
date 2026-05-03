"""Tests for neuralmem.auth RBAC module."""

import threading
import time

import pytest

from neuralmem.auth import APIKey, AuthManager, Role


@pytest.fixture
def auth():
    return AuthManager()


@pytest.fixture
def db_auth(tmp_path):
    return AuthManager(db_path=str(tmp_path / "auth.db"))


class TestCreateKey:
    def test_create_key_returns_api_key(self, auth):
        key = auth.create_key(Role.READER, "user1")
        assert isinstance(key, APIKey)
        assert key.role == Role.READER
        assert key.user_id == "user1"
        assert key.is_active is True
        assert isinstance(key.key, str) and len(key.key) == 32

    def test_create_key_unique(self, auth):
        k1 = auth.create_key(Role.READER)
        k2 = auth.create_key(Role.READER)
        assert k1.key != k2.key


class TestValidateKey:
    def test_validate_key_valid(self, auth):
        key = auth.create_key(Role.WRITER)
        result = auth.validate_key(key.key)
        assert result is not None
        assert result.key == key.key
        assert result.role == Role.WRITER

    def test_validate_key_invalid(self, auth):
        assert auth.validate_key("nonexistent") is None

    def test_validate_key_expired(self, auth):
        key = auth.create_key(Role.READER, expires_in_seconds=0.01)
        time.sleep(0.05)
        assert auth.validate_key(key.key) is None

    def test_validate_key_revoked(self, auth):
        key = auth.create_key(Role.READER)
        auth.revoke_key(key.key)
        assert auth.validate_key(key.key) is None


class TestPermissions:
    def test_check_permission_reader_can_read(self, auth):
        key = auth.create_key(Role.READER)
        assert auth.check_permission(key.key, Role.READER) is True

    def test_check_permission_reader_cannot_write(self, auth):
        key = auth.create_key(Role.READER)
        assert auth.check_permission(key.key, Role.WRITER) is False

    def test_check_permission_writer_can_read(self, auth):
        key = auth.create_key(Role.WRITER)
        assert auth.check_permission(key.key, Role.READER) is True

    def test_check_permission_writer_can_write(self, auth):
        key = auth.create_key(Role.WRITER)
        assert auth.check_permission(key.key, Role.WRITER) is True

    def test_check_permission_writer_cannot_admin(self, auth):
        key = auth.create_key(Role.WRITER)
        assert auth.check_permission(key.key, Role.ADMIN) is False

    def test_check_permission_admin_can_all(self, auth):
        key = auth.create_key(Role.ADMIN)
        assert auth.check_permission(key.key, Role.READER) is True
        assert auth.check_permission(key.key, Role.WRITER) is True
        assert auth.check_permission(key.key, Role.ADMIN) is True

    def test_role_hierarchy_ordering(self):
        assert Role.READER < Role.WRITER < Role.ADMIN


class TestRevoke:
    def test_revoke_key(self, auth):
        key = auth.create_key(Role.READER)
        assert auth.revoke_key(key.key) is True
        assert auth.validate_key(key.key) is None

    def test_revoke_nonexistent(self, auth):
        assert auth.revoke_key("nonexistent") is False


class TestListKeys:
    def test_list_keys(self, auth):
        auth.create_key(Role.READER)
        auth.create_key(Role.WRITER)
        keys = auth.list_keys()
        assert len(keys) == 2


class TestRotateKey:
    def test_rotate_key(self, auth):
        old = auth.create_key(Role.WRITER, "user1")
        new = auth.rotate_key(old.key)
        assert new is not None
        assert new.key != old.key
        assert new.role == Role.WRITER
        assert new.user_id == "user1"
        assert auth.validate_key(old.key) is None
        assert auth.validate_key(new.key) is not None

    def test_rotate_nonexistent(self, auth):
        assert auth.rotate_key("nonexistent") is None


class TestCleanup:
    def test_cleanup_expired(self, auth):
        auth.create_key(Role.READER, expires_in_seconds=0.01)
        time.sleep(0.05)
        auth.create_key(Role.READER)  # no expiry
        assert auth.cleanup_expired() == 1

    def test_cleanup_preserves_active(self, auth):
        key = auth.create_key(Role.READER)
        auth.cleanup_expired()
        assert auth.validate_key(key.key) is not None


class TestSQLitePersistence:
    def test_sqlite_persistence(self, tmp_path):
        db_path = str(tmp_path / "auth.db")
        mgr = AuthManager(db_path=db_path)
        key = mgr.create_key(Role.ADMIN, "admin1")
        key_str = key.key

        # Re-open and verify
        mgr2 = AuthManager(db_path=db_path)
        loaded = mgr2.validate_key(key_str)
        assert loaded is not None
        assert loaded.role == Role.ADMIN
        assert loaded.user_id == "admin1"


class TestConcurrency:
    def test_concurrent_access(self):
        auth = AuthManager()
        results: list[APIKey] = []
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(20):
                    key = auth.create_key(Role.READER)
                    auth.validate_key(key.key)
                    results.append(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 80
        assert len(auth.list_keys()) == 80
