"""Tests for TenantManager and multi-tenant isolation."""
from __future__ import annotations

import threading
import time

import pytest

from neuralmem.tenancy.manager import TenantContext, TenantManager
from neuralmem.tenancy.models import TenantConfig

# ---------------------------------------------------------------------------
# TenantConfig model
# ---------------------------------------------------------------------------

class TestTenantConfig:
    def test_defaults(self):
        cfg = TenantConfig(tenant_id="acme")
        assert cfg.tenant_id == "acme"
        assert cfg.max_memories == 10_000
        assert cfg.max_queries_per_minute == 60
        assert cfg.allowed_memory_types == ()
        assert cfg.storage_namespace == ""

    def test_frozen(self):
        cfg = TenantConfig(tenant_id="acme")
        with pytest.raises(Exception):
            cfg.tenant_id = "other"  # type: ignore[misc]

    def test_custom_values(self):
        cfg = TenantConfig(
            tenant_id="acme",
            max_memories=500,
            max_queries_per_minute=30,
            allowed_memory_types=("fact", "preference"),
            storage_namespace="ns:acme:",
        )
        assert cfg.max_memories == 500
        assert cfg.allowed_memory_types == ("fact", "preference")
        assert cfg.storage_namespace == "ns:acme:"

    def test_min_validation(self):
        with pytest.raises(Exception):
            TenantConfig(tenant_id="x", max_memories=0)


# ---------------------------------------------------------------------------
# TenantContext thread-local
# ---------------------------------------------------------------------------

class TestTenantContext:
    def test_default_none(self):
        TenantContext.clear()
        assert TenantContext.get_tenant_id() is None

    def test_set_and_get(self):
        TenantContext.set_tenant_id("t1")
        assert TenantContext.get_tenant_id() == "t1"
        TenantContext.clear()

    def test_clear(self):
        TenantContext.set_tenant_id("t1")
        TenantContext.clear()
        assert TenantContext.get_tenant_id() is None

    def test_namespace(self):
        TenantContext.set_namespace("ns:")
        assert TenantContext.get_namespace() == "ns:"
        TenantContext.clear()

    def test_thread_isolation(self):
        TenantContext.clear()
        results = {}

        def worker(tid):
            TenantContext.set_tenant_id(tid)
            time.sleep(0.05)
            results[tid] = TenantContext.get_tenant_id()

        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert results["a"] == "a"
        assert results["b"] == "b"


# ---------------------------------------------------------------------------
# TenantManager CRUD
# ---------------------------------------------------------------------------

class TestTenantManagerCRUD:
    def test_create_tenant(self):
        mgr = TenantManager()
        cfg = TenantConfig(tenant_id="acme")
        result = mgr.create_tenant(cfg)
        assert result.tenant_id == "acme"

    def test_create_auto_generates_namespace(self):
        mgr = TenantManager()
        cfg = TenantConfig(tenant_id="acme")
        result = mgr.create_tenant(cfg)
        assert result.storage_namespace == "tenant:acme:"

    def test_create_preserves_explicit_namespace(self):
        mgr = TenantManager()
        cfg = TenantConfig(tenant_id="acme", storage_namespace="custom:")
        result = mgr.create_tenant(cfg)
        assert result.storage_namespace == "custom:"

    def test_create_duplicate_raises(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        with pytest.raises(ValueError, match="already exists"):
            mgr.create_tenant(TenantConfig(tenant_id="acme"))

    def test_get_tenant(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        cfg = mgr.get_tenant("acme")
        assert cfg.tenant_id == "acme"

    def test_get_tenant_not_found(self):
        mgr = TenantManager()
        with pytest.raises(KeyError, match="not found"):
            mgr.get_tenant("nonexistent")

    def test_list_tenants(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="a"))
        mgr.create_tenant(TenantConfig(tenant_id="b"))
        tenants = mgr.list_tenants()
        ids = {t.tenant_id for t in tenants}
        assert ids == {"a", "b"}

    def test_delete_tenant(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        mgr.delete_tenant("acme")
        assert mgr.list_tenants() == []

    def test_delete_nonexistent_raises(self):
        mgr = TenantManager()
        with pytest.raises(KeyError, match="not found"):
            mgr.delete_tenant("nope")


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestWithTenant:
    def test_sets_context(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        with mgr.with_tenant("acme"):
            assert TenantContext.get_tenant_id() == "acme"
        assert TenantContext.get_tenant_id() is None

    def test_sets_namespace(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        with mgr.with_tenant("acme"):
            assert TenantContext.get_namespace() == "tenant:acme:"

    def test_nonexistent_raises(self):
        mgr = TenantManager()
        with pytest.raises(KeyError):
            with mgr.with_tenant("nope"):
                pass

    def test_restores_on_exception(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        try:
            with mgr.with_tenant("acme"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert TenantContext.get_tenant_id() is None

    def test_nested_contexts(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="a"))
        mgr.create_tenant(TenantConfig(tenant_id="b"))
        with mgr.with_tenant("a"):
            assert TenantContext.get_tenant_id() == "a"
            with mgr.with_tenant("b"):
                assert TenantContext.get_tenant_id() == "b"
            assert TenantContext.get_tenant_id() == "a"
        assert TenantContext.get_tenant_id() is None


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimit:
    def test_rate_limit_allows_first_request(self):
        mgr = TenantManager()
        cfg = TenantConfig(
            tenant_id="acme",
            max_queries_per_minute=2,
        )
        mgr.create_tenant(cfg)
        allowed, info = mgr.check_rate_limit("acme")
        assert allowed is True

    def test_rate_limit_blocks_after_exhaustion(self):
        mgr = TenantManager()
        cfg = TenantConfig(
            tenant_id="acme",
            max_queries_per_minute=1,
        )
        mgr.create_tenant(cfg)
        # Exhaust tokens (burst_size = min(1, 10) = 1)
        with mgr.with_tenant("acme"):
            pass
        # Second request should be blocked
        with pytest.raises(PermissionError, match="Rate limit"):
            with mgr.with_tenant("acme"):
                pass

    def test_rate_limit_not_found(self):
        mgr = TenantManager()
        with pytest.raises(KeyError):
            mgr.check_rate_limit("nope")


# ---------------------------------------------------------------------------
# Memory limits
# ---------------------------------------------------------------------------

class TestMemoryLimits:
    def test_check_memory_limit_no_neuralmem(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        assert mgr.check_memory_limit("acme") is True

    def test_get_memory_count_no_neuralmem(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        assert mgr.get_memory_count("acme") == 0

    def test_memory_limit_not_found(self):
        mgr = TenantManager()
        with pytest.raises(KeyError):
            mgr.check_memory_limit("nope")


# ---------------------------------------------------------------------------
# Memory type validation
# ---------------------------------------------------------------------------

class TestMemoryTypeValidation:
    def test_all_types_allowed_when_empty(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        assert mgr.validate_memory_type("acme", "fact") is True
        assert mgr.validate_memory_type("acme", "anything") is True

    def test_restricted_types(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(
            tenant_id="acme",
            allowed_memory_types=("fact", "preference"),
        ))
        assert mgr.validate_memory_type("acme", "fact") is True
        assert mgr.validate_memory_type("acme", "episodic") is False

    def test_validate_not_found(self):
        mgr = TenantManager()
        with pytest.raises(KeyError):
            mgr.validate_memory_type("nope", "fact")


# ---------------------------------------------------------------------------
# Deletion with NeuralMem
# ---------------------------------------------------------------------------

class TestDeleteWithNeuralMem:
    def test_delete_calls_storage(self):
        """When NeuralMem is provided, delete_tenant calls delete_memories."""
        from unittest.mock import MagicMock

        mock_nm = MagicMock()
        mgr = TenantManager(neuralmem=mock_nm)
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        mgr.delete_tenant("acme")
        mock_nm.storage.delete_memories.assert_called_once_with(
            user_id="acme"
        )
