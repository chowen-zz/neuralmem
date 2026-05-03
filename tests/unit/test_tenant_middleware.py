"""Tests for TenantMiddleware MCP integration."""
from __future__ import annotations

import pytest

from neuralmem.tenancy.manager import TenantContext, TenantManager
from neuralmem.tenancy.middleware import TenantMiddleware
from neuralmem.tenancy.models import TenantConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manager() -> TenantManager:
    mgr = TenantManager()
    mgr.create_tenant(TenantConfig(tenant_id="acme"))
    mgr.create_tenant(TenantConfig(tenant_id="globex"))
    return mgr


# ---------------------------------------------------------------------------
# process_request
# ---------------------------------------------------------------------------

class TestProcessRequest:
    def test_sets_tenant_context(self):
        mw = TenantMiddleware(_manager())
        with mw.process_request({"tenant_id": "acme"}) as ctx:
            assert ctx["tenant_id"] == "acme"
            assert TenantContext.get_tenant_id() == "acme"

    def test_restores_context_on_exit(self):
        mw = TenantMiddleware(_manager())
        with mw.process_request({"tenant_id": "acme"}):
            pass
        assert TenantContext.get_tenant_id() is None

    def test_missing_tenant_id_raises(self):
        mw = TenantMiddleware(_manager())
        with pytest.raises(ValueError, match="Missing tenant_id"):
            with mw.process_request({}):
                pass

    def test_nonexistent_tenant_raises(self):
        mw = TenantMiddleware(_manager())
        with pytest.raises(KeyError, match="not found"):
            with mw.process_request({"tenant_id": "unknown"}):
                pass

    def test_returns_namespace(self):
        mw = TenantMiddleware(_manager())
        with mw.process_request({"tenant_id": "acme"}) as ctx:
            assert ctx["namespace"] == "tenant:acme:"

    def test_returns_config(self):
        mw = TenantMiddleware(_manager())
        with mw.process_request({"tenant_id": "acme"}) as ctx:
            assert ctx["config"].tenant_id == "acme"

    def test_restores_on_exception(self):
        mw = TenantMiddleware(_manager())
        try:
            with mw.process_request({"tenant_id": "acme"}):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert TenantContext.get_tenant_id() is None


# ---------------------------------------------------------------------------
# validate_request
# ---------------------------------------------------------------------------

class TestValidateRequest:
    def test_valid_request(self):
        mw = TenantMiddleware(_manager())
        result = mw.validate_request({"tenant_id": "acme"})
        assert result["tenant_id"] == "acme"
        assert result["namespace"] == "tenant:acme:"

    def test_missing_tenant_id(self):
        mw = TenantMiddleware(_manager())
        with pytest.raises(ValueError, match="Missing tenant_id"):
            mw.validate_request({})

    def test_nonexistent_tenant(self):
        mw = TenantMiddleware(_manager())
        with pytest.raises(KeyError):
            mw.validate_request({"tenant_id": "nope"})

    def test_validate_does_not_set_context(self):
        """validate_request should NOT mutate thread-local context."""
        mw = TenantMiddleware(_manager())
        TenantContext.clear()
        mw.validate_request({"tenant_id": "acme"})
        assert TenantContext.get_tenant_id() is None


# ---------------------------------------------------------------------------
# _extract_tenant_id key variants
# ---------------------------------------------------------------------------

class TestExtractTenantId:
    def test_underscore_key(self):
        mw = TenantMiddleware(_manager())
        result = mw.validate_request({"tenant_id": "acme"})
        assert result["tenant_id"] == "acme"

    def test_hyphen_key(self):
        mw = TenantMiddleware(_manager())
        result = mw.validate_request({"tenant-id": "acme"})
        assert result["tenant_id"] == "acme"

    def test_header_key(self):
        mw = TenantMiddleware(_manager())
        result = mw.validate_request({"X-Tenant-Id": "acme"})
        assert result["tenant_id"] == "acme"

    def test_underscore_header_key(self):
        mw = TenantMiddleware(_manager())
        result = mw.validate_request({"x_tenant_id": "acme"})
        assert result["tenant_id"] == "acme"
