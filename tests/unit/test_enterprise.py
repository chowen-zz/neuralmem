"""Unit tests for NeuralMem Enterprise V1.3 features.

All external dependencies are mocked; tests are fast and isolated.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from neuralmem.enterprise import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    ComplianceStandard,
    Permission,
    RBACManager,
    ResourceAction,
    Role,
    TenantConfig,
    TenantContext,
    TenantManager,
)


# ------------------------------------------------------------------
# TenantManager tests
# ------------------------------------------------------------------

class TestTenantManager:
    def test_create_tenant(self):
        mgr = TenantManager()
        cfg = TenantConfig(tenant_id="acme", name="Acme Corp")
        result = mgr.create_tenant(cfg)
        assert result.tenant_id == "acme"
        assert result.storage_namespace == "tenant:acme:"

    def test_create_tenant_duplicate_raises(self):
        mgr = TenantManager()
        cfg = TenantConfig(tenant_id="acme")
        mgr.create_tenant(cfg)
        with pytest.raises(ValueError, match="already exists"):
            mgr.create_tenant(cfg)

    def test_delete_tenant(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        mgr.delete_tenant("acme")
        assert not mgr.tenant_exists("acme")

    def test_delete_tenant_not_found(self):
        mgr = TenantManager()
        with pytest.raises(KeyError):
            mgr.delete_tenant("missing")

    def test_get_tenant(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme", name="Acme"))
        cfg = mgr.get_tenant("acme")
        assert cfg.name == "Acme"

    def test_list_tenants(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="a"))
        mgr.create_tenant(TenantConfig(tenant_id="b"))
        assert len(mgr.list_tenants()) == 2

    def test_tenant_exists(self):
        mgr = TenantManager()
        assert not mgr.tenant_exists("acme")
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        assert mgr.tenant_exists("acme")

    def test_with_tenant_context(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        assert TenantContext.get_tenant_id() is None
        with mgr.with_tenant("acme"):
            assert TenantContext.get_tenant_id() == "acme"
            assert TenantContext.get_namespace() == "tenant:acme:"
        assert TenantContext.get_tenant_id() is None

    def test_with_tenant_restores_previous(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        mgr.create_tenant(TenantConfig(tenant_id="globex"))
        TenantContext.set_tenant_id("globex")
        with mgr.with_tenant("acme"):
            assert TenantContext.get_tenant_id() == "acme"
        assert TenantContext.get_tenant_id() == "globex"
        TenantContext.clear()

    def test_with_tenant_missing_raises(self):
        mgr = TenantManager()
        with pytest.raises(KeyError):
            with mgr.with_tenant("acme"):
                pass  # pragma: no cover

    def test_put_get_delete(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        mgr.put("acme", "key1", "value1")
        assert mgr.get("acme", "key1") == "value1"
        assert mgr.get("acme", "missing") is None
        assert mgr.delete("acme", "key1") is True
        assert mgr.delete("acme", "key1") is False

    def test_list_keys(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        mgr.put("acme", "a", 1)
        mgr.put("acme", "b", 2)
        assert sorted(mgr.list_keys("acme")) == ["a", "b"]

    def test_clear_tenant_data(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        mgr.put("acme", "a", 1)
        mgr.clear_tenant_data("acme")
        assert mgr.list_keys("acme") == []
        assert mgr.tenant_exists("acme")

    def test_check_memory_limit(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme", max_memories=2))
        assert mgr.check_memory_limit("acme") is True
        mgr.put("acme", "a", 1)
        assert mgr.check_memory_limit("acme") is True
        mgr.put("acme", "b", 2)
        assert mgr.check_memory_limit("acme") is False

    def test_get_memory_count(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        assert mgr.get_memory_count("acme") == 0
        mgr.put("acme", "x", 1)
        assert mgr.get_memory_count("acme") == 1

    def test_validate_memory_type(self):
        mgr = TenantManager()
        mgr.create_tenant(
            TenantConfig(tenant_id="acme", allowed_memory_types=("fact", "episodic"))
        )
        assert mgr.validate_memory_type("acme", "fact") is True
        assert mgr.validate_memory_type("acme", "semantic") is False

    def test_validate_memory_type_empty_allows_all(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        assert mgr.validate_memory_type("acme", "anything") is True

    def test_scoped_key(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        assert mgr.scoped_key("acme", "mem1") == "tenant:acme:mem1"

    def test_tenant_storage_raises_for_missing_tenant(self):
        mgr = TenantManager()
        with pytest.raises(KeyError):
            mgr.put("missing", "k", "v")

    def test_concurrent_tenant_access(self):
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="acme", max_memories=200))
        errors: list[Exception] = []

        def worker():
            try:
                for i in range(50):
                    mgr.put("acme", f"key_{threading.current_thread().name}_{i}", i)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, name=f"t{n}") for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert mgr.get_memory_count("acme") == 200

    def test_storage_backend_mock(self):
        mock_backend = MagicMock()
        mgr = TenantManager(storage_backend=mock_backend)
        mgr.create_tenant(TenantConfig(tenant_id="acme"))
        assert mgr._storage is mock_backend


# ------------------------------------------------------------------
# AuditLogger tests
# ------------------------------------------------------------------

class TestAuditLogger:
    def test_log_event(self):
        logger = AuditLogger()
        event = logger.log(
            AuditEventType.MEMORY_CREATE,
            tenant_id="acme",
            user_id="alice",
            memory_id="mem1",
        )
        assert event.event_type == AuditEventType.MEMORY_CREATE
        assert event.tenant_id == "acme"
        assert event.user_id == "alice"
        assert event.memory_id == "mem1"
        assert event.success is True
        assert logger.count == 1

    def test_log_memory_access(self):
        logger = AuditLogger()
        event = logger.log_memory_access(
            AuditEventType.MEMORY_READ,
            tenant_id="acme",
            user_id="alice",
            memory_id="mem1",
        )
        assert event.resource == "memory"
        assert event.action == "memory_read"

    def test_log_access_denied(self):
        logger = AuditLogger()
        event = logger.log_access_denied(
            tenant_id="acme",
            user_id="bob",
            action="delete",
            resource="memory",
            reason="insufficient_permissions",
        )
        assert event.success is False
        assert event.event_type == AuditEventType.ACCESS_DENIED
        assert event.details["reason"] == "insufficient_permissions"

    def test_log_data_purge(self):
        logger = AuditLogger()
        event = logger.log_data_purge(
            tenant_id="acme",
            user_id="gdpr_officer",
            memory_ids=["m1", "m2"],
            reason="gdpr_request",
        )
        assert event.event_type == AuditEventType.DATA_PURGE
        assert event.details["memory_ids"] == ["m1", "m2"]

    def test_query_by_event_type(self):
        logger = AuditLogger()
        logger.log(AuditEventType.MEMORY_CREATE, tenant_id="acme")
        logger.log(AuditEventType.MEMORY_READ, tenant_id="acme")
        logger.log(AuditEventType.MEMORY_DELETE, tenant_id="acme")
        results = logger.query(event_type=AuditEventType.MEMORY_READ)
        assert len(results) == 1
        assert results[0].event_type == AuditEventType.MEMORY_READ

    def test_query_by_tenant(self):
        logger = AuditLogger()
        logger.log(AuditEventType.MEMORY_CREATE, tenant_id="acme")
        logger.log(AuditEventType.MEMORY_CREATE, tenant_id="globex")
        assert len(logger.query(tenant_id="acme")) == 1

    def test_query_by_user(self):
        logger = AuditLogger()
        logger.log(AuditEventType.MEMORY_CREATE, tenant_id="acme", user_id="alice")
        logger.log(AuditEventType.MEMORY_CREATE, tenant_id="acme", user_id="bob")
        assert len(logger.query(user_id="alice")) == 1

    def test_query_by_success(self):
        logger = AuditLogger()
        logger.log(AuditEventType.MEMORY_CREATE, success=True)
        logger.log(AuditEventType.ACCESS_DENIED, success=False)
        assert len(logger.query(success=True)) == 1
        assert len(logger.query(success=False)) == 1

    def test_query_time_range(self):
        logger = AuditLogger()
        now = datetime.now(timezone.utc)
        logger.log(AuditEventType.MEMORY_CREATE)
        results = logger.query(since=now - timedelta(seconds=1), until=now + timedelta(seconds=1))
        assert len(results) == 1
        results = logger.query(since=now + timedelta(seconds=10))
        assert len(results) == 0

    def test_query_limit(self):
        logger = AuditLogger()
        for _ in range(10):
            logger.log(AuditEventType.MEMORY_CREATE)
        assert len(logger.query(limit=3)) == 3

    def test_query_by_compliance(self):
        logger = AuditLogger(compliance_standards=(ComplianceStandard.GDPR,))
        logger.log(AuditEventType.MEMORY_CREATE)
        results = logger.query_by_compliance(ComplianceStandard.GDPR)
        assert len(results) == 1
        assert all(ComplianceStandard.GDPR.value in e.compliance_tags for e in results)

    def test_get_event_counts(self):
        logger = AuditLogger()
        logger.log(AuditEventType.MEMORY_CREATE)
        logger.log(AuditEventType.MEMORY_CREATE)
        logger.log(AuditEventType.MEMORY_READ)
        counts = logger.get_event_counts()
        assert counts["memory_create"] == 2
        assert counts["memory_read"] == 1

    def test_clear(self):
        logger = AuditLogger()
        logger.log(AuditEventType.MEMORY_CREATE)
        logger.clear()
        assert logger.count == 0
        assert logger.get_event_counts() == {}

    def test_get_user_activity(self):
        logger = AuditLogger()
        logger.log(AuditEventType.MEMORY_CREATE, user_id="alice")
        logger.log(AuditEventType.MEMORY_READ, user_id="alice")
        logger.log(AuditEventType.MEMORY_CREATE, user_id="bob")
        results = logger.get_user_activity("alice")
        assert len(results) == 2
        assert all(e.user_id == "alice" for e in results)

    def test_get_tenant_activity(self):
        logger = AuditLogger()
        logger.log(AuditEventType.MEMORY_CREATE, tenant_id="acme")
        logger.log(AuditEventType.TENANT_DELETE, tenant_id="acme")
        results = logger.get_tenant_activity("acme")
        assert len(results) == 2

    def test_export_for_compliance(self):
        logger = AuditLogger()
        logger.log(AuditEventType.MEMORY_CREATE, tenant_id="acme", user_id="alice")
        exported = logger.export_for_compliance(ComplianceStandard.GDPR)
        assert len(exported) == 1
        assert exported[0]["compliance_standard"] == "gdpr"
        assert "timestamp" in exported[0]

    def test_ring_buffer_eviction(self):
        logger = AuditLogger(max_events=3)
        logger.log(AuditEventType.MEMORY_CREATE)
        logger.log(AuditEventType.MEMORY_READ)
        logger.log(AuditEventType.MEMORY_UPDATE)
        logger.log(AuditEventType.MEMORY_DELETE)
        assert logger.count == 3
        # oldest should be evicted
        oldest = logger.query(limit=3)[-1]
        assert oldest.event_type == AuditEventType.MEMORY_READ

    def test_concurrent_logging(self):
        logger = AuditLogger()
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(100):
                    logger.log(AuditEventType.MEMORY_CREATE)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert logger.count == 400


# ------------------------------------------------------------------
# RBACManager tests
# ------------------------------------------------------------------

class TestRBACManager:
    def test_builtin_roles_exist(self):
        rbac = RBACManager()
        roles = rbac.list_roles()
        assert "viewer" in roles
        assert "editor" in roles
        assert "admin" in roles
        assert "auditor" in roles
        assert "gdpr_controller" in roles

    def test_add_custom_role(self):
        rbac = RBACManager()
        role = Role(
            name="custom",
            permissions={"memory": Permission.READ, "special": Permission.ADMIN},
        )
        rbac.add_role(role)
        assert rbac.get_role("custom") == role

    def test_remove_custom_role(self):
        rbac = RBACManager()
        rbac.add_role(Role(name="tmp", permissions={"memory": Permission.READ}))
        assert rbac.remove_role("tmp") is True
        assert rbac.get_role("tmp") is None

    def test_remove_builtin_role_fails(self):
        rbac = RBACManager()
        assert rbac.remove_role("admin") is False
        assert rbac.get_role("admin") is not None

    def test_grant_and_check(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice", "viewer")
        assert rbac.check("acme", "alice", "memory", ResourceAction.READ) is True
        assert rbac.check("acme", "alice", "memory", ResourceAction.CREATE) is False

    def test_grant_invalid_role_raises(self):
        rbac = RBACManager()
        with pytest.raises(ValueError, match="does not exist"):
            rbac.grant("acme", "alice", "nonexistent")

    def test_revoke_specific_role(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice", "viewer")
        rbac.grant("acme", "alice", "editor")
        assert rbac.revoke("acme", "alice", "viewer") is True
        assert rbac.check("acme", "alice", "memory", ResourceAction.READ) is True  # editor still
        assert rbac.check("acme", "alice", "memory", ResourceAction.CREATE) is True

    def test_revoke_all_roles(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice", "viewer")
        assert rbac.revoke("acme", "alice") is True
        assert rbac.check("acme", "alice", "memory", ResourceAction.READ) is False

    def test_revoke_nonexistent(self):
        rbac = RBACManager()
        assert rbac.revoke("acme", "alice") is False

    def test_list_grants(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice", "viewer")
        rbac.grant("acme", "bob", "editor")
        rbac.grant("globex", "alice", "admin")
        assert len(rbac.list_grants()) == 3
        assert len(rbac.list_grants(tenant_id="acme")) == 2
        assert len(rbac.list_grants(user_id="alice")) == 2

    def test_assert_allowed_passes(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice", "admin")
        rbac.assert_allowed("acme", "alice", "memory", ResourceAction.DELETE)

    def test_assert_allowed_raises(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice", "viewer")
        with pytest.raises(PermissionError):
            rbac.assert_allowed("acme", "alice", "memory", ResourceAction.DELETE)

    def test_get_effective_permissions(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice", "viewer")
        rbac.grant("acme", "alice", "editor")
        perms = rbac.get_effective_permissions("acme", "alice")
        assert perms["memory"] == Permission.WRITE

    def test_get_tenant_users(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice", "viewer")
        rbac.grant("acme", "bob", "editor")
        rbac.grant("globex", "carol", "admin")
        assert rbac.get_tenant_users("acme") == ["alice", "bob"]

    def test_get_user_roles(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice", "viewer")
        rbac.grant("acme", "alice", "auditor")
        assert rbac.get_user_roles("acme", "alice") == ["viewer", "auditor"]

    def test_has_any_grant(self):
        rbac = RBACManager()
        assert rbac.has_any_grant("acme", "alice") is False
        rbac.grant("acme", "alice", "viewer")
        assert rbac.has_any_grant("acme", "alice") is True

    def test_role_hierarchy(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice", "admin")
        assert rbac.can_read_memory("acme", "alice") is True
        assert rbac.can_write_memory("acme", "alice") is True
        assert rbac.can_delete_memory("acme", "alice") is True
        assert rbac.can_purge_memory("acme", "alice") is True
        assert rbac.can_admin_tenant("acme", "alice") is True
        assert rbac.can_read_audit("acme", "alice") is True
        assert rbac.can_admin_rbac("acme", "alice") is True

    def test_viewer_restrictions(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice", "viewer")
        assert rbac.can_read_memory("acme", "alice") is True
        assert rbac.can_write_memory("acme", "alice") is False
        assert rbac.can_delete_memory("acme", "alice") is False
        assert rbac.can_admin_tenant("acme", "alice") is False

    def test_editor_permissions(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice", "editor")
        assert rbac.can_read_memory("acme", "alice") is True
        assert rbac.can_write_memory("acme", "alice") is True
        assert rbac.can_delete_memory("acme", "alice") is False

    def test_auditor_permissions(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice", "auditor")
        assert rbac.can_read_memory("acme", "alice") is True
        assert rbac.can_read_audit("acme", "alice") is True
        assert rbac.can_admin_rbac("acme", "alice") is False

    def test_gdpr_controller_permissions(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice", "gdpr_controller")
        assert rbac.can_read_memory("acme", "alice") is True
        assert rbac.can_read_audit("acme", "alice") is True
        assert rbac.can_write_memory("acme", "alice") is False
        assert rbac.can_delete_memory("acme", "alice") is False

    def test_gdpr_controller_can_purge(self):
        rbac = RBACManager()
        # Give delete permission via a custom role to test purge
        custom = Role(
            name="gdpr_purger",
            permissions={
                "memory": Permission.DELETE,
                "tenant": Permission.READ,
                "audit": Permission.ADMIN,
            },
        )
        rbac.add_role(custom)
        rbac.grant("acme", "alice", "gdpr_purger")
        assert rbac.can_delete_memory("acme", "alice") is True
        assert rbac.can_purge_memory("acme", "alice") is True

    def test_concurrent_grants(self):
        rbac = RBACManager()
        errors: list[Exception] = []

        def worker(n: int):
            try:
                for i in range(25):
                    rbac.grant(f"tenant_{n}", f"user_{i}", "viewer")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(rbac.list_grants()) == 100

    def test_role_can_method(self):
        role = Role(
            name="test",
            permissions={"memory": Permission.WRITE, "audit": Permission.READ},
        )
        assert role.can("memory", ResourceAction.READ) is True
        assert role.can("memory", ResourceAction.CREATE) is True
        assert role.can("memory", ResourceAction.DELETE) is False
        assert role.can("audit", ResourceAction.ADMIN) is False

    def test_action_to_permission_mapping(self):
        from neuralmem.enterprise.rbac import _action_to_permission

        assert _action_to_permission(ResourceAction.READ) == Permission.READ
        assert _action_to_permission(ResourceAction.LIST) == Permission.READ
        assert _action_to_permission(ResourceAction.CREATE) == Permission.WRITE
        assert _action_to_permission(ResourceAction.UPDATE) == Permission.WRITE
        assert _action_to_permission(ResourceAction.EXPORT) == Permission.READ
        assert _action_to_permission(ResourceAction.DELETE) == Permission.DELETE
        assert _action_to_permission(ResourceAction.PURGE) == Permission.DELETE
        assert _action_to_permission(ResourceAction.ADMIN) == Permission.ADMIN


# ------------------------------------------------------------------
# Integration-style tests across enterprise modules
# ------------------------------------------------------------------

class TestEnterpriseIntegration:
    def test_tenant_rbac_audit_flow(self):
        """End-to-end: create tenant, grant role, log audit, verify access."""
        tenant_mgr = TenantManager()
        rbac = RBACManager()
        audit = AuditLogger()

        # 1. Create tenant
        tenant_mgr.create_tenant(TenantConfig(tenant_id="acme", name="Acme Corp"))

        # 2. Grant role
        rbac.grant("acme", "alice", "editor")

        # 3. Verify access
        assert rbac.can_write_memory("acme", "alice") is True

        # 4. Log audit
        event = audit.log_memory_access(
            AuditEventType.MEMORY_CREATE,
            tenant_id="acme",
            user_id="alice",
            memory_id="mem1",
            details={"content_preview": "hello"},
        )
        assert event.tenant_id == "acme"

        # 5. Tenant-scoped storage
        tenant_mgr.put("acme", "mem1", {"content": "hello"})
        assert tenant_mgr.get("acme", "mem1") == {"content": "hello"}

        # 6. Verify audit query
        assert len(audit.get_tenant_activity("acme")) == 1
        assert len(audit.get_user_activity("alice")) == 1

    def test_access_denied_logged(self):
        tenant_mgr = TenantManager()
        rbac = RBACManager()
        audit = AuditLogger()

        tenant_mgr.create_tenant(TenantConfig(tenant_id="acme"))
        rbac.grant("acme", "bob", "viewer")

        # bob tries to delete (denied)
        allowed = rbac.can_delete_memory("acme", "bob")
        assert allowed is False

        audit.log_access_denied(
            tenant_id="acme",
            user_id="bob",
            action="memory_delete",
            resource="memory",
            reason="viewer_role",
        )

        denied_events = audit.query(event_type=AuditEventType.ACCESS_DENIED)
        assert len(denied_events) == 1
        assert denied_events[0].user_id == "bob"

    def test_gdpr_purge_flow(self):
        tenant_mgr = TenantManager()
        rbac = RBACManager()
        audit = AuditLogger()

        tenant_mgr.create_tenant(TenantConfig(tenant_id="acme"))

        # Give officer a custom purger role since built-in gdpr_controller is read-only
        purger_role = Role(
            name="gdpr_purger",
            permissions={
                "memory": Permission.DELETE,
                "tenant": Permission.READ,
                "audit": Permission.ADMIN,
            },
        )
        rbac.add_role(purger_role)
        rbac.grant("acme", "gdpr_officer", "gdpr_purger")

        # Store some PII
        tenant_mgr.put("acme", "user_email", "alice@example.com")
        tenant_mgr.put("acme", "user_phone", "+1234567890")

        # Verify officer can purge
        assert rbac.can_purge_memory("acme", "gdpr_officer") is True

        # Perform purge
        tenant_mgr.delete("acme", "user_email")
        tenant_mgr.delete("acme", "user_phone")

        # Log purge
        audit.log_data_purge(
            tenant_id="acme",
            user_id="gdpr_officer",
            memory_ids=["user_email", "user_phone"],
            reason="gdpr_article_17",
        )

        # Verify audit trail
        purge_events = audit.query(event_type=AuditEventType.DATA_PURGE)
        assert len(purge_events) == 1
        assert purge_events[0].details["reason"] == "gdpr_article_17"

    def test_mock_storage_backend_integration(self):
        mock_backend = MagicMock()
        mock_backend.get_stats.return_value = {"total_memories": 42}

        tenant_mgr = TenantManager(storage_backend=mock_backend)
        tenant_mgr.create_tenant(TenantConfig(tenant_id="acme"))

        # Simulate backend interaction
        stats = mock_backend.get_stats(user_id="acme")
        assert stats["total_memories"] == 42
        mock_backend.get_stats.assert_called_once_with(user_id="acme")

    def test_compliance_export_integration(self):
        audit = AuditLogger()
        audit.log(
            AuditEventType.MEMORY_CREATE,
            tenant_id="acme",
            user_id="alice",
        )
        audit.log(
            AuditEventType.MEMORY_DELETE,
            tenant_id="acme",
            user_id="alice",
        )

        gdpr_export = audit.export_for_compliance(ComplianceStandard.GDPR)
        assert len(gdpr_export) == 2

        soc2_export = audit.export_for_compliance(ComplianceStandard.SOC2)
        assert len(soc2_export) == 2

    def test_tenant_isolation(self):
        """Ensure data from one tenant is not visible to another."""
        mgr = TenantManager()
        mgr.create_tenant(TenantConfig(tenant_id="tenant_a"))
        mgr.create_tenant(TenantConfig(tenant_id="tenant_b"))

        mgr.put("tenant_a", "secret", "a_secret")
        mgr.put("tenant_b", "secret", "b_secret")

        assert mgr.get("tenant_a", "secret") == "a_secret"
        assert mgr.get("tenant_b", "secret") == "b_secret"
        assert mgr.list_keys("tenant_a") == ["secret"]
        assert mgr.list_keys("tenant_b") == ["secret"]

    def test_rbac_cross_tenant_denial(self):
        """A user granted in one tenant should have no access in another."""
        rbac = RBACManager()
        rbac.grant("acme", "alice", "admin")
        assert rbac.can_admin_tenant("acme", "alice") is True
        assert rbac.can_admin_tenant("globex", "alice") is False
