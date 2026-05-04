"""Comprehensive unit tests for the access control module."""
from __future__ import annotations

import pytest

from neuralmem.access.permission import Permission
from neuralmem.access.role import (
    Role,
    PredefinedRole,
    make_viewer,
    make_editor,
    make_admin,
    make_owner,
    make_system,
    get_role,
    PREDEFINED_ROLES,
)
from neuralmem.access.control import (
    AccessControl,
    AccessDeniedError,
    ResourcePermission,
    SpacePermission,
)


# ---------------------------------------------------------------------------
# Permission tests
# ---------------------------------------------------------------------------

class TestPermission:
    def test_permission_values(self):
        assert Permission.READ.value == "read"
        assert Permission.WRITE.value == "write"
        assert Permission.DELETE.value == "delete"
        assert Permission.ADMIN.value == "admin"

    def test_permission_all(self):
        all_perms = Permission.all()
        assert len(all_perms) >= 11
        assert Permission.READ in all_perms
        assert Permission.ADMIN in all_perms

    def test_permission_memory_subset(self):
        mem_perms = Permission.memory_permissions()
        assert Permission.READ in mem_perms
        assert Permission.WRITE in mem_perms
        assert Permission.SEARCH in mem_perms
        assert Permission.ADMIN not in mem_perms

    def test_permission_space_subset(self):
        space_perms = Permission.space_permissions()
        assert Permission.SPACE_READ in space_perms
        assert Permission.SPACE_MANAGE_MEMBERS in space_perms
        assert Permission.READ not in space_perms

    def test_permission_admin_subset(self):
        admin_perms = Permission.admin_permissions()
        assert Permission.ADMIN in admin_perms
        assert Permission.ADMIN_AUDIT in admin_perms
        assert Permission.READ not in admin_perms


# ---------------------------------------------------------------------------
# Role tests
# ---------------------------------------------------------------------------

class TestRole:
    def test_role_creation(self):
        role = Role(name="test", permissions=frozenset({Permission.READ}))
        assert role.name == "test"
        assert role.has_permission(Permission.READ)
        assert not role.has_permission(Permission.WRITE)

    def test_role_has_any(self):
        role = Role(name="test", permissions=frozenset({Permission.READ, Permission.WRITE}))
        assert role.has_any(Permission.READ, Permission.ADMIN)
        assert role.has_any(Permission.ADMIN, Permission.WRITE)
        assert not role.has_any(Permission.ADMIN, Permission.DELETE)

    def test_role_has_all(self):
        role = Role(name="test", permissions=frozenset({Permission.READ, Permission.WRITE}))
        assert role.has_all(Permission.READ, Permission.WRITE)
        assert not role.has_all(Permission.READ, Permission.ADMIN)

    def test_role_with_permissions(self):
        role = Role(name="test", permissions=frozenset({Permission.READ}))
        new_role = role.with_permissions(Permission.WRITE)
        assert new_role.has_permission(Permission.READ)
        assert new_role.has_permission(Permission.WRITE)
        assert not role.has_permission(Permission.WRITE)  # original unchanged

    def test_role_without_permissions(self):
        role = Role(name="test", permissions=frozenset({Permission.READ, Permission.WRITE}))
        new_role = role.without_permissions(Permission.WRITE)
        assert new_role.has_permission(Permission.READ)
        assert not new_role.has_permission(Permission.WRITE)
        assert role.has_permission(Permission.WRITE)  # original unchanged

    def test_role_immutability(self):
        role = Role(name="test", permissions=frozenset({Permission.READ}))
        with pytest.raises(AttributeError):
            role.name = "changed"

    def test_role_repr(self):
        role = Role(name="viewer", permissions=frozenset({Permission.READ}))
        assert "viewer" in repr(role)
        assert "read" in repr(role)


class TestPredefinedRoles:
    def test_viewer_permissions(self):
        role = make_viewer()
        assert role.name == "viewer"
        assert role.has_permission(Permission.READ)
        assert role.has_permission(Permission.SEARCH)
        assert role.has_permission(Permission.SPACE_READ)
        assert not role.has_permission(Permission.WRITE)
        assert not role.has_permission(Permission.DELETE)

    def test_editor_permissions(self):
        role = make_editor()
        assert role.name == "editor"
        assert role.has_all(Permission.READ, Permission.WRITE, Permission.DELETE, Permission.SHARE)
        assert role.has_permission(Permission.SPACE_WRITE)
        assert not role.has_permission(Permission.SPACE_MANAGE_MEMBERS)
        assert not role.has_permission(Permission.ADMIN)

    def test_admin_permissions(self):
        role = make_admin()
        assert role.name == "admin"
        assert role.has_permission(Permission.SPACE_MANAGE_MEMBERS)
        assert role.has_permission(Permission.SPACE_CHANGE_SETTINGS)
        assert role.has_permission(Permission.SPACE_DELETE)
        assert not role.has_permission(Permission.ADMIN)
        assert not role.has_permission(Permission.ADMIN_AUDIT)

    def test_owner_permissions(self):
        role = make_owner()
        assert role.name == "owner"
        assert role.has_permission(Permission.ADMIN)
        assert role.has_permission(Permission.ADMIN_AUDIT)
        assert role.has_permission(Permission.ADMIN_IMPERSONATE)

    def test_system_permissions(self):
        role = make_system()
        assert role.name == "system"
        assert role.has_all(*Permission.all())

    def test_predefined_role_lookup(self):
        assert get_role("viewer") == make_viewer()
        assert get_role("editor") == make_editor()
        assert get_role("admin") == make_admin()
        assert get_role("owner") == make_owner()
        assert get_role("system") == make_system()
        assert get_role("nonexistent") is None

    def test_predefined_roles_dict(self):
        assert len(PREDEFINED_ROLES) == 5
        assert all(isinstance(r, Role) for r in PREDEFINED_ROLES.values())


# ---------------------------------------------------------------------------
# AccessControl — role assignment tests
# ---------------------------------------------------------------------------

class TestAccessControlRoles:
    def test_assign_role(self):
        ac = AccessControl()
        role = make_viewer()
        result = ac.assign_role("user-1", role)
        assert result == role
        assert ac.get_role("user-1") == role

    def test_assign_role_by_name(self):
        ac = AccessControl()
        role = ac.assign_role_by_name("user-1", "editor")
        assert role.name == "editor"
        assert role.has_permission(Permission.WRITE)

    def test_assign_unknown_role_raises(self):
        ac = AccessControl()
        with pytest.raises(ValueError, match="Unknown predefined role"):
            ac.assign_role_by_name("user-1", "wizard")

    def test_remove_role(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-1", "viewer")
        assert ac.remove_role("user-1") is True
        assert ac.get_role("user-1") is None

    def test_remove_nonexistent_role(self):
        ac = AccessControl()
        assert ac.remove_role("user-1") is False


# ---------------------------------------------------------------------------
# AccessControl — check() tests
# ---------------------------------------------------------------------------

class TestAccessControlCheck:
    def test_check_role_based_allowed(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-1", "editor")
        assert ac.check("user-1", Permission.WRITE) is True
        assert ac.check("user-1", Permission.READ) is True

    def test_check_role_based_denied(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-1", "viewer")
        assert ac.check("user-1", Permission.WRITE) is False
        assert ac.check("user-1", Permission.DELETE) is False

    def test_check_no_role(self):
        ac = AccessControl()
        assert ac.check("user-1", Permission.READ) is False

    def test_check_resource_grant_overrides_role(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-1", "viewer")  # no WRITE by default
        ac.grant("user-1", Permission.WRITE, resource_id="mem-123")
        # Without resource scope, still denied
        assert ac.check("user-1", Permission.WRITE) is False
        # With resource scope, allowed
        assert ac.check("user-1", Permission.WRITE, resource_id="mem-123") is True
        # Different resource, still denied
        assert ac.check("user-1", Permission.WRITE, resource_id="mem-999") is False

    def test_check_space_grant_overrides_role(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-1", "viewer")  # no SPACE_WRITE by default
        ac.grant("user-1", Permission.SPACE_WRITE, space_id="space-1")
        # Without space scope, denied
        assert ac.check("user-1", Permission.SPACE_WRITE) is False
        # With space scope, allowed
        assert ac.check("user-1", Permission.SPACE_WRITE, space_id="space-1") is True
        # Different space, denied
        assert ac.check("user-1", Permission.SPACE_WRITE, space_id="space-2") is False

    def test_check_resource_takes_priority_over_space(self):
        ac = AccessControl()
        ac.grant("user-1", Permission.DELETE, resource_id="mem-1")
        ac.grant("user-1", Permission.DELETE, space_id="space-1")
        # Resource grant should work even with different space
        assert ac.check("user-1", Permission.DELETE, resource_id="mem-1", space_id="space-2") is True

    def test_check_multiple_users_isolated(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-1", "editor")
        ac.assign_role_by_name("user-2", "viewer")
        assert ac.check("user-1", Permission.WRITE) is True
        assert ac.check("user-2", Permission.WRITE) is False


# ---------------------------------------------------------------------------
# AccessControl — require() tests
# ---------------------------------------------------------------------------

class TestAccessControlRequire:
    def test_require_allowed(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-1", "editor")
        ac.require("user-1", Permission.WRITE)  # should not raise

    def test_require_denied_raises(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-1", "viewer")
        with pytest.raises(AccessDeniedError, match="lacks permission"):
            ac.require("user-1", Permission.WRITE)

    def test_require_with_resource_scope(self):
        ac = AccessControl()
        ac.grant("user-1", Permission.DELETE, resource_id="mem-1")
        ac.require("user-1", Permission.DELETE, resource_id="mem-1")

    def test_require_with_resource_scope_denied(self):
        ac = AccessControl()
        with pytest.raises(AccessDeniedError, match="on resource"):
            ac.require("user-1", Permission.DELETE, resource_id="mem-1")

    def test_require_with_space_scope(self):
        ac = AccessControl()
        ac.grant("user-1", Permission.SPACE_DELETE, space_id="space-1")
        ac.require("user-1", Permission.SPACE_DELETE, space_id="space-1")

    def test_require_with_space_scope_denied(self):
        ac = AccessControl()
        with pytest.raises(AccessDeniedError, match="in space"):
            ac.require("user-1", Permission.SPACE_DELETE, space_id="space-1")


# ---------------------------------------------------------------------------
# AccessControl — grant() tests
# ---------------------------------------------------------------------------

class TestAccessControlGrant:
    def test_grant_resource_level(self):
        ac = AccessControl()
        grant = ac.grant("user-1", Permission.WRITE, resource_id="mem-1", granted_by="admin-1")
        assert isinstance(grant, ResourcePermission)
        assert grant.user_id == "user-1"
        assert grant.resource_id == "mem-1"
        assert grant.permission == Permission.WRITE
        assert grant.granted_by == "admin-1"

    def test_grant_space_level(self):
        ac = AccessControl()
        grant = ac.grant("user-1", Permission.SPACE_WRITE, space_id="space-1")
        assert isinstance(grant, SpacePermission)
        assert grant.space_id == "space-1"

    def test_grant_role_level(self):
        ac = AccessControl()
        grant = ac.grant("user-1", Permission.READ)
        # Returns SpacePermission with empty space_id for role-level
        assert isinstance(grant, SpacePermission)
        assert grant.space_id == ""
        assert ac.check("user-1", Permission.READ) is True

    def test_grant_with_metadata(self):
        ac = AccessControl()
        grant = ac.grant(
            "user-1",
            Permission.WRITE,
            resource_id="mem-1",
            metadata={"expiry": "2025-12-31", "reason": "project-x"},
        )
        assert grant.metadata["reason"] == "project-x"

    def test_grant_both_resource_and_space_raises(self):
        ac = AccessControl()
        with pytest.raises(ValueError, match="Cannot specify both"):
            ac.grant("user-1", Permission.READ, resource_id="mem-1", space_id="space-1")

    def test_grant_adds_to_existing_role(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-1", "viewer")
        ac.grant("user-1", Permission.WRITE)  # role-level
        assert ac.check("user-1", Permission.READ) is True  # from viewer
        assert ac.check("user-1", Permission.WRITE) is True  # granted

    def test_grant_creates_custom_role_if_none(self):
        ac = AccessControl()
        ac.grant("user-1", Permission.ADMIN)
        role = ac.get_role("user-1")
        assert role is not None
        assert role.name.startswith("custom-")
        assert role.has_permission(Permission.ADMIN)


# ---------------------------------------------------------------------------
# AccessControl — revoke() tests
# ---------------------------------------------------------------------------

class TestAccessControlRevoke:
    def test_revoke_resource_grant(self):
        ac = AccessControl()
        ac.grant("user-1", Permission.WRITE, resource_id="mem-1")
        assert ac.check("user-1", Permission.WRITE, resource_id="mem-1") is True
        assert ac.revoke("user-1", Permission.WRITE, resource_id="mem-1") is True
        assert ac.check("user-1", Permission.WRITE, resource_id="mem-1") is False

    def test_revoke_space_grant(self):
        ac = AccessControl()
        ac.grant("user-1", Permission.SPACE_WRITE, space_id="space-1")
        assert ac.check("user-1", Permission.SPACE_WRITE, space_id="space-1") is True
        assert ac.revoke("user-1", Permission.SPACE_WRITE, space_id="space-1") is True
        assert ac.check("user-1", Permission.SPACE_WRITE, space_id="space-1") is False

    def test_revoke_role_level_permission(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-1", "editor")
        assert ac.check("user-1", Permission.WRITE) is True
        assert ac.revoke("user-1", Permission.WRITE) is True
        assert ac.check("user-1", Permission.WRITE) is False
        assert ac.check("user-1", Permission.READ) is True  # other perms intact

    def test_revoke_nonexistent_returns_false(self):
        ac = AccessControl()
        assert ac.revoke("user-1", Permission.WRITE, resource_id="mem-1") is False
        assert ac.revoke("user-1", Permission.WRITE) is False

    def test_revoke_only_removes_specified_permission(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-1", "admin")
        ac.revoke("user-1", Permission.SPACE_DELETE)
        role = ac.get_role("user-1")
        assert not role.has_permission(Permission.SPACE_DELETE)
        assert role.has_permission(Permission.SPACE_MANAGE_MEMBERS)
        assert role.has_permission(Permission.WRITE)


# ---------------------------------------------------------------------------
# AccessControl — query tests
# ---------------------------------------------------------------------------

class TestAccessControlQueries:
    def test_list_resource_grants(self):
        ac = AccessControl()
        ac.grant("user-1", Permission.WRITE, resource_id="mem-1")
        ac.grant("user-1", Permission.DELETE, resource_id="mem-2")
        ac.grant("user-2", Permission.READ, resource_id="mem-1")

        all_grants = ac.list_resource_grants()
        assert len(all_grants) == 3

        user1_grants = ac.list_resource_grants(user_id="user-1")
        assert len(user1_grants) == 2
        assert all(g.user_id == "user-1" for g in user1_grants)

        mem1_grants = ac.list_resource_grants(resource_id="mem-1")
        assert len(mem1_grants) == 2
        assert all(g.resource_id == "mem-1" for g in mem1_grants)

    def test_list_space_grants(self):
        ac = AccessControl()
        ac.grant("user-1", Permission.SPACE_WRITE, space_id="space-1")
        ac.grant("user-1", Permission.SPACE_DELETE, space_id="space-2")
        ac.grant("user-2", Permission.SPACE_READ, space_id="space-1")

        all_grants = ac.list_space_grants()
        assert len(all_grants) == 3

        user1_grants = ac.list_space_grants(user_id="user-1")
        assert len(user1_grants) == 2

        space1_grants = ac.list_space_grants(space_id="space-1")
        assert len(space1_grants) == 2

    def test_list_user_permissions_role_only(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-1", "editor")
        perms = ac.list_user_permissions("user-1")
        assert Permission.READ in perms
        assert Permission.WRITE in perms
        assert Permission.DELETE in perms
        assert Permission.ADMIN not in perms

    def test_list_user_permissions_with_resource(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-1", "viewer")
        ac.grant("user-1", Permission.WRITE, resource_id="mem-1")
        ac.grant("user-1", Permission.DELETE, resource_id="mem-1")
        ac.grant("user-1", Permission.SHARE, resource_id="mem-2")

        perms = ac.list_user_permissions("user-1", resource_id="mem-1")
        assert Permission.READ in perms  # from role
        assert Permission.WRITE in perms  # from resource grant
        assert Permission.DELETE in perms  # from resource grant
        assert Permission.SHARE not in perms  # different resource

    def test_list_user_permissions_with_space(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-1", "viewer")
        ac.grant("user-1", Permission.SPACE_WRITE, space_id="space-1")
        ac.grant("user-1", Permission.SPACE_MANAGE_MEMBERS, space_id="space-1")

        perms = ac.list_user_permissions("user-1", space_id="space-1")
        assert Permission.READ in perms  # from role
        assert Permission.SPACE_WRITE in perms  # from space grant
        assert Permission.SPACE_MANAGE_MEMBERS in perms
        assert Permission.SPACE_DELETE not in perms

    def test_list_user_permissions_combined(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-1", "viewer")
        ac.grant("user-1", Permission.WRITE, resource_id="mem-1")
        ac.grant("user-1", Permission.SPACE_WRITE, space_id="space-1")

        perms = ac.list_user_permissions("user-1", resource_id="mem-1", space_id="space-1")
        assert Permission.READ in perms
        assert Permission.WRITE in perms
        assert Permission.SPACE_WRITE in perms


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------

class TestAccessControlIntegration:
    def test_full_workflow(self):
        ac = AccessControl()

        # Setup users with roles
        ac.assign_role_by_name("alice", "owner")
        ac.assign_role_by_name("bob", "editor")
        ac.assign_role_by_name("charlie", "viewer")

        # Role-based checks
        assert ac.check("alice", Permission.ADMIN) is True
        assert ac.check("bob", Permission.WRITE) is True
        assert ac.check("bob", Permission.ADMIN) is False
        assert ac.check("charlie", Permission.READ) is True
        assert ac.check("charlie", Permission.WRITE) is False

        # Grant Charlie write on a specific memory
        ac.grant("charlie", Permission.WRITE, resource_id="mem-special", granted_by="alice")
        assert ac.check("charlie", Permission.WRITE, resource_id="mem-special") is True
        assert ac.check("charlie", Permission.WRITE, resource_id="mem-other") is False
        assert ac.check("charlie", Permission.WRITE) is False

        # Grant Charlie space management on one space
        ac.grant("charlie", Permission.SPACE_MANAGE_MEMBERS, space_id="space-a")
        assert ac.check("charlie", Permission.SPACE_MANAGE_MEMBERS, space_id="space-a") is True
        assert ac.check("charlie", Permission.SPACE_MANAGE_MEMBERS, space_id="space-b") is False

        # Revoke the resource grant
        ac.revoke("charlie", Permission.WRITE, resource_id="mem-special")
        assert ac.check("charlie", Permission.WRITE, resource_id="mem-special") is False

        # Revoke the space grant
        ac.revoke("charlie", Permission.SPACE_MANAGE_MEMBERS, space_id="space-a")
        assert ac.check("charlie", Permission.SPACE_MANAGE_MEMBERS, space_id="space-a") is False

        # Role-level: promote Charlie to editor
        ac.assign_role_by_name("charlie", "editor")
        assert ac.check("charlie", Permission.WRITE) is True
        assert ac.check("charlie", Permission.SPACE_MANAGE_MEMBERS) is False

        # Demote: remove role entirely
        ac.remove_role("charlie")
        assert ac.check("charlie", Permission.READ) is False

    def test_owner_can_do_everything(self):
        ac = AccessControl()
        ac.assign_role_by_name("owner", "owner")
        for perm in Permission.all():
            assert ac.check("owner", perm) is True, f"Owner should have {perm}"

    def test_system_role(self):
        ac = AccessControl()
        ac.assign_role_by_name("system", "system")
        assert ac.check("system", Permission.ADMIN_IMPERSONATE) is True
        assert ac.check("system", Permission.SPACE_DELETE) is True
        assert ac.check("system", Permission.READ) is True

    def test_isolation_between_users(self):
        ac = AccessControl()
        ac.assign_role_by_name("user-a", "editor")
        ac.assign_role_by_name("user-b", "viewer")
        ac.grant("user-a", Permission.DELETE, resource_id="mem-1")
        ac.grant("user-b", Permission.WRITE, space_id="space-1")

        # Resource grants are isolated per user
        assert ac.check("user-a", Permission.DELETE, resource_id="mem-1") is True
        assert ac.check("user-b", Permission.DELETE, resource_id="mem-1") is False

        # Space grants are isolated per user
        assert ac.check("user-b", Permission.WRITE, space_id="space-1") is True
        assert ac.check("user-a", Permission.WRITE, space_id="space-1") is True  # editor has global WRITE

        # user-b does NOT have DELETE globally or on mem-1
        assert ac.check("user-b", Permission.DELETE, resource_id="mem-1") is False

    def test_grant_revoke_idempotency(self):
        ac = AccessControl()
        ac.grant("user-1", Permission.WRITE, resource_id="mem-1")
        # Grant again (overwrites)
        ac.grant("user-1", Permission.WRITE, resource_id="mem-1", metadata={"version": 2})
        assert ac.check("user-1", Permission.WRITE, resource_id="mem-1") is True
        # Revoke twice
        assert ac.revoke("user-1", Permission.WRITE, resource_id="mem-1") is True
        assert ac.revoke("user-1", Permission.WRITE, resource_id="mem-1") is False
        assert ac.check("user-1", Permission.WRITE, resource_id="mem-1") is False
