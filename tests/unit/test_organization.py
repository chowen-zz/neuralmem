"""Comprehensive unit tests for NeuralMem V2.0 organization module.

Covers:
  • Organization (dataclass, settings, serialization)
  • OrganizationMember (roles, status, permissions)
  • OrgManager (CRUD, member management, settings, permissions, ownership transfer)
  • SQLite persistence (optional db_path)
  • Thread safety
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from neuralmem.organization import (
    Organization,
    OrgSettings,
    OrgStatus,
    OrganizationMember,
    MemberRole,
    MembershipStatus,
    OrgManager,
    OrgNotFoundError,
    MemberNotFoundError,
    DuplicateMemberError,
    RoleAssignmentError,
)


# =============================================================================
# Organization
# =============================================================================

class TestOrganizationCreation:
    def test_create_defaults(self):
        org = Organization(name="Acme Corp", owner_id="u1")
        assert org.name == "Acme Corp"
        assert org.owner_id == "u1"
        assert org.slug == "acme-corp"
        assert org.status == OrgStatus.ACTIVE
        assert org.org_id.startswith("org_")
        assert isinstance(org.settings, OrgSettings)

    def test_create_with_slug(self):
        org = Organization(name="Acme Corp", slug="custom-slug", owner_id="u1")
        assert org.slug == "custom-slug"

    def test_create_with_description(self):
        org = Organization(name="Acme", description="A test org", owner_id="u1")
        assert org.description == "A test org"

    def test_is_active(self):
        org = Organization(name="Acme", owner_id="u1", status=OrgStatus.ACTIVE)
        assert org.is_active() is True
        org.status = OrgStatus.SUSPENDED
        assert org.is_active() is False


class TestOrganizationUpdate:
    def test_update_name(self):
        org = Organization(name="Old Name", owner_id="u1")
        org.update(name="New Name")
        assert org.name == "New Name"
        assert org.slug == "new-name"

    def test_update_description(self):
        org = Organization(name="Acme", owner_id="u1")
        org.update(description="Updated")
        assert org.description == "Updated"

    def test_update_status(self):
        org = Organization(name="Acme", owner_id="u1")
        org.update(status=OrgStatus.SUSPENDED)
        assert org.status == OrgStatus.SUSPENDED

    def test_update_metadata(self):
        org = Organization(name="Acme", owner_id="u1")
        org.update(metadata={"key": "value"})
        assert org.metadata == {"key": "value"}

    def test_update_owner_id(self):
        org = Organization(name="Acme", owner_id="u1")
        org.update(owner_id="u2")
        assert org.owner_id == "u2"

    def test_update_ignores_invalid_keys(self):
        org = Organization(name="Acme", owner_id="u1")
        org.update(invalid_key="should_be_ignored")
        assert not hasattr(org, "invalid_key")

    def test_update_updates_timestamp(self):
        org = Organization(name="Acme", owner_id="u1")
        before = org.updated_at
        org.update(name="New")
        assert org.updated_at > before


class TestOrganizationSettings:
    def test_default_settings(self):
        settings = OrgSettings()
        assert settings.max_members == 100
        assert settings.enable_sharing is True
        assert settings.enable_sso is False

    def test_update_settings(self):
        org = Organization(name="Acme", owner_id="u1")
        org.update_settings(max_members=50, enable_sso=True)
        assert org.settings.max_members == 50
        assert org.settings.enable_sso is True

    def test_settings_to_dict(self):
        settings = OrgSettings(max_members=200)
        d = settings.to_dict()
        assert d["max_members"] == 200
        assert d["enable_sharing"] is True

    def test_settings_from_dict(self):
        d = {"max_members": 50, "enable_sso": True}
        settings = OrgSettings.from_dict(d)
        assert settings.max_members == 50
        assert settings.enable_sso is True

    def test_settings_roundtrip(self):
        original = OrgSettings(max_members=42, custom_domains=["example.com"])
        restored = OrgSettings.from_dict(original.to_dict())
        assert restored.max_members == 42
        assert restored.custom_domains == ["example.com"]


class TestOrganizationSerialization:
    def test_to_dict(self):
        org = Organization(name="Acme", owner_id="u1")
        d = org.to_dict()
        assert d["name"] == "Acme"
        assert d["slug"] == "acme"
        assert d["status"] == "active"
        assert "settings" in d
        assert "created_at" in d


# =============================================================================
# OrganizationMember
# =============================================================================

class TestMemberCreation:
    def test_create_defaults(self):
        member = OrganizationMember(user_id="u1", org_id="org_1")
        assert member.user_id == "u1"
        assert member.org_id == "org_1"
        assert member.role == MemberRole.VIEWER
        assert member.status == MembershipStatus.ACTIVE

    def test_create_with_role(self):
        member = OrganizationMember(user_id="u1", org_id="org_1", role=MemberRole.ADMIN)
        assert member.role == MemberRole.ADMIN

    def test_create_with_status(self):
        member = OrganizationMember(
            user_id="u1", org_id="org_1", status=MembershipStatus.PENDING
        )
        assert member.status == MembershipStatus.PENDING


class TestMemberRoleComparison:
    def test_owner_greatest(self):
        assert MemberRole.OWNER > MemberRole.ADMIN
        assert MemberRole.OWNER > MemberRole.EDITOR
        assert MemberRole.OWNER > MemberRole.VIEWER
        assert MemberRole.OWNER > MemberRole.GUEST

    def test_admin_greater_than_editor(self):
        assert MemberRole.ADMIN > MemberRole.EDITOR
        assert MemberRole.ADMIN >= MemberRole.EDITOR

    def test_editor_greater_than_viewer(self):
        assert MemberRole.EDITOR > MemberRole.VIEWER

    def test_viewer_greater_than_guest(self):
        assert MemberRole.VIEWER > MemberRole.GUEST

    def test_equal_roles(self):
        assert MemberRole.EDITOR >= MemberRole.EDITOR
        assert MemberRole.EDITOR <= MemberRole.EDITOR

    def test_guest_least(self):
        assert MemberRole.GUEST < MemberRole.VIEWER
        assert MemberRole.GUEST < MemberRole.EDITOR
        assert MemberRole.GUEST < MemberRole.ADMIN
        assert MemberRole.GUEST < MemberRole.OWNER


class TestMemberStatus:
    def test_is_active(self):
        member = OrganizationMember(user_id="u1", org_id="org_1")
        assert member.is_active() is True
        member.update_status(MembershipStatus.SUSPENDED)
        assert member.is_active() is False
        member.update_status(MembershipStatus.REVOKED)
        assert member.is_active() is False

    def test_update_status(self):
        member = OrganizationMember(user_id="u1", org_id="org_1")
        member.update_status(MembershipStatus.PENDING)
        assert member.status == MembershipStatus.PENDING

    def test_update_status_updates_timestamp(self):
        member = OrganizationMember(user_id="u1", org_id="org_1")
        before = member.updated_at
        import time
        time.sleep(0.001)
        member.update_status(MembershipStatus.SUSPENDED)
        assert member.updated_at >= before


class TestMemberPermissions:
    def test_can_with_sufficient_role(self):
        member = OrganizationMember(user_id="u1", org_id="org_1", role=MemberRole.ADMIN)
        assert member.can(MemberRole.EDITOR) is True
        assert member.can(MemberRole.VIEWER) is True

    def test_can_with_insufficient_role(self):
        member = OrganizationMember(user_id="u1", org_id="org_1", role=MemberRole.VIEWER)
        assert member.can(MemberRole.EDITOR) is False
        assert member.can(MemberRole.ADMIN) is False

    def test_can_inactive_member(self):
        member = OrganizationMember(
            user_id="u1", org_id="org_1", role=MemberRole.OWNER, status=MembershipStatus.SUSPENDED
        )
        assert member.can(MemberRole.VIEWER) is False

    def test_update_role(self):
        member = OrganizationMember(user_id="u1", org_id="org_1", role=MemberRole.VIEWER)
        member.update_role(MemberRole.EDITOR)
        assert member.role == MemberRole.EDITOR

    def test_update_role_updates_timestamp(self):
        member = OrganizationMember(user_id="u1", org_id="org_1", role=MemberRole.VIEWER)
        before = member.updated_at
        import time; time.sleep(0.001)
        member.update_role(MemberRole.EDITOR)
        assert member.updated_at > before


class TestMemberSerialization:
    def test_to_dict(self):
        member = OrganizationMember(user_id="u1", org_id="org_1", role=MemberRole.ADMIN)
        d = member.to_dict()
        assert d["user_id"] == "u1"
        assert d["org_id"] == "org_1"
        assert d["role"] == "admin"
        assert d["status"] == "active"


# =============================================================================
# OrgManager — Organization CRUD
# =============================================================================

class TestOrgManagerCreate:
    def test_create_org(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme Corp", owner_id="u1")
        assert org.name == "Acme Corp"
        assert org.owner_id == "u1"
        assert org.slug == "acme-corp"
        assert org.is_active()

    def test_create_org_with_settings(self):
        mgr = OrgManager()
        settings = OrgSettings(max_members=50)
        org = mgr.create_org("Acme", owner_id="u1", settings=settings)
        assert org.settings.max_members == 50

    def test_create_org_with_metadata(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1", metadata={"industry": "tech"})
        assert org.metadata["industry"] == "tech"

    def test_create_org_owner_is_member(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        member = mgr.get_member(org.org_id, "u1")
        assert member is not None
        assert member.role == MemberRole.OWNER
        assert member.status == MembershipStatus.ACTIVE


class TestOrgManagerGet:
    def test_get_org(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        fetched = mgr.get_org(org.org_id)
        assert fetched is not None
        assert fetched.org_id == org.org_id

    def test_get_org_missing(self):
        mgr = OrgManager()
        assert mgr.get_org("nonexistent") is None


class TestOrgManagerList:
    def test_list_all_orgs(self):
        mgr = OrgManager()
        org1 = mgr.create_org("Org1", owner_id="u1")
        org2 = mgr.create_org("Org2", owner_id="u2")
        orgs = mgr.list_orgs()
        assert len(orgs) == 2

    def test_list_by_user(self):
        mgr = OrgManager()
        org1 = mgr.create_org("Org1", owner_id="u1")
        mgr.create_org("Org2", owner_id="u2")
        orgs = mgr.list_orgs(user_id="u1")
        assert len(orgs) == 1
        assert orgs[0].org_id == org1.org_id

    def test_list_by_status(self):
        mgr = OrgManager()
        org1 = mgr.create_org("Org1", owner_id="u1")
        org2 = mgr.create_org("Org2", owner_id="u2")
        mgr.update_org(org2.org_id, status=OrgStatus.SUSPENDED)
        active = mgr.list_orgs(status=OrgStatus.ACTIVE)
        suspended = mgr.list_orgs(status=OrgStatus.SUSPENDED)
        assert len(active) == 1
        assert len(suspended) == 1

    def test_list_combined_filters(self):
        mgr = OrgManager()
        org1 = mgr.create_org("Org1", owner_id="u1")
        org2 = mgr.create_org("Org2", owner_id="u1")
        mgr.update_org(org2.org_id, status=OrgStatus.SUSPENDED)
        result = mgr.list_orgs(user_id="u1", status=OrgStatus.ACTIVE)
        assert len(result) == 1
        assert result[0].org_id == org1.org_id


class TestOrgManagerUpdate:
    def test_update_org_name(self):
        mgr = OrgManager()
        org = mgr.create_org("Old Name", owner_id="u1")
        updated = mgr.update_org(org.org_id, name="New Name")
        assert updated.name == "New Name"
        assert updated.slug == "new-name"

    def test_update_org_status(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        updated = mgr.update_org(org.org_id, status=OrgStatus.SUSPENDED)
        assert updated.status == OrgStatus.SUSPENDED

    def test_update_org_not_found(self):
        mgr = OrgManager()
        with pytest.raises(OrgNotFoundError):
            mgr.update_org("nonexistent", name="New")


class TestOrgManagerUpdateSettings:
    def test_update_settings(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        updated = mgr.update_org_settings(org.org_id, max_members=200, enable_sso=True)
        assert updated.settings.max_members == 200
        assert updated.settings.enable_sso is True

    def test_update_settings_not_found(self):
        mgr = OrgManager()
        with pytest.raises(OrgNotFoundError):
            mgr.update_org_settings("nonexistent", max_members=50)


class TestOrgManagerDelete:
    def test_delete_org(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        assert mgr.delete_org(org.org_id) is True
        assert mgr.get_org(org.org_id) is None
        assert mgr.get_member(org.org_id, "u1") is None

    def test_delete_org_missing(self):
        mgr = OrgManager()
        assert mgr.delete_org("nonexistent") is False


# =============================================================================
# OrgManager — Member Management
# =============================================================================

class TestOrgManagerAddMember:
    def test_add_member(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        member = mgr.add_member(org.org_id, "u2", MemberRole.EDITOR, invited_by="u1")
        assert member.user_id == "u2"
        assert member.role == MemberRole.EDITOR
        assert member.invited_by == "u1"

    def test_add_member_default_role(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        member = mgr.add_member(org.org_id, "u2")
        assert member.role == MemberRole.VIEWER

    def test_add_duplicate_member(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2")
        with pytest.raises(DuplicateMemberError):
            mgr.add_member(org.org_id, "u2")

    def test_add_member_org_not_found(self):
        mgr = OrgManager()
        with pytest.raises(OrgNotFoundError):
            mgr.add_member("nonexistent", "u2")

    def test_add_second_owner_raises(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        with pytest.raises(RoleAssignmentError):
            mgr.add_member(org.org_id, "u2", MemberRole.OWNER)


class TestOrgManagerRemoveMember:
    def test_remove_member(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2")
        assert mgr.remove_member(org.org_id, "u2") is True
        assert mgr.get_member(org.org_id, "u2") is None

    def test_remove_owner_fails(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        assert mgr.remove_member(org.org_id, "u1") is False
        assert mgr.get_member(org.org_id, "u1") is not None

    def test_remove_missing_member(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        assert mgr.remove_member(org.org_id, "u2") is False

    def test_remove_member_org_not_found(self):
        mgr = OrgManager()
        with pytest.raises(OrgNotFoundError):
            mgr.remove_member("nonexistent", "u2")


class TestOrgManagerGetMember:
    def test_get_member(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.EDITOR)
        member = mgr.get_member(org.org_id, "u2")
        assert member is not None
        assert member.role == MemberRole.EDITOR

    def test_get_member_missing(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        assert mgr.get_member(org.org_id, "u2") is None


class TestOrgManagerUpdateMemberRole:
    def test_update_role(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.VIEWER)
        updated = mgr.update_member_role(org.org_id, "u2", MemberRole.EDITOR)
        assert updated.role == MemberRole.EDITOR

    def test_promote_to_admin(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.EDITOR)
        updated = mgr.update_member_role(org.org_id, "u2", MemberRole.ADMIN)
        assert updated.role == MemberRole.ADMIN

    def test_demote_owner_fails(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        with pytest.raises(RoleAssignmentError):
            mgr.update_member_role(org.org_id, "u1", MemberRole.ADMIN)

    def test_promote_to_owner(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.ADMIN)
        # First transfer ownership away from u1
        mgr.transfer_ownership(org.org_id, "u2")
        updated = mgr.update_member_role(org.org_id, "u2", MemberRole.OWNER)
        assert updated.role == MemberRole.OWNER

    def test_promote_to_owner_when_owner_exists_fails(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.ADMIN)
        with pytest.raises(RoleAssignmentError):
            mgr.update_member_role(org.org_id, "u2", MemberRole.OWNER)

    def test_update_role_member_not_found(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        with pytest.raises(MemberNotFoundError):
            mgr.update_member_role(org.org_id, "u2", MemberRole.EDITOR)


class TestOrgManagerUpdateMemberStatus:
    def test_update_status(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2")
        updated = mgr.update_member_status(org.org_id, "u2", MembershipStatus.SUSPENDED)
        assert updated.status == MembershipStatus.SUSPENDED

    def test_update_status_member_not_found(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        with pytest.raises(MemberNotFoundError):
            mgr.update_member_status(org.org_id, "u2", MembershipStatus.SUSPENDED)


class TestOrgManagerListMembers:
    def test_list_members(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.EDITOR)
        mgr.add_member(org.org_id, "u3", MemberRole.VIEWER)
        members = mgr.list_members(org.org_id)
        assert len(members) == 3  # owner + 2

    def test_list_members_by_status(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2")
        mgr.update_member_status(org.org_id, "u2", MembershipStatus.SUSPENDED)
        active = mgr.list_members(org.org_id, status=MembershipStatus.ACTIVE)
        suspended = mgr.list_members(org.org_id, status=MembershipStatus.SUSPENDED)
        assert len(active) == 1  # owner
        assert len(suspended) == 1

    def test_list_members_by_role(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.EDITOR)
        mgr.add_member(org.org_id, "u3", MemberRole.VIEWER)
        editors = mgr.list_members(org.org_id, role=MemberRole.EDITOR)
        viewers = mgr.list_members(org.org_id, role=MemberRole.VIEWER)
        owners = mgr.list_members(org.org_id, role=MemberRole.OWNER)
        assert len(editors) == 1
        assert len(viewers) == 1
        assert len(owners) == 1

    def test_list_members_org_not_found(self):
        mgr = OrgManager()
        with pytest.raises(OrgNotFoundError):
            mgr.list_members("nonexistent")


class TestOrgManagerListUserOrgs:
    def test_list_user_orgs(self):
        mgr = OrgManager()
        org1 = mgr.create_org("Org1", owner_id="u1")
        org2 = mgr.create_org("Org2", owner_id="u1")
        mgr.create_org("Org3", owner_id="u2")
        orgs = mgr.list_user_orgs("u1")
        assert len(orgs) == 2
        ids = {o.org_id for o in orgs}
        assert org1.org_id in ids
        assert org2.org_id in ids


# =============================================================================
# OrgManager — Permissions
# =============================================================================

class TestOrgManagerPermissions:
    def test_check_permission(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.EDITOR)
        mgr.add_member(org.org_id, "u3", MemberRole.VIEWER)
        assert mgr.check_permission(org.org_id, "u1", MemberRole.OWNER) is True
        assert mgr.check_permission(org.org_id, "u2", MemberRole.EDITOR) is True
        assert mgr.check_permission(org.org_id, "u2", MemberRole.ADMIN) is False
        assert mgr.check_permission(org.org_id, "u3", MemberRole.EDITOR) is False
        assert mgr.check_permission(org.org_id, "u4", MemberRole.VIEWER) is False

    def test_is_owner(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.ADMIN)
        assert mgr.is_owner(org.org_id, "u1") is True
        assert mgr.is_owner(org.org_id, "u2") is False

    def test_is_admin(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.ADMIN)
        mgr.add_member(org.org_id, "u3", MemberRole.EDITOR)
        assert mgr.is_admin(org.org_id, "u1") is True
        assert mgr.is_admin(org.org_id, "u2") is True
        assert mgr.is_admin(org.org_id, "u3") is False

    def test_can_edit(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.EDITOR)
        mgr.add_member(org.org_id, "u3", MemberRole.VIEWER)
        assert mgr.can_edit(org.org_id, "u1") is True
        assert mgr.can_edit(org.org_id, "u2") is True
        assert mgr.can_edit(org.org_id, "u3") is False

    def test_can_view(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.VIEWER)
        assert mgr.can_view(org.org_id, "u1") is True
        assert mgr.can_view(org.org_id, "u2") is True
        assert mgr.can_view(org.org_id, "u3") is False

    def test_suspended_member_cannot_view(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.VIEWER)
        mgr.update_member_status(org.org_id, "u2", MembershipStatus.SUSPENDED)
        assert mgr.can_view(org.org_id, "u2") is False


# =============================================================================
# OrgManager — Ownership Transfer
# =============================================================================

class TestOrgManagerTransferOwnership:
    def test_transfer_ownership(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.ADMIN)
        updated = mgr.transfer_ownership(org.org_id, "u2")
        assert updated.owner_id == "u2"
        # Old owner demoted to admin
        old_owner = mgr.get_member(org.org_id, "u1")
        assert old_owner is not None
        assert old_owner.role == MemberRole.ADMIN
        # New owner promoted
        new_owner = mgr.get_member(org.org_id, "u2")
        assert new_owner.role == MemberRole.OWNER

    def test_transfer_ownership_with_validation(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.ADMIN)
        updated = mgr.transfer_ownership(org.org_id, "u2", current_owner_id="u1")
        assert updated.owner_id == "u2"

    def test_transfer_ownership_wrong_current_owner(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.ADMIN)
        with pytest.raises(RoleAssignmentError):
            mgr.transfer_ownership(org.org_id, "u2", current_owner_id="u3")

    def test_transfer_ownership_to_non_member(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        with pytest.raises(MemberNotFoundError):
            mgr.transfer_ownership(org.org_id, "u2")


# =============================================================================
# OrgManager — Statistics
# =============================================================================

class TestOrgManagerStats:
    def test_get_org_stats(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2", MemberRole.EDITOR)
        mgr.add_member(org.org_id, "u3")
        mgr.update_member_status(org.org_id, "u3", MembershipStatus.PENDING)
        stats = mgr.get_org_stats(org.org_id)
        assert stats["name"] == "Acme"
        assert stats["total_members"] == 3
        assert stats["active_members"] == 2
        assert stats["pending_members"] == 1
        assert stats["suspended_members"] == 0

    def test_get_org_stats_not_found(self):
        mgr = OrgManager()
        with pytest.raises(OrgNotFoundError):
            mgr.get_org_stats("nonexistent")

    def test_get_member_count(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2")
        assert mgr.get_member_count(org.org_id) == 2

    def test_get_active_member_count(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2")
        mgr.update_member_status(org.org_id, "u2", MembershipStatus.SUSPENDED)
        assert mgr.get_active_member_count(org.org_id) == 1


# =============================================================================
# OrgManager — SQLite Persistence
# =============================================================================

class TestOrgManagerPersistence:
    def test_persist_org(self, tmp_path):
        db_path = str(tmp_path / "orgs.db")
        mgr = OrgManager(db_path=db_path)
        org = mgr.create_org("Acme", owner_id="u1", metadata={"key": "val"})
        mgr.add_member(org.org_id, "u2", MemberRole.EDITOR, invited_by="u1")

        # Create new manager pointing to same DB
        mgr2 = OrgManager(db_path=db_path)
        # Note: data is not auto-loaded; would need explicit load in real usage
        # But we can verify via direct SQLite query
        import sqlite3
        conn = sqlite3.connect(db_path)
        org_rows = conn.execute("SELECT name, owner_id FROM organizations").fetchall()
        member_rows = conn.execute(
            "SELECT user_id, role FROM org_members"
        ).fetchall()
        conn.close()
        assert len(org_rows) == 1
        assert org_rows[0] == ("Acme", "u1")
        assert len(member_rows) == 2
        roles = {r[1] for r in member_rows}
        assert roles == {"owner", "editor"}

    def test_delete_org_removes_from_db(self, tmp_path):
        db_path = str(tmp_path / "orgs.db")
        mgr = OrgManager(db_path=db_path)
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.delete_org(org.org_id)
        import sqlite3
        conn = sqlite3.connect(db_path)
        org_rows = conn.execute("SELECT * FROM organizations").fetchall()
        member_rows = conn.execute("SELECT * FROM org_members").fetchall()
        conn.close()
        assert len(org_rows) == 0
        assert len(member_rows) == 0

    def test_remove_member_removes_from_db(self, tmp_path):
        db_path = str(tmp_path / "orgs.db")
        mgr = OrgManager(db_path=db_path)
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(org.org_id, "u2")
        mgr.remove_member(org.org_id, "u2")
        import sqlite3
        conn = sqlite3.connect(db_path)
        member_rows = conn.execute(
            "SELECT * FROM org_members WHERE user_id = ?", ("u2",)
        ).fetchall()
        conn.close()
        assert len(member_rows) == 0


# =============================================================================
# OrgManager — Thread Safety
# =============================================================================

class TestOrgManagerThreadSafety:
    def test_concurrent_create(self):
        mgr = OrgManager()
        errors: list[Exception] = []

        def worker(start: int, count: int):
            try:
                for i in range(start, start + count):
                    mgr.create_org(f"Org-{i}", owner_id=f"u{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i * 10, 10)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(mgr.list_orgs()) == 50

    def test_concurrent_add_members(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u0")
        errors: list[Exception] = []

        def worker(start: int, count: int):
            try:
                for i in range(start, start + count):
                    mgr.add_member(org.org_id, f"u{i}", MemberRole.VIEWER)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i * 10 + 1, 10)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert mgr.get_member_count(org.org_id) == 51  # owner + 50


# =============================================================================
# Edge Cases & Integration
# =============================================================================

class TestOrgManagerEdgeCases:
    def test_empty_org_name(self):
        mgr = OrgManager()
        org = mgr.create_org("", owner_id="u1")
        assert org.name == ""
        assert org.slug == ""

    def test_org_with_long_name(self):
        mgr = OrgManager()
        long_name = "A" * 100
        org = mgr.create_org(long_name, owner_id="u1")
        assert org.name == long_name
        assert len(org.slug) <= 50

    def test_multiple_orgs_same_owner(self):
        mgr = OrgManager()
        org1 = mgr.create_org("Org1", owner_id="u1")
        org2 = mgr.create_org("Org2", owner_id="u1")
        orgs = mgr.list_user_orgs("u1")
        assert len(orgs) == 2

    def test_member_joins_multiple_orgs(self):
        mgr = OrgManager()
        org1 = mgr.create_org("Org1", owner_id="u1")
        org2 = mgr.create_org("Org2", owner_id="u2")
        mgr.add_member(org1.org_id, "u3", MemberRole.EDITOR)
        mgr.add_member(org2.org_id, "u3", MemberRole.VIEWER)
        assert mgr.get_member(org1.org_id, "u3").role == MemberRole.EDITOR
        assert mgr.get_member(org2.org_id, "u3").role == MemberRole.VIEWER

    def test_pending_member_cannot_access(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(
            org.org_id, "u2", MemberRole.VIEWER, status=MembershipStatus.PENDING
        )
        assert mgr.can_view(org.org_id, "u2") is False

    def test_revoked_member_cannot_access(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.add_member(
            org.org_id, "u2", MemberRole.VIEWER, status=MembershipStatus.ACTIVE
        )
        mgr.update_member_status(org.org_id, "u2", MembershipStatus.REVOKED)
        assert mgr.can_view(org.org_id, "u2") is False

    def test_owner_can_do_everything(self):
        mgr = OrgManager()
        org = mgr.create_org("Acme", owner_id="u1")
        assert mgr.is_owner(org.org_id, "u1")
        assert mgr.is_admin(org.org_id, "u1")
        assert mgr.can_edit(org.org_id, "u1")
        assert mgr.can_view(org.org_id, "u1")

    def test_settings_persisted_on_update(self, tmp_path):
        db_path = str(tmp_path / "orgs.db")
        mgr = OrgManager(db_path=db_path)
        org = mgr.create_org("Acme", owner_id="u1")
        mgr.update_org_settings(
            org.org_id,
            max_members=500,
            enable_sso=True,
            custom_domains=["acme.com"],
        )
        import sqlite3
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT settings FROM organizations WHERE org_id = ?", (org.org_id,)
        ).fetchone()
        conn.close()
        import json
        settings = json.loads(row[0])
        assert settings["max_members"] == 500
        assert settings["enable_sso"] is True
        assert settings["custom_domains"] == ["acme.com"]
