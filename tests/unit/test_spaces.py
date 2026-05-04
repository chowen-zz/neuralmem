"""Comprehensive unit tests for the spaces module."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.spaces.space import Space, SpaceVisibility
from neuralmem.spaces.membership import SpaceMembership, SpaceRole
from neuralmem.spaces.manager import (
    SpaceManager,
    SpaceNotFoundError,
    SpaceAlreadyExistsError,
    MemberNotFoundError,
    PermissionDeniedError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    """A fully functional in-memory mock store."""
    store = MagicMock()
    _spaces: dict[str, Space] = {}
    _memberships: dict[tuple[str, str], SpaceMembership] = {}

    def save_space(space: Space) -> Space:
        _spaces[space.id] = space
        return space

    def get_space(space_id: str) -> Space | None:
        return _spaces.get(space_id)

    def delete_space(space_id: str) -> bool:
        return _spaces.pop(space_id, None) is not None

    def list_spaces(
        owner=None, visibility=None, limit=100, offset=0
    ) -> list[Space]:
        results = list(_spaces.values())
        if owner is not None:
            results = [s for s in results if s.owner == owner]
        if visibility is not None:
            results = [s for s in results if s.visibility == visibility]
        return results[offset:offset + limit]

    def save_membership(m: SpaceMembership) -> SpaceMembership:
        _memberships[(m.space_id, m.user_id)] = m
        return m

    def get_membership(space_id: str, user_id: str) -> SpaceMembership | None:
        return _memberships.get((space_id, user_id))

    def delete_membership(space_id: str, user_id: str) -> bool:
        return _memberships.pop((space_id, user_id), None) is not None

    def list_memberships(space_id: str) -> list[SpaceMembership]:
        return [
            m for (sid, _), m in _memberships.items() if sid == space_id
        ]

    store.save_space.side_effect = save_space
    store.get_space.side_effect = get_space
    store.delete_space.side_effect = delete_space
    store.list_spaces.side_effect = list_spaces
    store.save_membership.side_effect = save_membership
    store.get_membership.side_effect = get_membership
    store.delete_membership.side_effect = delete_membership
    store.list_memberships.side_effect = list_memberships

    # Attach internal dicts for test inspection
    store._spaces = _spaces
    store._memberships = _memberships
    return store


@pytest.fixture
def manager(mock_store):
    """SpaceManager backed by the mock store."""
    return SpaceManager(mock_store)


@pytest.fixture
def sample_space(manager):
    """A pre-created space with owner 'user-1'."""
    return manager.create(name="Test Space", owner="user-1", description="A test space")


# ---------------------------------------------------------------------------
# SpaceRole tests
# ---------------------------------------------------------------------------

class TestSpaceRole:
    def test_role_permissions(self):
        assert SpaceRole.can_manage_members(SpaceRole.OWNER) is True
        assert SpaceRole.can_manage_members(SpaceRole.ADMIN) is True
        assert SpaceRole.can_manage_members(SpaceRole.EDITOR) is False
        assert SpaceRole.can_manage_members(SpaceRole.VIEWER) is False

    def test_edit_permissions(self):
        assert SpaceRole.can_edit(SpaceRole.OWNER) is True
        assert SpaceRole.can_edit(SpaceRole.ADMIN) is True
        assert SpaceRole.can_edit(SpaceRole.EDITOR) is True
        assert SpaceRole.can_edit(SpaceRole.VIEWER) is False

    def test_delete_space_permission(self):
        assert SpaceRole.can_delete_space(SpaceRole.OWNER) is True
        assert SpaceRole.can_delete_space(SpaceRole.ADMIN) is False

    def test_change_settings_permission(self):
        assert SpaceRole.can_change_settings(SpaceRole.OWNER) is True
        assert SpaceRole.can_change_settings(SpaceRole.ADMIN) is True
        assert SpaceRole.can_change_settings(SpaceRole.EDITOR) is False


# ---------------------------------------------------------------------------
# Space dataclass tests
# ---------------------------------------------------------------------------

class TestSpace:
    def test_space_creation_defaults(self):
        space = Space(name="My Space", owner="user-1")
        assert space.name == "My Space"
        assert space.owner == "user-1"
        assert space.visibility == SpaceVisibility.PRIVATE
        assert space.description == ""
        assert space.settings == {}
        assert len(space.id) == 26  # ULID length
        assert space.memory_ids == ()

    def test_space_with_all_fields(self):
        space = Space(
            name="Full Space",
            owner="user-2",
            description="Detailed description",
            visibility=SpaceVisibility.PUBLIC,
            settings={"theme": "dark", "retention_days": 30},
        )
        assert space.visibility == SpaceVisibility.PUBLIC
        assert space.settings == {"theme": "dark", "retention_days": 30}

    def test_space_immutable_update(self):
        space = Space(name="Original", owner="user-1")
        updated = space.with_fields(name="Updated", description="New desc")
        assert updated.name == "Updated"
        assert updated.description == "New desc"
        assert updated.id != space.id  # new ID
        assert space.name == "Original"  # original unchanged

    def test_space_validation_name_required(self):
        with pytest.raises(Exception):  # pydantic validation error
            Space(name="", owner="user-1")

    def test_space_validation_owner_required(self):
        with pytest.raises(Exception):
            Space(name="Test", owner="")


# ---------------------------------------------------------------------------
# SpaceMembership tests
# ---------------------------------------------------------------------------

class TestSpaceMembership:
    def test_membership_defaults(self):
        m = SpaceMembership(space_id="sp-1", user_id="user-1")
        assert m.role == SpaceRole.VIEWER
        assert m.invited_by is None
        assert len(m.id) == 26

    def test_membership_with_role(self):
        m = SpaceMembership(space_id="sp-1", user_id="user-1", role=SpaceRole.ADMIN)
        assert m.role == SpaceRole.ADMIN

    def test_membership_immutable_role_update(self):
        m = SpaceMembership(space_id="sp-1", user_id="user-1", role=SpaceRole.VIEWER)
        updated = m.with_role(SpaceRole.EDITOR)
        assert updated.role == SpaceRole.EDITOR
        assert updated.space_id == m.space_id
        assert updated.user_id == m.user_id
        assert updated.id != m.id  # new ID
        assert m.role == SpaceRole.VIEWER  # original unchanged


# ---------------------------------------------------------------------------
# SpaceManager.create tests
# ---------------------------------------------------------------------------

class TestCreateSpace:
    def test_create_basic(self, manager):
        space = manager.create(name="My Project", owner="user-1")
        assert space.name == "My Project"
        assert space.owner == "user-1"
        assert space.visibility == SpaceVisibility.PRIVATE

    def test_create_with_options(self, manager):
        space = manager.create(
            name="Public Project",
            owner="user-1",
            description="A public project space",
            visibility=SpaceVisibility.PUBLIC,
            settings={"key": "value"},
        )
        assert space.description == "A public project space"
        assert space.visibility == SpaceVisibility.PUBLIC
        assert space.settings == {"key": "value"}

    def test_create_auto_adds_owner_membership(self, manager, mock_store):
        space = manager.create(name="Auto Member", owner="user-1")
        membership = mock_store.get_membership(space.id, "user-1")
        assert membership is not None
        assert membership.role == SpaceRole.OWNER

    def test_create_duplicate_name_same_owner_raises(self, manager):
        manager.create(name="Duplicate", owner="user-1")
        with pytest.raises(SpaceAlreadyExistsError):
            manager.create(name="Duplicate", owner="user-1")

    def test_create_same_name_different_owner_ok(self, manager):
        s1 = manager.create(name="Shared Name", owner="user-1")
        s2 = manager.create(name="Shared Name", owner="user-2")
        assert s1.id != s2.id
        assert s1.owner == "user-1"
        assert s2.owner == "user-2"


# ---------------------------------------------------------------------------
# SpaceManager.get / update / delete tests
# ---------------------------------------------------------------------------

class TestGetUpdateDelete:
    def test_get_existing(self, manager, sample_space):
        result = manager.get(sample_space.id)
        assert result.id == sample_space.id
        assert result.name == sample_space.name

    def test_get_nonexistent_raises(self, manager):
        with pytest.raises(SpaceNotFoundError):
            manager.get("nonexistent-id")

    def test_update_name(self, manager, sample_space):
        updated = manager.update(
            sample_space.id, actor_id="user-1", name="New Name"
        )
        assert updated.name == "New Name"
        assert updated.id != sample_space.id  # immutable

    def test_update_description(self, manager, sample_space):
        updated = manager.update(
            sample_space.id, actor_id="user-1", description="Updated desc"
        )
        assert updated.description == "Updated desc"

    def test_update_visibility(self, manager, sample_space):
        updated = manager.update(
            sample_space.id,
            actor_id="user-1",
            visibility=SpaceVisibility.INTERNAL,
        )
        assert updated.visibility == SpaceVisibility.INTERNAL

    def test_update_settings(self, manager, sample_space):
        updated = manager.update(
            sample_space.id,
            actor_id="user-1",
            settings={"foo": "bar"},
        )
        assert updated.settings == {"foo": "bar"}

    def test_update_no_changes_returns_same(self, manager, sample_space):
        updated = manager.update(sample_space.id, actor_id="user-1")
        assert updated.id == sample_space.id

    def test_update_nonexistent_space_raises(self, manager):
        with pytest.raises(SpaceNotFoundError):
            manager.update("fake-id", actor_id="user-1", name="X")

    def test_update_by_non_member_raises(self, manager, sample_space):
        with pytest.raises(PermissionDeniedError):
            manager.update(
                sample_space.id, actor_id="user-2", name="Hacked"
            )

    def test_update_by_viewer_raises(self, manager, sample_space):
        manager.add_member(
            sample_space.id, actor_id="user-1", user_id="viewer-1"
        )
        with pytest.raises(PermissionDeniedError):
            manager.update(
                sample_space.id, actor_id="viewer-1", name="Hacked"
            )

    def test_delete_by_owner(self, manager, sample_space):
        result = manager.delete(sample_space.id, actor_id="user-1")
        assert result is True
        with pytest.raises(SpaceNotFoundError):
            manager.get(sample_space.id)

    def test_delete_by_non_owner_raises(self, manager, sample_space):
        manager.add_member(
            sample_space.id, actor_id="user-1", user_id="admin-1", role=SpaceRole.ADMIN
        )
        with pytest.raises(PermissionDeniedError):
            manager.delete(sample_space.id, actor_id="admin-1")

    def test_delete_nonexistent_raises(self, manager):
        with pytest.raises(SpaceNotFoundError):
            manager.delete("fake-id", actor_id="user-1")


# ---------------------------------------------------------------------------
# SpaceManager.list_spaces tests
# ---------------------------------------------------------------------------

class TestListSpaces:
    def test_list_empty(self, manager):
        assert manager.list_spaces() == []

    def test_list_all(self, manager):
        s1 = manager.create(name="S1", owner="user-1")
        s2 = manager.create(name="S2", owner="user-2")
        results = manager.list_spaces()
        assert len(results) == 2
        assert {s.name for s in results} == {"S1", "S2"}

    def test_list_by_owner(self, manager):
        manager.create(name="A", owner="user-1")
        manager.create(name="B", owner="user-1")
        manager.create(name="C", owner="user-2")
        results = manager.list_spaces(owner="user-1")
        assert len(results) == 2
        assert all(s.owner == "user-1" for s in results)

    def test_list_by_visibility(self, manager):
        manager.create(name="Priv", owner="user-1")
        manager.create(
            name="Pub", owner="user-1", visibility=SpaceVisibility.PUBLIC
        )
        results = manager.list_spaces(visibility=SpaceVisibility.PUBLIC)
        assert len(results) == 1
        assert results[0].name == "Pub"

    def test_list_pagination(self, manager):
        for i in range(5):
            manager.create(name=f"Space-{i}", owner="user-1")
        assert len(manager.list_spaces(limit=2)) == 2
        assert len(manager.list_spaces(limit=2, offset=2)) == 2
        assert len(manager.list_spaces(limit=2, offset=4)) == 1


# ---------------------------------------------------------------------------
# SpaceManager member management tests
# ---------------------------------------------------------------------------

class TestAddMember:
    def test_add_member_by_owner(self, manager, sample_space):
        m = manager.add_member(
            sample_space.id,
            actor_id="user-1",
            user_id="member-1",
            role=SpaceRole.EDITOR,
        )
        assert m.space_id == sample_space.id
        assert m.user_id == "member-1"
        assert m.role == SpaceRole.EDITOR
        assert m.invited_by == "user-1"

    def test_add_member_by_admin(self, manager, sample_space):
        manager.add_member(
            sample_space.id,
            actor_id="user-1",
            user_id="admin-1",
            role=SpaceRole.ADMIN,
        )
        m = manager.add_member(
            sample_space.id,
            actor_id="admin-1",
            user_id="member-1",
        )
        assert m.role == SpaceRole.VIEWER

    def test_add_member_by_editor_raises(self, manager, sample_space):
        manager.add_member(
            sample_space.id,
            actor_id="user-1",
            user_id="editor-1",
            role=SpaceRole.EDITOR,
        )
        with pytest.raises(PermissionDeniedError):
            manager.add_member(
                sample_space.id, actor_id="editor-1", user_id="member-1"
            )

    def test_add_member_by_viewer_raises(self, manager, sample_space):
        manager.add_member(
            sample_space.id,
            actor_id="user-1",
            user_id="viewer-1",
            role=SpaceRole.VIEWER,
        )
        with pytest.raises(PermissionDeniedError):
            manager.add_member(
                sample_space.id, actor_id="viewer-1", user_id="member-1"
            )

    def test_add_existing_member_updates_role(self, manager, sample_space):
        manager.add_member(
            sample_space.id,
            actor_id="user-1",
            user_id="member-1",
            role=SpaceRole.VIEWER,
        )
        updated = manager.add_member(
            sample_space.id,
            actor_id="user-1",
            user_id="member-1",
            role=SpaceRole.EDITOR,
        )
        assert updated.role == SpaceRole.EDITOR

    def test_add_member_to_nonexistent_space_raises(self, manager):
        with pytest.raises(SpaceNotFoundError):
            manager.add_member("fake-id", actor_id="user-1", user_id="member-1")


class TestRemoveMember:
    def test_remove_member_by_owner(self, manager, sample_space):
        manager.add_member(
            sample_space.id, actor_id="user-1", user_id="member-1"
        )
        result = manager.remove_member(
            sample_space.id, actor_id="user-1", user_id="member-1"
        )
        assert result is True
        with pytest.raises(MemberNotFoundError):
            manager.get_member(sample_space.id, "member-1")

    def test_remove_member_by_admin(self, manager, sample_space):
        manager.add_member(
            sample_space.id,
            actor_id="user-1",
            user_id="admin-1",
            role=SpaceRole.ADMIN,
        )
        manager.add_member(
            sample_space.id, actor_id="user-1", user_id="member-1"
        )
        result = manager.remove_member(
            sample_space.id, actor_id="admin-1", user_id="member-1"
        )
        assert result is True

    def test_remove_owner_raises(self, manager, sample_space):
        with pytest.raises(PermissionDeniedError):
            manager.remove_member(
                sample_space.id, actor_id="user-1", user_id="user-1"
            )

    def test_remove_non_member_raises(self, manager, sample_space):
        with pytest.raises(MemberNotFoundError):
            manager.remove_member(
                sample_space.id, actor_id="user-1", user_id="not-a-member"
            )

    def test_remove_by_unauthorized_raises(self, manager, sample_space):
        manager.add_member(
            sample_space.id,
            actor_id="user-1",
            user_id="editor-1",
            role=SpaceRole.EDITOR,
        )
        with pytest.raises(PermissionDeniedError):
            manager.remove_member(
                sample_space.id, actor_id="editor-1", user_id="user-1"
            )


class TestGetMember:
    def test_get_existing(self, manager, sample_space):
        manager.add_member(
            sample_space.id, actor_id="user-1", user_id="member-1"
        )
        m = manager.get_member(sample_space.id, "member-1")
        assert m.user_id == "member-1"

    def test_get_non_member_raises(self, manager, sample_space):
        with pytest.raises(MemberNotFoundError):
            manager.get_member(sample_space.id, "not-a-member")

    def test_get_nonexistent_space_raises(self, manager):
        with pytest.raises(SpaceNotFoundError):
            manager.get_member("fake-id", "user-1")


class TestListMembers:
    def test_list_members(self, manager, sample_space):
        manager.add_member(
            sample_space.id, actor_id="user-1", user_id="m1"
        )
        manager.add_member(
            sample_space.id, actor_id="user-1", user_id="m2"
        )
        members = manager.list_members(sample_space.id)
        assert len(members) == 3  # owner + 2 added
        user_ids = {m.user_id for m in members}
        assert user_ids == {"user-1", "m1", "m2"}

    def test_list_members_nonexistent_space_raises(self, manager):
        with pytest.raises(SpaceNotFoundError):
            manager.list_members("fake-id")


class TestChangeRole:
    def test_change_role_by_owner(self, manager, sample_space):
        manager.add_member(
            sample_space.id, actor_id="user-1", user_id="member-1"
        )
        updated = manager.change_role(
            sample_space.id,
            actor_id="user-1",
            user_id="member-1",
            new_role=SpaceRole.ADMIN,
        )
        assert updated.role == SpaceRole.ADMIN

    def test_change_role_by_admin(self, manager, sample_space):
        manager.add_member(
            sample_space.id,
            actor_id="user-1",
            user_id="admin-1",
            role=SpaceRole.ADMIN,
        )
        manager.add_member(
            sample_space.id, actor_id="user-1", user_id="member-1"
        )
        updated = manager.change_role(
            sample_space.id,
            actor_id="admin-1",
            user_id="member-1",
            new_role=SpaceRole.EDITOR,
        )
        assert updated.role == SpaceRole.EDITOR

    def test_change_role_to_owner_by_owner(self, manager, sample_space):
        manager.add_member(
            sample_space.id, actor_id="user-1", user_id="member-1"
        )
        updated = manager.change_role(
            sample_space.id,
            actor_id="user-1",
            user_id="member-1",
            new_role=SpaceRole.OWNER,
        )
        assert updated.role == SpaceRole.OWNER

    def test_change_role_to_owner_by_admin_raises(self, manager, sample_space):
        manager.add_member(
            sample_space.id,
            actor_id="user-1",
            user_id="admin-1",
            role=SpaceRole.ADMIN,
        )
        manager.add_member(
            sample_space.id, actor_id="user-1", user_id="member-1"
        )
        with pytest.raises(PermissionDeniedError):
            manager.change_role(
                sample_space.id,
                actor_id="admin-1",
                user_id="member-1",
                new_role=SpaceRole.OWNER,
            )

    def test_change_role_by_editor_raises(self, manager, sample_space):
        manager.add_member(
            sample_space.id,
            actor_id="user-1",
            user_id="editor-1",
            role=SpaceRole.EDITOR,
        )
        manager.add_member(
            sample_space.id, actor_id="user-1", user_id="member-1"
        )
        with pytest.raises(PermissionDeniedError):
            manager.change_role(
                sample_space.id,
                actor_id="editor-1",
                user_id="member-1",
                new_role=SpaceRole.ADMIN,
            )

    def test_change_role_non_member_raises(self, manager, sample_space):
        with pytest.raises(MemberNotFoundError):
            manager.change_role(
                sample_space.id,
                actor_id="user-1",
                user_id="not-a-member",
                new_role=SpaceRole.EDITOR,
            )


# ---------------------------------------------------------------------------
# Edge cases and integration-style tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_full_lifecycle(self, manager):
        # Create
        space = manager.create(
            name="Lifecycle Test",
            owner="user-1",
            description="Testing full lifecycle",
            visibility=SpaceVisibility.INTERNAL,
        )
        assert space.visibility == SpaceVisibility.INTERNAL

        # Add members
        manager.add_member(space.id, "user-1", "editor-1", SpaceRole.EDITOR)
        manager.add_member(space.id, "user-1", "viewer-1", SpaceRole.VIEWER)

        # Admin tries to update (should work)
        manager.change_role(space.id, "user-1", "editor-1", SpaceRole.ADMIN)
        updated = manager.update(
            space.id, actor_id="editor-1", description="Updated by admin"
        )
        assert updated.description == "Updated by admin"

        # Viewer tries to update (should fail)
        with pytest.raises(PermissionDeniedError):
            manager.update(space.id, actor_id="viewer-1", description="Hacked")

        # List members
        members = manager.list_members(space.id)
        assert len(members) == 3

        # Change role
        manager.change_role(
            space.id, "user-1", "viewer-1", SpaceRole.ADMIN
        )
        # Now admin can add members
        manager.add_member(space.id, "viewer-1", "new-member")

        # Delete
        assert manager.delete(space.id, "user-1") is True

    def test_multiple_spaces_isolation(self, manager):
        s1 = manager.create(name="Project A", owner="user-1")
        s2 = manager.create(name="Project B", owner="user-1")

        manager.add_member(s1.id, "user-1", "shared-member", SpaceRole.EDITOR)
        manager.add_member(s2.id, "user-1", "shared-member", SpaceRole.VIEWER)

        # Member should have different roles in different spaces
        m1 = manager.get_member(s1.id, "shared-member")
        m2 = manager.get_member(s2.id, "shared-member")
        assert m1.role == SpaceRole.EDITOR
        assert m2.role == SpaceRole.VIEWER

    def test_space_settings_persistence(self, manager):
        space = manager.create(
            name="Settings Test",
            owner="user-1",
            settings={"retention": 7, "auto_archive": True},
        )
        fetched = manager.get(space.id)
        assert fetched.settings == {"retention": 7, "auto_archive": True}

        updated = manager.update(
            space.id,
            actor_id="user-1",
            settings={"retention": 14, "auto_archive": False},
        )
        assert updated.settings == {"retention": 14, "auto_archive": False}
