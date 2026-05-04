"""SpaceManager — CRUD operations and member management for spaces."""
from __future__ import annotations

from typing import Any, Protocol

from neuralmem.spaces.space import Space, SpaceVisibility
from neuralmem.spaces.membership import SpaceMembership, SpaceRole


class _SpaceStore(Protocol):
    """Protocol for space persistence layer (mock-testable)."""

    def save_space(self, space: Space) -> Space:
        ...

    def get_space(self, space_id: str) -> Space | None:
        ...

    def delete_space(self, space_id: str) -> bool:
        ...

    def list_spaces(
        self,
        owner: str | None = None,
        visibility: SpaceVisibility | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Space]:
        ...

    def save_membership(self, membership: SpaceMembership) -> SpaceMembership:
        ...

    def get_membership(self, space_id: str, user_id: str) -> SpaceMembership | None:
        ...

    def delete_membership(self, space_id: str, user_id: str) -> bool:
        ...

    def list_memberships(self, space_id: str) -> list[SpaceMembership]:
        ...


class SpaceNotFoundError(Exception):
    """Raised when a space does not exist."""
    pass


class SpaceAlreadyExistsError(Exception):
    """Raised when creating a space with a duplicate name for the same owner."""
    pass


class MemberNotFoundError(Exception):
    """Raised when a membership does not exist."""
    pass


class PermissionDeniedError(Exception):
    """Raised when a user lacks permission for an action."""
    pass


class SpaceManager:
    """Manages spaces and their memberships.

    Uses a pluggable store conforming to _SpaceStore for persistence,
    making it fully mock-testable without real storage.
    """

    def __init__(self, store: _SpaceStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Space CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        owner: str,
        description: str = "",
        visibility: SpaceVisibility = SpaceVisibility.PRIVATE,
        settings: dict[str, Any] | None = None,
    ) -> Space:
        """Create a new space and auto-add the owner as a member.

        Args:
            name: Display name (1-128 chars).
            owner: User ID of the owner.
            description: Optional description (max 2048 chars).
            visibility: Visibility level.
            settings: Arbitrary space settings dict.

        Returns:
            The created Space.

        Raises:
            SpaceAlreadyExistsError: If owner already has a space with this name.
        """
        # Check for duplicate name under same owner
        existing = self._store.list_spaces(owner=owner)
        if any(s.name == name for s in existing):
            raise SpaceAlreadyExistsError(
                f"Space '{name}' already exists for owner '{owner}'"
            )

        space = Space(
            name=name,
            owner=owner,
            description=description,
            visibility=visibility,
            settings=settings or {},
        )
        saved = self._store.save_space(space)

        # Auto-add owner as member with OWNER role
        self._store.save_membership(
            SpaceMembership(
                space_id=saved.id,
                user_id=owner,
                role=SpaceRole.OWNER,
            )
        )
        return saved

    def get(self, space_id: str) -> Space:
        """Get a space by ID.

        Raises:
            SpaceNotFoundError: If the space does not exist.
        """
        space = self._store.get_space(space_id)
        if space is None:
            raise SpaceNotFoundError(f"Space '{space_id}' not found")
        return space

    def update(
        self,
        space_id: str,
        actor_id: str,
        name: str | None = None,
        description: str | None = None,
        visibility: SpaceVisibility | None = None,
        settings: dict[str, Any] | None = None,
    ) -> Space:
        """Update a space's fields.

        Args:
            space_id: ID of the space to update.
            actor_id: User ID performing the update (for permission check).
            name, description, visibility, settings: Fields to update.

        Returns:
            The updated Space.

        Raises:
            SpaceNotFoundError: If the space does not exist.
            PermissionDeniedError: If actor lacks permission.
        """
        space = self.get(space_id)
        self._require_permission(space_id, actor_id, SpaceRole.can_change_settings)

        updates: dict[str, Any] = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        if visibility is not None:
            updates["visibility"] = visibility
        if settings is not None:
            updates["settings"] = settings

        if not updates:
            return space

        updated = space.with_fields(**updates)
        return self._store.save_space(updated)

    def delete(self, space_id: str, actor_id: str) -> bool:
        """Delete a space (owner only).

        Args:
            space_id: ID of the space to delete.
            actor_id: User ID performing the deletion.

        Returns:
            True if deleted.

        Raises:
            SpaceNotFoundError: If the space does not exist.
            PermissionDeniedError: If actor is not the owner.
        """
        space = self.get(space_id)
        self._require_permission(space_id, actor_id, SpaceRole.can_delete_space)
        return self._store.delete_space(space.id)

    def list_spaces(
        self,
        owner: str | None = None,
        visibility: SpaceVisibility | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Space]:
        """List spaces with optional filters."""
        return self._store.list_spaces(
            owner=owner,
            visibility=visibility,
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # Member management
    # ------------------------------------------------------------------

    def add_member(
        self,
        space_id: str,
        actor_id: str,
        user_id: str,
        role: SpaceRole = SpaceRole.VIEWER,
    ) -> SpaceMembership:
        """Add a member to a space.

        Args:
            space_id: ID of the space.
            actor_id: User ID performing the action (must be admin+).
            user_id: User ID to add.
            role: Role to assign.

        Returns:
            The created SpaceMembership.

        Raises:
            SpaceNotFoundError: If the space does not exist.
            PermissionDeniedError: If actor lacks permission.
        """
        self.get(space_id)  # validate space exists
        self._require_permission(space_id, actor_id, SpaceRole.can_manage_members)

        existing = self._store.get_membership(space_id, user_id)
        if existing is not None:
            # Update role if already a member
            updated = existing.with_role(role)
            return self._store.save_membership(updated)

        membership = SpaceMembership(
            space_id=space_id,
            user_id=user_id,
            role=role,
            invited_by=actor_id,
        )
        return self._store.save_membership(membership)

    def remove_member(
        self,
        space_id: str,
        actor_id: str,
        user_id: str,
    ) -> bool:
        """Remove a member from a space.

        Args:
            space_id: ID of the space.
            actor_id: User ID performing the action.
            user_id: User ID to remove.

        Returns:
            True if removed.

        Raises:
            SpaceNotFoundError: If the space does not exist.
            PermissionDeniedError: If actor lacks permission.
            MemberNotFoundError: If user is not a member.
        """
        self.get(space_id)
        self._require_permission(space_id, actor_id, SpaceRole.can_manage_members)

        membership = self._store.get_membership(space_id, user_id)
        if membership is None:
            raise MemberNotFoundError(
                f"User '{user_id}' is not a member of space '{space_id}'"
            )

        # Prevent removing the owner (must transfer ownership first)
        if membership.role == SpaceRole.OWNER:
            raise PermissionDeniedError(
                "Cannot remove the space owner. Transfer ownership first."
            )

        return self._store.delete_membership(space_id, user_id)

    def get_member(self, space_id: str, user_id: str) -> SpaceMembership:
        """Get a specific membership.

        Raises:
            SpaceNotFoundError: If the space does not exist.
            MemberNotFoundError: If the user is not a member.
        """
        self.get(space_id)
        membership = self._store.get_membership(space_id, user_id)
        if membership is None:
            raise MemberNotFoundError(
                f"User '{user_id}' is not a member of space '{space_id}'"
            )
        return membership

    def list_members(self, space_id: str) -> list[SpaceMembership]:
        """List all members of a space.

        Raises:
            SpaceNotFoundError: If the space does not exist.
        """
        self.get(space_id)
        return self._store.list_memberships(space_id)

    def change_role(
        self,
        space_id: str,
        actor_id: str,
        user_id: str,
        new_role: SpaceRole,
    ) -> SpaceMembership:
        """Change a member's role.

        Args:
            space_id: ID of the space.
            actor_id: User ID performing the action.
            user_id: User ID whose role is changing.
            new_role: New role to assign.

        Returns:
            The updated SpaceMembership.

        Raises:
            SpaceNotFoundError: If the space does not exist.
            PermissionDeniedError: If actor lacks permission.
            MemberNotFoundError: If user is not a member.
        """
        self.get(space_id)
        self._require_permission(space_id, actor_id, SpaceRole.can_manage_members)

        membership = self._store.get_membership(space_id, user_id)
        if membership is None:
            raise MemberNotFoundError(
                f"User '{user_id}' is not a member of space '{space_id}'"
            )

        # Only owner can assign owner role
        if new_role == SpaceRole.OWNER:
            actor = self._store.get_membership(space_id, actor_id)
            if actor is None or actor.role != SpaceRole.OWNER:
                raise PermissionDeniedError("Only the owner can assign ownership")

        updated = membership.with_role(new_role)
        return self._store.save_membership(updated)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_permission(
        self,
        space_id: str,
        actor_id: str,
        permission_check,
    ) -> None:
        """Check that actor_id has the required permission in space_id."""
        membership = self._store.get_membership(space_id, actor_id)
        if membership is None or not permission_check(membership.role):
            raise PermissionDeniedError(
                f"User '{actor_id}' does not have permission for this action"
            )
