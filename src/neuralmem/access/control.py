"""AccessControl — check, grant, and revoke permissions for users/resources."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neuralmem.access.permission import Permission
from neuralmem.access.role import Role, get_role, PREDEFINED_ROLES


class AccessDeniedError(Exception):
    """Raised when a user lacks permission for an action on a resource."""
    pass


@dataclass(frozen=True)
class ResourcePermission:
    """A permission grant scoped to a specific resource.

    Attributes:
        user_id: The user this grant applies to.
        resource_id: The resource (memory, space, etc.) this grant applies to.
        permission: The specific permission granted.
        granted_by: User ID who granted this permission.
        metadata: Optional extra data (e.g., expiry, reason).
    """
    user_id: str
    resource_id: str
    permission: Permission
    granted_by: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpacePermission:
    """A permission grant scoped to an entire space.

    Attributes:
        user_id: The user this grant applies to.
        space_id: The space this grant applies to.
        permission: The specific permission granted.
        granted_by: User ID who granted this permission.
        metadata: Optional extra data.
    """
    user_id: str
    space_id: str
    permission: Permission
    granted_by: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AccessControl:
    """Central access-control engine for NeuralMem.

    Maintains user-to-role mappings, resource-level overrides, and
    space-level overrides.  Supports:

    - ``check(user_id, action, resource_id=None, space_id=None)``
    - ``grant(user_id, permission, resource_id=None, space_id=None, granted_by="")``
    - ``revoke(user_id, permission, resource_id=None, space_id=None)``

    The engine resolves permissions in this priority order:
    1. Resource-level explicit grant
    2. Space-level explicit grant
    3. User's role-based permissions
    4. Deny
    """

    def __init__(self) -> None:
        # user_id -> Role
        self._user_roles: dict[str, Role] = {}
        # (user_id, resource_id, permission) -> ResourcePermission
        self._resource_grants: dict[tuple[str, str, Permission], ResourcePermission] = {}
        # (user_id, space_id, permission) -> SpacePermission
        self._space_grants: dict[tuple[str, str, Permission], SpacePermission] = {}

    # ------------------------------------------------------------------
    # Role assignment
    # ------------------------------------------------------------------

    def assign_role(self, user_id: str, role: Role) -> Role:
        """Assign a role to a user.

        Returns:
            The assigned role.
        """
        self._user_roles[user_id] = role
        return role

    def assign_role_by_name(self, user_id: str, role_name: str) -> Role:
        """Assign a predefined role to a user by name.

        Raises:
            ValueError: If the role name is not predefined.
        """
        role = get_role(role_name)
        if role is None:
            raise ValueError(f"Unknown predefined role: {role_name!r}")
        return self.assign_role(user_id, role)

    def get_role(self, user_id: str) -> Role | None:
        """Get the role assigned to a user, if any."""
        return self._user_roles.get(user_id)

    def remove_role(self, user_id: str) -> bool:
        """Remove a user's role assignment.

        Returns:
            True if a role was removed.
        """
        return self._user_roles.pop(user_id, None) is not None

    # ------------------------------------------------------------------
    # Permission checking
    # ------------------------------------------------------------------

    def check(
        self,
        user_id: str,
        action: Permission,
        resource_id: str | None = None,
        space_id: str | None = None,
    ) -> bool:
        """Check whether *user_id* may perform *action*.

        Resolution order:
        1. Explicit resource-level grant for (user_id, resource_id, action)
        2. Explicit space-level grant for (user_id, space_id, action)
        3. Role-based permission from the user's assigned role
        4. Deny

        Args:
            user_id: The user requesting access.
            action: The permission being requested.
            resource_id: Optional specific resource ID.
            space_id: Optional space ID for space-scoped checks.

        Returns:
            True if permitted, False otherwise.
        """
        # 1. Resource-level grant
        if resource_id is not None:
            key = (user_id, resource_id, action)
            if key in self._resource_grants:
                return True

        # 2. Space-level grant
        if space_id is not None:
            key = (user_id, space_id, action)
            if key in self._space_grants:
                return True

        # 3. Role-based
        role = self._user_roles.get(user_id)
        if role is not None and role.has_permission(action):
            return True

        return False

    def require(
        self,
        user_id: str,
        action: Permission,
        resource_id: str | None = None,
        space_id: str | None = None,
    ) -> None:
        """Like ``check`` but raises AccessDeniedError on failure.

        Raises:
            AccessDeniedError: If the user lacks the required permission.
        """
        if not self.check(user_id, action, resource_id=resource_id, space_id=space_id):
            scope = ""
            if resource_id:
                scope = f" on resource '{resource_id}'"
            elif space_id:
                scope = f" in space '{space_id}'"
            raise AccessDeniedError(
                f"User '{user_id}' lacks permission '{action.value}'{scope}"
            )

    # ------------------------------------------------------------------
    # Grant / Revoke
    # ------------------------------------------------------------------

    def grant(
        self,
        user_id: str,
        permission: Permission,
        resource_id: str | None = None,
        space_id: str | None = None,
        granted_by: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ResourcePermission | SpacePermission:
        """Grant a permission to a user.

        Args:
            user_id: The user receiving the grant.
            permission: The permission to grant.
            resource_id: If given, grant is scoped to this resource.
            space_id: If given (and resource_id not given), grant is scoped to this space.
            granted_by: User ID who is granting the permission.
            metadata: Optional metadata dict.

        Returns:
            The created ResourcePermission or SpacePermission.

        Raises:
            ValueError: If both resource_id and space_id are provided, or neither is provided
                        when granting a non-role permission.
        """
        meta = metadata or {}

        if resource_id is not None and space_id is not None:
            raise ValueError("Cannot specify both resource_id and space_id")

        if resource_id is not None:
            grant = ResourcePermission(
                user_id=user_id,
                resource_id=resource_id,
                permission=permission,
                granted_by=granted_by,
                metadata=meta,
            )
            self._resource_grants[(user_id, resource_id, permission)] = grant
            return grant

        if space_id is not None:
            grant = SpacePermission(
                user_id=user_id,
                space_id=space_id,
                permission=permission,
                granted_by=granted_by,
                metadata=meta,
            )
            self._space_grants[(user_id, space_id, permission)] = grant
            return grant

        # No resource/space scope — this is a role-level grant.
        # We treat it as adding the permission to the user's current role,
        # or creating a custom role if none exists.
        role = self._user_roles.get(user_id)
        if role is None:
            role = Role(name=f"custom-{user_id}", permissions=frozenset())
        new_role = role.with_permissions(permission)
        self._user_roles[user_id] = new_role
        # Return a synthetic SpacePermission with empty space_id
        return SpacePermission(
            user_id=user_id,
            space_id="",
            permission=permission,
            granted_by=granted_by,
            metadata=meta,
        )

    def revoke(
        self,
        user_id: str,
        permission: Permission,
        resource_id: str | None = None,
        space_id: str | None = None,
    ) -> bool:
        """Revoke a permission from a user.

        Args:
            user_id: The user losing the grant.
            permission: The permission to revoke.
            resource_id: If given, revoke resource-scoped grant.
            space_id: If given, revoke space-scoped grant.

        Returns:
            True if a grant was removed, False if nothing was found.
        """
        if resource_id is not None:
            key = (user_id, resource_id, permission)
            return self._resource_grants.pop(key, None) is not None

        if space_id is not None:
            key = (user_id, space_id, permission)
            return self._space_grants.pop(key, None) is not None

        # Role-level: remove permission from user's role
        role = self._user_roles.get(user_id)
        if role is not None and role.has_permission(permission):
            new_role = role.without_permissions(permission)
            self._user_roles[user_id] = new_role
            return True

        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_resource_grants(
        self, user_id: str | None = None, resource_id: str | None = None
    ) -> list[ResourcePermission]:
        """List resource-level grants, optionally filtered."""
        results = list(self._resource_grants.values())
        if user_id is not None:
            results = [g for g in results if g.user_id == user_id]
        if resource_id is not None:
            results = [g for g in results if g.resource_id == resource_id]
        return results

    def list_space_grants(
        self, user_id: str | None = None, space_id: str | None = None
    ) -> list[SpacePermission]:
        """List space-level grants, optionally filtered."""
        results = list(self._space_grants.values())
        if user_id is not None:
            results = [g for g in results if g.user_id == user_id]
        if space_id is not None:
            results = [g for g in results if g.space_id == space_id]
        return results

    def list_user_permissions(
        self, user_id: str, resource_id: str | None = None, space_id: str | None = None
    ) -> set[Permission]:
        """Return all permissions a user has, including role-based and explicit grants.

        Args:
            user_id: The user to query.
            resource_id: If given, include resource-level grants for this resource.
            space_id: If given, include space-level grants for this space.

        Returns:
            A set of all applicable permissions.
        """
        perms: set[Permission] = set()

        role = self._user_roles.get(user_id)
        if role is not None:
            perms.update(role.permissions)

        if resource_id is not None:
            for key, grant in self._resource_grants.items():
                if key[0] == user_id and key[1] == resource_id:
                    perms.add(grant.permission)

        if space_id is not None:
            for key, grant in self._space_grants.items():
                if key[0] == user_id and key[1] == space_id:
                    perms.add(grant.permission)

        return perms
