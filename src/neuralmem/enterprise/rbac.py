"""RBAC manager — role-based access control for memories.

No external database dependency; all grants and role assignments are held in
memory.  Designed to work with the enterprise tenant layer.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any


class Permission(IntEnum):
    """Permission levels ordered by privilege."""

    NONE = 0
    READ = 1
    WRITE = 2
    DELETE = 3
    ADMIN = 4


class ResourceAction(str, Enum):
    """Actions that can be performed on resources."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    EXPORT = "export"
    PURGE = "purge"
    ADMIN = "admin"


@dataclass(frozen=True)
class Role:
    """A named role with a set of allowed actions on resource types."""

    name: str
    permissions: dict[str, Permission] = field(default_factory=dict)
    description: str = ""

    def can(self, resource: str, action: ResourceAction) -> bool:
        """Check if this role can perform *action* on *resource*."""
        perm = self.permissions.get(resource, Permission.NONE)
        required = _action_to_permission(action)
        return perm >= required


@dataclass(frozen=True)
class Grant:
    """A grant of a role to a user (or service account) within a tenant."""

    tenant_id: str
    user_id: str
    role_name: str
    granted_by: str = ""
    granted_at: str = ""


def _action_to_permission(action: ResourceAction) -> Permission:
    """Map a resource action to the minimum permission required."""
    mapping = {
        ResourceAction.READ: Permission.READ,
        ResourceAction.LIST: Permission.READ,
        ResourceAction.CREATE: Permission.WRITE,
        ResourceAction.UPDATE: Permission.WRITE,
        ResourceAction.EXPORT: Permission.READ,
        ResourceAction.DELETE: Permission.DELETE,
        ResourceAction.PURGE: Permission.DELETE,
        ResourceAction.ADMIN: Permission.ADMIN,
    }
    return mapping.get(action, Permission.ADMIN)


# ------------------------------------------------------------------
# Built-in roles
# ------------------------------------------------------------------

BUILTIN_ROLES: dict[str, Role] = {
    "viewer": Role(
        name="viewer",
        permissions={
            "memory": Permission.READ,
            "tenant": Permission.READ,
            "audit": Permission.READ,
        },
        description="Read-only access to memories and tenant info",
    ),
    "editor": Role(
        name="editor",
        permissions={
            "memory": Permission.WRITE,
            "tenant": Permission.READ,
            "audit": Permission.READ,
        },
        description="Can create and update memories",
    ),
    "admin": Role(
        name="admin",
        permissions={
            "memory": Permission.ADMIN,
            "tenant": Permission.ADMIN,
            "audit": Permission.ADMIN,
            "rbac": Permission.ADMIN,
        },
        description="Full administrative access",
    ),
    "auditor": Role(
        name="auditor",
        permissions={
            "memory": Permission.READ,
            "tenant": Permission.READ,
            "audit": Permission.ADMIN,
            "rbac": Permission.READ,
        },
        description="Read access + full audit log access for compliance",
    ),
    "gdpr_controller": Role(
        name="gdpr_controller",
        permissions={
            "memory": Permission.READ,
            "tenant": Permission.READ,
            "audit": Permission.ADMIN,
        },
        description="Can read data and view audit logs (GDPR data controller)",
    ),
}


class RBACManager:
    """In-memory RBAC manager for tenant-scoped memory access.

    Parameters
    ----------
    roles
        Optional dict of custom roles.  Built-in roles are always available.
    """

    def __init__(
        self,
        roles: dict[str, Role] | None = None,
    ) -> None:
        self._roles: dict[str, Role] = {**BUILTIN_ROLES}
        if roles:
            self._roles.update(roles)
        self._grants: dict[str, list[Grant]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Role management
    # ------------------------------------------------------------------

    def add_role(self, role: Role) -> None:
        """Register a custom role."""
        with self._lock:
            self._roles[role.name] = role

    def get_role(self, name: str) -> Role | None:
        """Return a role by name."""
        with self._lock:
            return self._roles.get(name)

    def list_roles(self) -> list[str]:
        """Return all registered role names."""
        with self._lock:
            return list(self._roles.keys())

    def remove_role(self, name: str) -> bool:
        """Remove a custom role.  Built-in roles cannot be removed."""
        with self._lock:
            if name in BUILTIN_ROLES:
                return False
            return self._roles.pop(name, None) is not None

    # ------------------------------------------------------------------
    # Grant management
    # ------------------------------------------------------------------

    def grant(
        self,
        tenant_id: str,
        user_id: str,
        role_name: str,
        granted_by: str = "system",
    ) -> Grant:
        """Assign a role to a user within a tenant.

        Raises ``ValueError`` if the role does not exist.
        """
        with self._lock:
            if role_name not in self._roles:
                raise ValueError(f"Role '{role_name}' does not exist")
            grant = Grant(
                tenant_id=tenant_id,
                user_id=user_id,
                role_name=role_name,
                granted_by=granted_by,
                granted_at=__import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
            )
            key = f"{tenant_id}:{user_id}"
            self._grants.setdefault(key, []).append(grant)
            return grant

    def revoke(
        self,
        tenant_id: str,
        user_id: str,
        role_name: str | None = None,
    ) -> bool:
        """Revoke role(s) from a user within a tenant.

        If *role_name* is ``None``, all grants for the user are removed.
        """
        with self._lock:
            key = f"{tenant_id}:{user_id}"
            grants = self._grants.get(key, [])
            if not grants:
                return False
            if role_name is None:
                del self._grants[key]
                return True
            original_len = len(grants)
            self._grants[key] = [g for g in grants if g.role_name != role_name]
            return len(self._grants[key]) < original_len

    def list_grants(
        self,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> list[Grant]:
        """Return grants filtered by tenant and/or user."""
        with self._lock:
            results: list[Grant] = []
            for key, grants in self._grants.items():
                t_id, u_id = key.split(":", 1)
                if tenant_id is not None and t_id != tenant_id:
                    continue
                if user_id is not None and u_id != user_id:
                    continue
                results.extend(grants)
            return results

    # ------------------------------------------------------------------
    # Permission checking
    # ------------------------------------------------------------------

    def check(
        self,
        tenant_id: str,
        user_id: str,
        resource: str,
        action: ResourceAction,
    ) -> bool:
        """Return ``True`` if the user is allowed to perform *action* on *resource*."""
        with self._lock:
            key = f"{tenant_id}:{user_id}"
            grants = self._grants.get(key, [])
            if not grants:
                return False
            for grant in grants:
                role = self._roles.get(grant.role_name)
                if role is None:
                    continue
                if role.can(resource, action):
                    return True
            return False

    def assert_allowed(
        self,
        tenant_id: str,
        user_id: str,
        resource: str,
        action: ResourceAction,
    ) -> None:
        """Raise ``PermissionError`` if the action is not allowed."""
        if not self.check(tenant_id, user_id, resource, action):
            raise PermissionError(
                f"User '{user_id}' is not allowed to '{action.value}' on '{resource}'"
                f" in tenant '{tenant_id}'"
            )

    def get_effective_permissions(
        self,
        tenant_id: str,
        user_id: str,
    ) -> dict[str, Permission]:
        """Return the highest permission level per resource for a user."""
        with self._lock:
            key = f"{tenant_id}:{user_id}"
            grants = self._grants.get(key, [])
            effective: dict[str, Permission] = {}
            for grant in grants:
                role = self._roles.get(grant.role_name)
                if role is None:
                    continue
                for res, perm in role.permissions.items():
                    effective[res] = max(effective.get(res, Permission.NONE), perm)
            return effective

    # ------------------------------------------------------------------
    # Tenant-level helpers
    # ------------------------------------------------------------------

    def get_tenant_users(self, tenant_id: str) -> list[str]:
        """Return all user IDs that have at least one grant in the tenant."""
        with self._lock:
            users: set[str] = set()
            for key in self._grants:
                t_id, u_id = key.split(":", 1)
                if t_id == tenant_id:
                    users.add(u_id)
            return sorted(users)

    def get_user_roles(
        self,
        tenant_id: str,
        user_id: str,
    ) -> list[str]:
        """Return the role names assigned to a user in a tenant."""
        with self._lock:
            key = f"{tenant_id}:{user_id}"
            return [g.role_name for g in self._grants.get(key, [])]

    def has_any_grant(self, tenant_id: str, user_id: str) -> bool:
        """Return ``True`` if the user has any grant in the tenant."""
        with self._lock:
            key = f"{tenant_id}:{user_id}"
            return key in self._grants and len(self._grants[key]) > 0

    # ------------------------------------------------------------------
    # Memory-specific helpers
    # ------------------------------------------------------------------

    def can_read_memory(self, tenant_id: str, user_id: str) -> bool:
        return self.check(tenant_id, user_id, "memory", ResourceAction.READ)

    def can_write_memory(self, tenant_id: str, user_id: str) -> bool:
        return self.check(tenant_id, user_id, "memory", ResourceAction.CREATE)

    def can_delete_memory(self, tenant_id: str, user_id: str) -> bool:
        return self.check(tenant_id, user_id, "memory", ResourceAction.DELETE)

    def can_purge_memory(self, tenant_id: str, user_id: str) -> bool:
        return self.check(tenant_id, user_id, "memory", ResourceAction.PURGE)

    def can_admin_tenant(self, tenant_id: str, user_id: str) -> bool:
        return self.check(tenant_id, user_id, "tenant", ResourceAction.ADMIN)

    def can_read_audit(self, tenant_id: str, user_id: str) -> bool:
        return self.check(tenant_id, user_id, "audit", ResourceAction.READ)

    def can_admin_rbac(self, tenant_id: str, user_id: str) -> bool:
        return self.check(tenant_id, user_id, "rbac", ResourceAction.ADMIN)
