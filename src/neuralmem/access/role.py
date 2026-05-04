"""Role — predefined and custom roles with permission sets."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from neuralmem.access.permission import Permission


class PredefinedRole(str, Enum):
    """Built-in role identifiers for NeuralMem access control."""

    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"
    OWNER = "owner"
    SYSTEM = "system"


@dataclass(frozen=True)
class Role:
    """A role is a named collection of permissions.

    Roles are immutable. Use ``with_permissions`` or ``without_permissions``
    to derive new roles.
    """

    name: str
    permissions: frozenset[Permission] = field(default_factory=frozenset)
    description: str = ""

    def has_permission(self, permission: Permission) -> bool:
        """Check if this role grants a specific permission."""
        return permission in self.permissions

    def has_any(self, *permissions: Permission) -> bool:
        """Check if this role grants any of the given permissions."""
        return any(p in self.permissions for p in permissions)

    def has_all(self, *permissions: Permission) -> bool:
        """Check if this role grants all of the given permissions."""
        return all(p in self.permissions for p in permissions)

    def with_permissions(self, *permissions: Permission) -> "Role":
        """Return a new role with additional permissions."""
        new_perms = self.permissions | set(permissions)
        return Role(
            name=self.name,
            permissions=frozenset(new_perms),
            description=self.description,
        )

    def without_permissions(self, *permissions: Permission) -> "Role":
        """Return a new role with the given permissions removed."""
        new_perms = self.permissions - set(permissions)
        return Role(
            name=self.name,
            permissions=frozenset(new_perms),
            description=self.description,
        )

    def __repr__(self) -> str:
        return f"Role({self.name!r}, permissions={sorted(self.permissions)!r})"


# ---------------------------------------------------------------------------
# Predefined role factories
# ---------------------------------------------------------------------------

def make_viewer() -> Role:
    """Read-only access to memories and spaces."""
    return Role(
        name=PredefinedRole.VIEWER.value,
        permissions=frozenset({
            Permission.READ,
            Permission.SEARCH,
            Permission.SPACE_READ,
        }),
        description="Can read and search memories, view spaces",
    )


def make_editor() -> Role:
    """Can read, write, and share memories; read spaces."""
    return Role(
        name=PredefinedRole.EDITOR.value,
        permissions=frozenset({
            Permission.READ,
            Permission.WRITE,
            Permission.DELETE,
            Permission.SEARCH,
            Permission.SHARE,
            Permission.SPACE_READ,
            Permission.SPACE_WRITE,
        }),
        description="Can create, edit, delete, and share memories",
    )


def make_admin() -> Role:
    """Full space management plus memory operations."""
    return Role(
        name=PredefinedRole.ADMIN.value,
        permissions=frozenset({
            Permission.READ,
            Permission.WRITE,
            Permission.DELETE,
            Permission.SEARCH,
            Permission.SHARE,
            Permission.SPACE_READ,
            Permission.SPACE_WRITE,
            Permission.SPACE_DELETE,
            Permission.SPACE_MANAGE_MEMBERS,
            Permission.SPACE_CHANGE_SETTINGS,
        }),
        description="Can manage spaces, members, and all memory operations",
    )


def make_owner() -> Role:
    """Full control including admin-level audit."""
    return Role(
        name=PredefinedRole.OWNER.value,
        permissions=frozenset({
            Permission.READ,
            Permission.WRITE,
            Permission.DELETE,
            Permission.SEARCH,
            Permission.SHARE,
            Permission.SPACE_READ,
            Permission.SPACE_WRITE,
            Permission.SPACE_DELETE,
            Permission.SPACE_MANAGE_MEMBERS,
            Permission.SPACE_CHANGE_SETTINGS,
            Permission.ADMIN,
            Permission.ADMIN_AUDIT,
            Permission.ADMIN_IMPERSONATE,
        }),
        description="Full control over spaces, members, memories, and audit",
    )


def make_system() -> Role:
    """System-level role with all permissions."""
    return Role(
        name=PredefinedRole.SYSTEM.value,
        permissions=Permission.all(),
        description="System service account with unrestricted access",
    )


# Convenience lookup
PREDEFINED_ROLES: dict[str, Role] = {
    PredefinedRole.VIEWER.value: make_viewer(),
    PredefinedRole.EDITOR.value: make_editor(),
    PredefinedRole.ADMIN.value: make_admin(),
    PredefinedRole.OWNER.value: make_owner(),
    PredefinedRole.SYSTEM.value: make_system(),
}


def get_role(name: str) -> Role | None:
    """Look up a predefined role by name."""
    return PREDEFINED_ROLES.get(name)
