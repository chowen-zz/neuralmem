"""Permission — fine-grained action permissions for NeuralMem resources."""
from __future__ import annotations

from enum import Enum


class Permission(str, Enum):
    """Fine-grained permissions for memory operations.

    Permissions are grouped by scope:
    - Memory-level: read, write, delete on individual memories
    - Space-level: manage spaces and their members
    - Admin-level: system administration
    """

    # Memory-level permissions
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SEARCH = "search"
    SHARE = "share"

    # Space-level permissions
    SPACE_READ = "space:read"
    SPACE_WRITE = "space:write"
    SPACE_DELETE = "space:delete"
    SPACE_MANAGE_MEMBERS = "space:manage_members"
    SPACE_CHANGE_SETTINGS = "space:change_settings"

    # Admin-level permissions
    ADMIN = "admin"
    ADMIN_AUDIT = "admin:audit"
    ADMIN_IMPERSONATE = "admin:impersonate"

    @classmethod
    def all(cls) -> set[Permission]:
        """Return the full set of all permissions."""
        return set(cls)

    @classmethod
    def memory_permissions(cls) -> set[Permission]:
        """Return permissions that operate on individual memories."""
        return {
            cls.READ,
            cls.WRITE,
            cls.DELETE,
            cls.SEARCH,
            cls.SHARE,
        }

    @classmethod
    def space_permissions(cls) -> set[Permission]:
        """Return permissions that operate on spaces."""
        return {
            cls.SPACE_READ,
            cls.SPACE_WRITE,
            cls.SPACE_DELETE,
            cls.SPACE_MANAGE_MEMBERS,
            cls.SPACE_CHANGE_SETTINGS,
        }

    @classmethod
    def admin_permissions(cls) -> set[Permission]:
        """Return administrative permissions."""
        return {
            cls.ADMIN,
            cls.ADMIN_AUDIT,
            cls.ADMIN_IMPERSONATE,
        }
