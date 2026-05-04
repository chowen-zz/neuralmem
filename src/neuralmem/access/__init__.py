"""NeuralMem Access Control — fine-grained RBAC permissions for memory resources."""
from __future__ import annotations

from neuralmem.access.permission import Permission
from neuralmem.access.role import Role, PredefinedRole
from neuralmem.access.control import (
    AccessControl,
    AccessDeniedError,
    ResourcePermission,
    SpacePermission,
)

__all__ = [
    "Permission",
    "Role",
    "PredefinedRole",
    "AccessControl",
    "AccessDeniedError",
    "ResourcePermission",
    "SpacePermission",
]
