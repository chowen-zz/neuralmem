"""NeuralMem organization package exports (V2.0).

Multi-user organization support with role-based membership and settings.
"""
from __future__ import annotations

from neuralmem.organization.org import Organization, OrgSettings, OrgStatus
from neuralmem.organization.member import (
    OrganizationMember,
    MemberRole,
    MembershipStatus,
)
from neuralmem.organization.manager import (
    OrgManager,
    OrgNotFoundError,
    MemberNotFoundError,
    DuplicateMemberError,
    RoleAssignmentError,
)

__all__ = [
    "Organization",
    "OrgSettings",
    "OrgStatus",
    "OrganizationMember",
    "MemberRole",
    "MembershipStatus",
    "OrgManager",
    "OrgNotFoundError",
    "MemberNotFoundError",
    "DuplicateMemberError",
    "RoleAssignmentError",
]
