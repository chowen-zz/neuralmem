"""OrganizationMember dataclass — user membership with roles.

V2.0: OrganizationMember
  • Role-based membership within an organization
  • Support for pending/approved/revoked membership states
  • Metadata and permissions tracking
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemberRole(str, Enum):
    """Roles within an organization, ordered by privilege."""
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    GUEST = "guest"

    @classmethod
    def _privilege_order(cls) -> list[MemberRole]:
        return [cls.GUEST, cls.VIEWER, cls.EDITOR, cls.ADMIN, cls.OWNER]

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, MemberRole):
            return NotImplemented
        order = self._privilege_order()
        return order.index(self) >= order.index(other)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, MemberRole):
            return NotImplemented
        order = self._privilege_order()
        return order.index(self) > order.index(other)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, MemberRole):
            return NotImplemented
        order = self._privilege_order()
        return order.index(self) <= order.index(other)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, MemberRole):
            return NotImplemented
        order = self._privilege_order()
        return order.index(self) < order.index(other)


class MembershipStatus(str, Enum):
    """Membership lifecycle status."""
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


@dataclass
class OrganizationMember:
    """A user membership within an organization.

    Attributes:
        user_id: Unique user identifier.
        org_id: Organization identifier.
        role: Member's role within the organization.
        status: Current membership status.
        invited_by: User ID of the inviter (if applicable).
        joined_at: UTC timestamp when the member joined.
        updated_at: UTC timestamp of last modification.
        metadata: Arbitrary extra data.
    """

    user_id: str
    org_id: str
    role: MemberRole = MemberRole.VIEWER
    status: MembershipStatus = MembershipStatus.ACTIVE
    invited_by: str = ""
    joined_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_role(self, new_role: MemberRole) -> None:
        """Update the member's role."""
        self.role = new_role
        self.updated_at = datetime.now(timezone.utc)

    def update_status(self, new_status: MembershipStatus) -> None:
        """Update the member's status."""
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)

    def is_active(self) -> bool:
        """Return True if the membership is active."""
        return self.status == MembershipStatus.ACTIVE

    def can(self, required_role: MemberRole) -> bool:
        """Check if this member has at least the required role privilege."""
        return self.is_active() and self.role >= required_role

    def to_dict(self) -> dict[str, Any]:
        """Serialize member to a dictionary."""
        return {
            "user_id": self.user_id,
            "org_id": self.org_id,
            "role": self.role.value,
            "status": self.status.value,
            "invited_by": self.invited_by,
            "joined_at": self.joined_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }
