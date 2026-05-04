"""SpaceMembership — role-based access for space members."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from neuralmem.core.types import _generate_ulid


class SpaceRole(str, Enum):
    """Role hierarchy for space membership.

    Permissions (highest to lowest):
    - owner:  full control, can delete space, transfer ownership
    - admin:  manage members, settings, moderate content
    - editor: add/edit memories, invite viewers
    - viewer: read-only access to memories
    """
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"

    @classmethod
    def can_manage_members(cls, role: "SpaceRole") -> bool:
        """Check if role can add/remove/manage members."""
        return role in (cls.OWNER, cls.ADMIN)

    @classmethod
    def can_edit(cls, role: "SpaceRole") -> bool:
        """Check if role can create/edit memories."""
        return role in (cls.OWNER, cls.ADMIN, cls.EDITOR)

    @classmethod
    def can_delete_space(cls, role: "SpaceRole") -> bool:
        """Check if role can delete the space."""
        return role == cls.OWNER

    @classmethod
    def can_change_settings(cls, role: "SpaceRole") -> bool:
        """Check if role can modify space settings."""
        return role in (cls.OWNER, cls.ADMIN)


class SpaceMembership(BaseModel):
    """A user's membership in a space."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=_generate_ulid)
    space_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    role: SpaceRole = SpaceRole.VIEWER
    invited_by: str | None = Field(default=None, description="User ID who invited this member")
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Timestamps
    joined_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def with_role(self, new_role: SpaceRole) -> "SpaceMembership":
        """Return a new membership with an updated role (immutable update)."""
        data = self.model_dump()
        data["role"] = new_role.value
        data.pop("id", None)
        data.pop("joined_at", None)
        return SpaceMembership(**data)
