"""Organization dataclass — multi-user organization with settings.

V2.0: Organization
  • Multi-user organization with configurable settings
  • Support for domains, quotas, and feature flags
  • In-memory + optional SQLite persistence
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _generate_org_id() -> str:
    """Generate a unique organization ID."""
    return f"org_{secrets.token_hex(8)}"


class OrgStatus(str, Enum):
    """Organization lifecycle status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"
    ARCHIVED = "archived"


@dataclass
class OrgSettings:
    """Configurable settings for an organization."""

    # Memory quotas
    max_members: int = field(default=100)
    max_memories_per_user: int = field(default=1000)
    max_storage_mb: int = field(default=1024)

    # Feature flags
    enable_sharing: bool = field(default=True)
    enable_analytics: bool = field(default=True)
    enable_audit_log: bool = field(default=True)
    enable_sso: bool = field(default=False)
    enable_api_keys: bool = field(default=True)

    # Collaboration
    allow_public_spaces: bool = field(default=False)
    allow_guest_access: bool = field(default=False)
    require_approval_for_sharing: bool = field(default=False)

    # Custom fields
    custom_domains: list[str] = field(default_factory=list)
    branding: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize settings to a dictionary."""
        return {
            "max_members": self.max_members,
            "max_memories_per_user": self.max_memories_per_user,
            "max_storage_mb": self.max_storage_mb,
            "enable_sharing": self.enable_sharing,
            "enable_analytics": self.enable_analytics,
            "enable_audit_log": self.enable_audit_log,
            "enable_sso": self.enable_sso,
            "enable_api_keys": self.enable_api_keys,
            "allow_public_spaces": self.allow_public_spaces,
            "allow_guest_access": self.allow_guest_access,
            "require_approval_for_sharing": self.require_approval_for_sharing,
            "custom_domains": list(self.custom_domains),
            "branding": dict(self.branding),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrgSettings":
        """Deserialize settings from a dictionary."""
        settings = cls()
        for key, value in data.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        return settings


@dataclass
class Organization:
    """A multi-user organization in NeuralMem.

    Attributes:
        org_id: Unique organization identifier.
        name: Human-readable organization name.
        slug: URL-friendly identifier.
        description: Optional organization description.
        owner_id: User ID of the organization owner.
        status: Current lifecycle status.
        settings: Organization-specific settings.
        created_at: UTC timestamp when the org was created.
        updated_at: UTC timestamp of last modification.
        metadata: Arbitrary extra data.
    """

    org_id: str = field(default_factory=_generate_org_id)
    name: str = ""
    slug: str = ""
    description: str = ""
    owner_id: str = ""
    status: OrgStatus = OrgStatus.ACTIVE
    settings: OrgSettings = field(default_factory=OrgSettings)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure slug is set from name if empty."""
        if not self.slug and self.name:
            self.slug = self._make_slug(self.name)

    @staticmethod
    def _make_slug(name: str) -> str:
        """Convert a name to a URL-friendly slug."""
        return name.lower().strip().replace(" ", "-").replace("_", "-")[:50]

    def update_settings(self, **kwargs: Any) -> None:
        """Update one or more settings fields."""
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        self.updated_at = datetime.now(timezone.utc)

    def update(self, **kwargs: Any) -> None:
        """Update organization fields (name, description, status, metadata)."""
        allowed = {"name", "description", "status", "metadata", "owner_id"}
        for key, value in kwargs.items():
            if key in allowed and hasattr(self, key):
                setattr(self, key, value)
                if key == "name" and value:
                    self.slug = self._make_slug(value)
        self.updated_at = datetime.now(timezone.utc)

    def is_active(self) -> bool:
        """Return True if the organization is active."""
        return self.status == OrgStatus.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        """Serialize organization to a dictionary."""
        return {
            "org_id": self.org_id,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "owner_id": self.owner_id,
            "status": self.status.value,
            "settings": self.settings.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }
