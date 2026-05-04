"""Space dataclass — project-level memory container with metadata."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from neuralmem.core.types import _generate_ulid


class SpaceVisibility(str, Enum):
    """Space visibility levels."""
    PRIVATE = "private"      # Only members can access
    INTERNAL = "internal"    # Organization members can discover
    PUBLIC = "public"        # Anyone can view (read-only for non-members)


class Space(BaseModel):
    """A project-level memory container.

    Spaces group related memories under a shared project context with
    configurable access control and settings.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=_generate_ulid)
    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="", max_length=2048)
    owner: str = Field(..., min_length=1, description="User ID of the space owner")
    visibility: SpaceVisibility = SpaceVisibility.PRIVATE
    settings: dict[str, Any] = Field(default_factory=dict)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Memory association (not a foreign key — just a logical grouping)
    memory_ids: tuple[str, ...] = Field(default_factory=tuple)

    def with_fields(self, **kwargs) -> "Space":
        """Return a new Space with updated fields (immutable update)."""
        data = self.model_dump()
        data.update(kwargs)
        data.pop("id", None)
        data.pop("created_at", None)
        return Space(**data)
