"""MemoryVersion — dataclass/model for tracking versions of a memory."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryVersion:
    """Represents a single version of a memory.

    Attributes:
        version_number: Sequential version number (1, 2, 3, ...).
        memory_id: The ID of the memory this version belongs to.
        parent: The parent version number (None for the first version).
        is_latest: Whether this is the latest version.
        changes: Dict describing what changed (e.g., {"content": "old → new"}).
        content: The actual content at this version.
        created_at: When this version was created.
        event: The event type (CREATE, UPDATE, DELETE, ROLLBACK).
        metadata: Additional metadata.
    """

    version_number: int
    memory_id: str
    content: str
    parent: int | None = None
    is_latest: bool = False
    changes: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    event: str = "UPDATE"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version_number < 1:
            raise ValueError("version_number must be >= 1")
        if not self.memory_id:
            raise ValueError("memory_id cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "version_number": self.version_number,
            "memory_id": self.memory_id,
            "content": self.content,
            "parent": self.parent,
            "is_latest": self.is_latest,
            "changes": dict(self.changes),
            "created_at": self.created_at.isoformat(),
            "event": self.event,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryVersion:
        """Deserialize from a plain dictionary."""
        return cls(
            version_number=data["version_number"],
            memory_id=data["memory_id"],
            content=data["content"],
            parent=data.get("parent"),
            is_latest=data.get("is_latest", False),
            changes=data.get("changes", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            event=data.get("event", "UPDATE"),
            metadata=data.get("metadata", {}),
        )
