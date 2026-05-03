"""MemoryVersioner — version tracking and rollback for memories."""
from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger(__name__)


class MemoryVersioner:
    """Wraps a storage backend to provide version tracking and rollback.

    Uses the storage's ``save_history`` / ``get_history`` methods to record
    and retrieve version information.

    Usage::

        from neuralmem.storage.sqlite import SQLiteStorage
        from neuralmem.versioning import MemoryVersioner

        storage = SQLiteStorage(config)
        versioner = MemoryVersioner(storage)

        # Record a new version
        versioner.save_version("mem-123", old="old text", new="new text")

        # Get all versions
        versions = versioner.get_versions("mem-123")

        # Rollback to version 0
        versioner.rollback("mem-123", version_index=0)
    """

    def __init__(self, storage: Any) -> None:
        """Initialise with any storage that has ``save_history`` and
        ``get_history`` methods (duck-typed)."""
        self._storage = storage

    def save_version(
        self,
        memory_id: str,
        old_content: str | None,
        new_content: str,
        event: str = "UPDATE",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a version change for *memory_id*.

        Delegates to ``storage.save_history``.
        """
        self._storage.save_history(
            memory_id,
            old_content,
            new_content,
            event=event,
            metadata=metadata,
        )
        _logger.debug(
            "Saved version for memory %s (event=%s)", memory_id, event
        )

    def get_versions(
        self, memory_id: str
    ) -> list[dict[str, object]]:
        """Return all recorded versions for *memory_id*.

        Each entry is a dict with keys: ``id``, ``memory_id``,
        ``old_content``, ``new_content``, ``event``, ``changed_at``,
        ``metadata``.
        """
        history = self._storage.get_history(memory_id)
        return history

    def rollback(
        self,
        memory_id: str,
        version_index: int,
    ) -> str:
        """Restore *memory_id* to the content at *version_index*.

        Args:
            memory_id: The memory to restore.
            version_index: 0-based index into the version history list.

        Returns:
            The content string that was restored.

        Raises:
            IndexError: If *version_index* is out of range.
            ValueError: If no history exists for the memory.
        """
        history = self._storage.get_history(memory_id)

        if not history:
            raise ValueError(
                f"No version history found for memory '{memory_id}'"
            )

        if version_index < 0 or version_index >= len(history):
            raise IndexError(
                f"Version index {version_index} out of range. "
                f"Available: 0..{len(history) - 1}"
            )

        target_entry = history[version_index]
        restored_content = target_entry["new_content"]
        assert isinstance(restored_content, str)

        current_memory = self._storage.get_memory(memory_id)
        current_content = (
            current_memory.content if current_memory else None
        )

        # Update the memory content
        self._storage.update_memory(memory_id, content=restored_content)

        # Record the rollback as a new version entry
        self._storage.save_history(
            memory_id,
            old_content=current_content,  # type: ignore[arg-type]
            new_content=restored_content,
            event="ROLLBACK",
            metadata={
                "rolled_back_to_index": version_index,
                "rolled_back_to_changed_at": str(
                    target_entry.get("changed_at", "")
                ),
            },
        )

        _logger.info(
            "Rolled back memory %s to version %d",
            memory_id,
            version_index,
        )
        return restored_content

    def get_version_count(self, memory_id: str) -> int:
        """Return the number of recorded versions for *memory_id*."""
        return len(self._storage.get_history(memory_id))

    def get_latest(
        self, memory_id: str
    ) -> dict[str, object] | None:
        """Return the most recent version entry, or ``None`` if empty."""
        history = self._storage.get_history(memory_id)
        return history[-1] if history else None
