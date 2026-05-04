"""VersionStore — save/load version history for memories."""
from __future__ import annotations

import logging
from typing import Any

from neuralmem.versioning.version import MemoryVersion

_logger = logging.getLogger(__name__)


class VersionStore:
    """In-memory version store for tracking memory history.

    Uses a duck-typed storage backend (anything with ``save_history``
    and ``get_history`` methods) when available, otherwise falls back
    to an internal dictionary.

    Usage::

        store = VersionStore()  # purely in-memory
        store.save_version(MemoryVersion(...))
        history = store.get_history("mem-123")
        version = store.get_version("mem-123", 2)
    """

    def __init__(self, storage: Any | None = None) -> None:
        """Initialise with optional backend storage.

        Args:
            storage: Optional backend with ``save_history`` / ``get_history``
                methods. If None, uses internal in-memory dict.
        """
        self._backend = storage
        self._versions: dict[str, list[MemoryVersion]] = {}

    # ── Core API ────────────────────────────────────────────────────────

    def save_version(self, version: MemoryVersion) -> None:
        """Save a version entry.

        If a backend is configured, also delegates to ``storage.save_history``
        for persistence. Updates the internal cache regardless.
        """
        memory_id = version.memory_id
        versions = self._versions.setdefault(memory_id, [])

        # Mark previous latest as not-latest
        if versions:
            # Replace the last version's is_latest flag
            prev = versions[-1]
            if prev.is_latest:
                versions[-1] = prev.__class__(
                    version_number=prev.version_number,
                    memory_id=prev.memory_id,
                    content=prev.content,
                    parent=prev.parent,
                    is_latest=False,
                    changes=prev.changes,
                    created_at=prev.created_at,
                    event=prev.event,
                    metadata=prev.metadata,
                )

        versions.append(version)

        # Delegate to backend if available
        if self._backend is not None:
            try:
                self._backend.save_history(
                    memory_id,
                    old_content=version.parent and versions[version.parent - 1].content if version.parent else None,
                    new_content=version.content,
                    event=version.event,
                    metadata=version.metadata,
                )
            except Exception:
                _logger.warning("Backend save_history failed for %s", memory_id)

        _logger.debug(
            "Saved version %d for memory %s (event=%s)",
            version.version_number,
            memory_id,
            version.event,
        )

    def get_history(self, memory_id: str) -> list[MemoryVersion]:
        """Return all versions for *memory_id*, oldest first."""
        return list(self._versions.get(memory_id, []))

    def get_version(
        self, memory_id: str, version_number: int
    ) -> MemoryVersion | None:
        """Return a specific version by number, or ``None`` if not found."""
        versions = self._versions.get(memory_id, [])
        for v in versions:
            if v.version_number == version_number:
                return v
        return None

    def get_latest(self, memory_id: str) -> MemoryVersion | None:
        """Return the latest version, or ``None`` if no history."""
        versions = self._versions.get(memory_id, [])
        return versions[-1] if versions else None

    def get_version_count(self, memory_id: str) -> int:
        """Return the number of recorded versions."""
        return len(self._versions.get(memory_id, []))

    def clear(self, memory_id: str) -> None:
        """Remove all versions for *memory_id*."""
        self._versions.pop(memory_id, None)
