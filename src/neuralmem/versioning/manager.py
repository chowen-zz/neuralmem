"""VersionManager — create new version, rollback, diff, list versions."""
from __future__ import annotations

import difflib
import logging
from typing import Any

from neuralmem.versioning.version import MemoryVersion
from neuralmem.versioning.store import VersionStore

_logger = logging.getLogger(__name__)


class VersionManager:
    """High-level version management for memories.

    Wraps a VersionStore to provide create_version, rollback, diff,
    and list_versions operations.

    Usage::

        manager = VersionManager(VersionStore())
        v1 = manager.create_version("mem-123", "initial content", event="CREATE")
        v2 = manager.create_version("mem-123", "updated content", parent=v1.version_number)
        diff = manager.diff("mem-123", v1.version_number, v2.version_number)
        restored = manager.rollback("mem-123", v1.version_number)
    """

    def __init__(self, store: VersionStore) -> None:
        self._store = store
        self._counters: dict[str, int] = {}

    # ── Version lifecycle ─────────────────────────────────────────────

    def create_version(
        self,
        memory_id: str,
        content: str,
        *,
        parent: int | None = None,
        event: str = "UPDATE",
        changes: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryVersion:
        """Create a new version for *memory_id*.

        Args:
            memory_id: The memory identifier.
            content: The content at this version.
            parent: Parent version number (auto-detected if None).
            event: Event type (CREATE, UPDATE, DELETE, etc.).
            changes: Dict describing what changed.
            metadata: Additional metadata.

        Returns:
            The created MemoryVersion.
        """
        if not memory_id:
            raise ValueError("memory_id cannot be empty")

        # Auto-detect parent from latest version
        latest = self._store.get_latest(memory_id)
        if parent is None and latest is not None:
            parent = latest.version_number

        version_number = self._next_version_number(memory_id)

        version = MemoryVersion(
            version_number=version_number,
            memory_id=memory_id,
            content=content,
            parent=parent,
            is_latest=True,
            changes=changes or {},
            event=event,
            metadata=metadata or {},
        )

        self._store.save_version(version)
        _logger.info(
            "Created version %d for memory %s (event=%s)",
            version_number,
            memory_id,
            event,
        )
        return version

    def rollback(
        self,
        memory_id: str,
        version_number: int,
    ) -> MemoryVersion:
        """Rollback *memory_id* to *version_number*.

        Creates a new version entry with the restored content and
        event="ROLLBACK".

        Args:
            memory_id: The memory to restore.
            version_number: Target version number.

        Returns:
            The new rollback version.

        Raises:
            ValueError: If no history or version not found.
        """
        history = self._store.get_history(memory_id)
        if not history:
            raise ValueError(
                f"No version history found for memory '{memory_id}'"
            )

        target = None
        for v in history:
            if v.version_number == version_number:
                target = v
                break

        if target is None:
            raise ValueError(
                f"Version {version_number} not found for memory '{memory_id}'"
            )

        latest = self._store.get_latest(memory_id)
        latest_content = latest.content if latest else None

        rollback_version = self.create_version(
            memory_id,
            content=target.content,
            parent=latest.version_number if latest else None,
            event="ROLLBACK",
            changes={
                "rollback": f"v{latest.version_number if latest else '?'} → v{version_number}"
            },
            metadata={
                "rolled_back_to": version_number,
                "rolled_back_to_created_at": target.created_at.isoformat(),
            },
        )

        _logger.info(
            "Rolled back memory %s to version %d",
            memory_id,
            version_number,
        )
        return rollback_version

    # ── Diff / comparison ───────────────────────────────────────────

    def diff(
        self,
        memory_id: str,
        version_a: int,
        version_b: int,
    ) -> str:
        """Return a unified diff between two versions.

        Args:
            memory_id: The memory identifier.
            version_a: First version number.
            version_b: Second version number.

        Returns:
            Unified diff string.

        Raises:
            ValueError: If either version is not found.
        """
        va = self._store.get_version(memory_id, version_a)
        vb = self._store.get_version(memory_id, version_b)

        if va is None:
            raise ValueError(
                f"Version {version_a} not found for memory '{memory_id}'"
            )
        if vb is None:
            raise ValueError(
                f"Version {version_b} not found for memory '{memory_id}'"
            )

        lines_a = va.content.splitlines(keepends=True)
        lines_b = vb.content.splitlines(keepends=True)

        # Ensure lines end with newline for clean diff
        if lines_a and not lines_a[-1].endswith("\n"):
            lines_a[-1] += "\n"
        if lines_b and not lines_b[-1].endswith("\n"):
            lines_b[-1] += "\n"

        diff = difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=f"v{version_a}",
            tofile=f"v{version_b}",
            lineterm="",
        )
        return "".join(diff)

    def list_versions(self, memory_id: str) -> list[dict[str, Any]]:
        """Return all versions as plain dicts for *memory_id*."""
        return [v.to_dict() for v in self._store.get_history(memory_id)]

    def get_latest(self, memory_id: str) -> MemoryVersion | None:
        """Return the latest version, or ``None``."""
        return self._store.get_latest(memory_id)

    def get_version_count(self, memory_id: str) -> int:
        """Return the number of versions."""
        return self._store.get_version_count(memory_id)

    # ── Internal helpers ──────────────────────────────────────────────

    def _next_version_number(self, memory_id: str) -> int:
        """Return the next sequential version number for *memory_id*."""
        count = self._store.get_version_count(memory_id)
        return count + 1
