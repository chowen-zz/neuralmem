"""NeuralMem community features — memory sharing between users/agents.

V1.3: MemorySharing
  • Share a memory with another user/agent (grant/revoke)
  • Query memories shared *to* or *from* a principal
  • In-memory implementation (no external DB required)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neuralmem.core.types import Memory


@dataclass
class ShareRecord:
    """A single share grant."""
    share_id: str
    memory_id: str
    owner_id: str          # who owns the memory
    recipient_id: str      # who can access it
    recipient_type: str    # "user" | "agent" | "team"
    permission: str        # "read" | "write"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MemorySharing:
    """In-memory memory sharing manager.

    Usage:
        sharing = MemorySharing()
        sharing.share_memory(mem, owner_id="u1", recipient_id="u2")
        shared = sharing.list_shared_to("u2")
    """

    def __init__(self) -> None:
        # memory_id -> list of ShareRecord
        self._shares: dict[str, list[ShareRecord]] = {}
        # memory store (populated externally or via register_memory)
        self._memories: dict[str, Memory] = {}
        self._counter: int = 0

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _next_id(self) -> str:
        self._counter += 1
        return f"shr-{self._counter:06d}"

    def _is_expired(self, rec: ShareRecord) -> bool:
        if rec.expires_at is None:
            return False
        return datetime.now(timezone.utc) > rec.expires_at

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def register_memory(self, memory: Memory) -> None:
        """Index a memory so it can be shared."""
        self._memories[memory.id] = memory

    def share_memory(
        self,
        memory: Memory,
        owner_id: str,
        recipient_id: str,
        *,
        recipient_type: str = "user",
        permission: str = "read",
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ShareRecord:
        """Grant access to *memory* to *recipient_id*.

        Args:
            memory: the Memory object to share.
            owner_id: principal that owns the memory.
            recipient_id: principal that receives access.
            recipient_type: "user", "agent", or "team".
            permission: "read" or "write".
            expires_at: optional expiration time.
            metadata: arbitrary extra data.

        Returns:
            The created ShareRecord.
        """
        if memory.id not in self._memories:
            self.register_memory(memory)

        record = ShareRecord(
            share_id=self._next_id(),
            memory_id=memory.id,
            owner_id=owner_id,
            recipient_id=recipient_id,
            recipient_type=recipient_type,
            permission=permission,
            expires_at=expires_at,
            metadata=metadata or {},
        )
        self._shares.setdefault(memory.id, []).append(record)
        return record

    def revoke_share(self, memory_id: str, recipient_id: str) -> bool:
        """Revoke all shares of *memory_id* to *recipient_id*.

        Returns True if any share was removed.
        """
        removed = False
        for recs in self._shares.values():
            to_remove = [r for r in recs if r.memory_id == memory_id and r.recipient_id == recipient_id]
            for r in to_remove:
                recs.remove(r)
                removed = True
        return removed

    def revoke_all_for_memory(self, memory_id: str) -> int:
        """Revoke every share for a given memory. Returns number removed."""
        if memory_id not in self._shares:
            return 0
        count = len(self._shares[memory_id])
        del self._shares[memory_id]
        return count

    def list_shared_to(
        self,
        recipient_id: str,
        *,
        recipient_type: str | None = None,
        include_expired: bool = False,
    ) -> list[tuple[Memory, ShareRecord]]:
        """Return all (Memory, ShareRecord) pairs shared **to** *recipient_id*.

        Args:
            recipient_id: the receiving principal.
            recipient_type: optionally filter by type.
            include_expired: if False, skip expired records.
        """
        results: list[tuple[Memory, ShareRecord]] = []
        for recs in self._shares.values():
            for rec in recs:
                if rec.recipient_id != recipient_id:
                    continue
                if recipient_type is not None and rec.recipient_type != recipient_type:
                    continue
                if not include_expired and self._is_expired(rec):
                    continue
                mem = self._memories.get(rec.memory_id)
                if mem is not None:
                    results.append((mem, rec))
        return results

    def list_shared_from(
        self,
        owner_id: str,
        *,
        include_expired: bool = False,
    ) -> list[tuple[Memory, ShareRecord]]:
        """Return all (Memory, ShareRecord) pairs shared **from** *owner_id*."""
        results: list[tuple[Memory, ShareRecord]] = []
        for recs in self._shares.values():
            for rec in recs:
                if rec.owner_id != owner_id:
                    continue
                if not include_expired and self._is_expired(rec):
                    continue
                mem = self._memories.get(rec.memory_id)
                if mem is not None:
                    results.append((mem, rec))
        return results

    def can_access(
        self,
        memory_id: str,
        principal_id: str,
        *,
        required_permission: str = "read",
    ) -> bool:
        """Check whether *principal_id* has access to *memory_id*.

        Write permission implies read permission.
        """
        recs = self._shares.get(memory_id, [])
        for rec in recs:
            if rec.recipient_id != principal_id:
                continue
            if self._is_expired(rec):
                continue
            if required_permission == "read":
                return True
            if required_permission == "write" and rec.permission == "write":
                return True
        return False

    def get_share_stats(self, memory_id: str) -> dict[str, Any]:
        """Return share statistics for a memory."""
        recs = self._shares.get(memory_id, [])
        active = [r for r in recs if not self._is_expired(r)]
        return {
            "memory_id": memory_id,
            "total_shares": len(recs),
            "active_shares": len(active),
            "expired_shares": len(recs) - len(active),
            "read_grants": len([r for r in active if r.permission == "read"]),
            "write_grants": len([r for r in active if r.permission == "write"]),
        }
