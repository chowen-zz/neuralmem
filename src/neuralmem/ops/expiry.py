"""Memory expiry policies for NeuralMem.

Supports three policy types:
- **TTL**: Delete memories older than *max_age_days*.
- **max_count**: Keep only the *N* most important memories per user.
- **importance_threshold**: Delete memories below a given importance score.

Integrates with ``storage.delete_memories()`` and ``storage.list_memories()``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpiryPolicy:
    """Configuration for a single expiry policy."""
    ttl_days: int | None = None
    max_count: int | None = None
    importance_threshold: float | None = None


@dataclass
class ExpiryResult:
    """Result of applying expiry policies."""
    expired_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class MemoryExpiry:
    """Apply configurable expiry policies to user memories.

    Parameters
    ----------
    storage:
        A storage backend that supports ``list_memories()``,
        ``delete_memories()``, and ``get_memory()``.
    policy:
        Default expiry policy applied to every ``apply_policies`` call.
    """

    def __init__(
        self,
        storage: StorageBackend,
        policy: ExpiryPolicy | None = None,
    ) -> None:
        self.storage = storage
        self.policy = policy or ExpiryPolicy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_policies(
        self,
        user_id: str | None = None,
        policy: ExpiryPolicy | None = None,
    ) -> ExpiryResult:
        """Apply all configured expiry policies and delete expired memories.

        Parameters
        ----------
        user_id:
            Scope to a specific user. ``None`` means all users.
        policy:
            Override the default policy for this invocation.

        Returns
        -------
        ExpiryResult
            Contains the list of deleted memory IDs and a details dict
            describing how many memories were expired by each policy.
        """
        p = policy or self.policy
        result = ExpiryResult()
        details: dict[str, Any] = {}

        # --- TTL policy ---
        ttl_ids = self._apply_ttl(user_id, p.ttl_days)
        details["ttl"] = len(ttl_ids)
        result.expired_ids.extend(ttl_ids)

        # --- Importance threshold policy ---
        imp_ids = self._apply_importance_threshold(
            user_id, p.importance_threshold
        )
        details["importance_threshold"] = len(imp_ids)
        result.expired_ids.extend(imp_ids)

        # --- Max-count policy ---
        count_ids = self._apply_max_count(user_id, p.max_count)
        details["max_count"] = len(count_ids)
        result.expired_ids.extend(count_ids)

        # Deduplicate
        result.expired_ids = list(dict.fromkeys(result.expired_ids))
        result.details = details

        _logger.info(
            "Expiry policies applied for user=%s: %d memories expired "
            "(ttl=%d, importance=%d, max_count=%d)",
            user_id,
            len(result.expired_ids),
            details["ttl"],
            details["importance_threshold"],
            details["max_count"],
        )
        return result

    # ------------------------------------------------------------------
    # Individual policies
    # ------------------------------------------------------------------

    def _apply_ttl(
        self, user_id: str | None, ttl_days: int | None
    ) -> list[str]:
        """Delete memories older than *ttl_days*."""
        if ttl_days is None:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
        return self._delete_and_collect_ids(
            user_id=user_id, before=cutoff
        )

    def _apply_importance_threshold(
        self, user_id: str | None, threshold: float | None
    ) -> list[str]:
        """Delete memories with importance below *threshold*."""
        if threshold is None:
            return []

        # Use max_importance parameter of delete_memories
        # (deletes where importance < threshold)
        # But we need IDs, so list first.
        try:
            memories = self.storage.list_memories(user_id=user_id)
        except Exception as exc:
            _logger.warning("Failed to list memories: %s", exc)
            return []

        to_delete = [
            m.id
            for m in memories
            if m.importance < threshold
        ]

        for mid in to_delete:
            try:
                self.storage.delete_memories(memory_id=mid)
            except Exception as exc:
                _logger.warning(
                    "Failed to delete memory %s: %s", mid, exc
                )

        _logger.debug(
            "Importance policy: deleted %d memories below %.2f",
            len(to_delete),
            threshold,
        )
        return to_delete

    def _apply_max_count(
        self, user_id: str | None, max_count: int | None
    ) -> list[str]:
        """Keep only the *max_count* most important memories."""
        if max_count is None:
            return []

        try:
            memories = self.storage.list_memories(user_id=user_id)
        except Exception as exc:
            _logger.warning("Failed to list memories: %s", exc)
            return []

        if len(memories) <= max_count:
            return []

        # Sort by importance descending; the tail are candidates for removal
        sorted_mems = sorted(
            memories, key=lambda m: m.importance, reverse=True
        )
        to_delete = [m.id for m in sorted_mems[max_count:]]

        for mid in to_delete:
            try:
                self.storage.delete_memories(memory_id=mid)
            except Exception as exc:
                _logger.warning(
                    "Failed to delete memory %s: %s", mid, exc
                )

        _logger.debug(
            "Max-count policy: deleted %d memories (keeping top %d)",
            len(to_delete),
            max_count,
        )
        return to_delete

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _delete_and_collect_ids(
        self,
        user_id: str | None = None,
        before: datetime | None = None,
    ) -> list[str]:
        """List memories matching criteria, collect IDs, then delete."""
        try:
            memories = self.storage.list_memories(user_id=user_id)
        except Exception as exc:
            _logger.warning("Failed to list memories for TTL: %s", exc)
            return []

        if before is not None:
            to_delete = [
                m.id for m in memories
                if m.created_at < before
            ]
        else:
            to_delete = []

        for mid in to_delete:
            try:
                self.storage.delete_memories(memory_id=mid)
            except Exception as exc:
                _logger.warning(
                    "Failed to delete memory %s: %s", mid, exc
                )

        return to_delete
