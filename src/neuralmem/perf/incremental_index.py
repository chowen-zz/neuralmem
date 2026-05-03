"""Incremental index tracker with hash-based change detection."""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from neuralmem.core.protocols import EmbedderProtocol, StorageProtocol

_logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """Statistics about the incremental index."""
    dirty_count: int = 0
    total_indexed: int = 0
    last_reindex_time: float = 0.0
    total_reindexes: int = 0


class IncrementalIndex:
    """Track which memories need re-indexing using a dirty set.

    Uses hash-based change detection to identify memories whose
    content has actually changed and need re-embedding.
    Thread-safe via Lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dirty: set[str] = set()
        self._clean: set[str] = set()
        self._content_hashes: dict[str, str] = {}
        self._last_reindex_time: float = 0.0
        self._total_reindexes: int = 0

    def mark_dirty(self, memory_id: str) -> None:
        """Mark a memory for re-indexing."""
        with self._lock:
            self._dirty.add(memory_id)
            self._clean.discard(memory_id)

    def mark_clean(self, memory_id: str) -> None:
        """Mark a memory as successfully indexed."""
        with self._lock:
            self._clean.add(memory_id)
            self._dirty.discard(memory_id)

    def get_dirty(self) -> set[str]:
        """Return all dirty memory IDs."""
        with self._lock:
            return set(self._dirty)

    def update_content_hash(
        self, memory_id: str, content: str
    ) -> bool:
        """Update the content hash for a memory.

        Returns True if the content actually changed
        (memory should be marked dirty), False if unchanged.
        """
        new_hash = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()
        with self._lock:
            old_hash = self._content_hashes.get(memory_id)
            if old_hash == new_hash:
                return False
            self._content_hashes[memory_id] = new_hash
            return True

    def reindex_dirty(
        self,
        embedder: EmbedderProtocol,
        storage: StorageProtocol,
        content_getter: Callable[[str, StorageProtocol], str | None] | None = None,
    ) -> int:
        """Batch re-index only dirty memories.

        Args:
            embedder: Embedding provider for encoding content.
            storage: Storage backend to read/write memories.
            content_getter: Optional callable(memory_id, storage)
                -> content string. If None, uses memory.content.

        Returns:
            Number of memories successfully re-indexed.
        """
        with self._lock:
            dirty_ids = list(self._dirty)

        if not dirty_ids:
            return 0

        count = 0
        for mid in dirty_ids:
            try:
                if content_getter is not None:
                    content = content_getter(mid, storage)
                else:
                    memory = storage.get_memory(mid)
                    if memory is None:
                        with self._lock:
                            self._dirty.discard(mid)
                        continue
                    content = getattr(memory, "content", None)
                    if content is None:
                        continue

                # Check if content actually changed
                new_hash = hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest()
                with self._lock:
                    old_hash = self._content_hashes.get(mid)
                    if old_hash == new_hash and mid in self._clean:
                        self._dirty.discard(mid)
                        continue
                    self._content_hashes[mid] = new_hash

                # Encode and update
                embedding = embedder.encode_one(content)
                storage.update_memory(mid, embedding=embedding)
                with self._lock:
                    self._clean.add(mid)
                    self._dirty.discard(mid)
                count += 1
            except Exception as exc:
                _logger.warning(
                    "Failed to reindex memory %s: %s", mid, exc
                )

        with self._lock:
            self._last_reindex_time = time.monotonic()
            self._total_reindexes += 1

        return count

    def stats(self) -> IndexStats:
        """Return current index statistics."""
        with self._lock:
            return IndexStats(
                dirty_count=len(self._dirty),
                total_indexed=len(self._clean),
                last_reindex_time=self._last_reindex_time,
                total_reindexes=self._total_reindexes,
            )
