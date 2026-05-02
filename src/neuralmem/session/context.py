"""Session-aware memory context manager with 3-layer architecture.

Layers:
    WORKING  — in-context items for current turn (not persisted, ephemeral)
    SESSION  — persisted to DB with session_layer='session', compressed on close
    LONG_TERM — persisted to DB with session_layer='long_term', permanent
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from neuralmem.core.exceptions import StorageError
from neuralmem.core.types import Memory, SessionLayer

if TYPE_CHECKING:
    from neuralmem.core.memory import NeuralMem

_logger = logging.getLogger(__name__)

# Default importance threshold for promoting session memories to long-term
_PROMOTE_IMPORTANCE_THRESHOLD = 0.7


class SessionContext:
    """Context manager that provides 3-layer session-aware memory.

    Usage::

        with mem.session(user_id="u1") as ctx:
            # Layer 1: working memory (ephemeral, in-process only)
            ctx.append_working("The user asked about Python decorators")

            # Layer 2: session memory (persisted to DB, compressed on exit)
            ctx.remember_to_session("User prefers concise explanations")

            # Layer 3: long-term recall (searches all layers)
            results = ctx.recall("What does the user prefer?")

        # On exit: session memories are compressed and important ones promoted
    """

    def __init__(
        self,
        mem: NeuralMem,
        conversation_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        promote_threshold: float = _PROMOTE_IMPORTANCE_THRESHOLD,
    ) -> None:
        self._mem = mem
        self.conversation_id: str = conversation_id or uuid4().hex[:16]
        self.user_id = user_id
        self.agent_id = agent_id
        self._promote_threshold = promote_threshold

        # Layer 1: working memory — list of plain strings, never persisted
        self._working_memory: list[str] = []

        # Track session-layer memory IDs for compression on exit
        self._session_memory_ids: list[str] = []

    # ---- Context manager protocol ----

    def __enter__(self) -> SessionContext:
        _logger.debug("Session %s started", self.conversation_id)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        try:
            self._compress_and_promote()
        except (StorageError, OSError):
            _logger.exception("Error during session end (conversation_id=%s)", self.conversation_id)
        finally:
            _logger.debug(
                "Session %s ended (%d session memories, %d working items)",
                self.conversation_id,
                len(self._session_memory_ids),
                len(self._working_memory),
            )

    # ---- Layer 1: Working memory ----

    def append_working(self, content: str) -> None:
        """Add an item to working memory (ephemeral, not persisted).

        Working memory is useful for tracking the current turn's context
        without polluting the persistent store.
        """
        self._working_memory.append(content)

    @property
    def working_memory(self) -> list[str]:
        """Return a copy of the current working memory items."""
        return list(self._working_memory)

    def clear_working(self) -> None:
        """Clear working memory."""
        self._working_memory.clear()

    # ---- Layer 2: Session memory ----

    def remember_to_session(
        self,
        content: str,
        *,
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> list[Memory]:
        """Persist content to the session layer.

        These memories are stored in the DB with session_layer='session'
        and will be compressed / promoted on session end.
        """
        memories = self._mem.remember(
            content,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self.conversation_id,
            tags=tags,
            importance=importance,
        )

        # Tag them as session-layer in storage
        for m in memories:
            self._mem.storage.update_memory(
                m.id,
                session_layer=SessionLayer.SESSION.value,
                conversation_id=self.conversation_id,
            )
            self._session_memory_ids.append(m.id)

        return memories

    # ---- Layer 3: Recall (all layers) ----

    def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        min_score: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Search across all three layers: working → session → long-term.

        Returns a unified list of results with a ``layer`` field indicating
        which layer the memory came from.
        """
        results: list[dict[str, Any]] = []

        # Layer 1: working memory — simple substring / keyword match
        for item in self._working_memory:
            if _simple_match(query, item):
                results.append({
                    "content": item,
                    "layer": SessionLayer.WORKING.value,
                    "score": 1.0,
                    "memory": None,
                })

        # Layers 2+3: delegate to NeuralMem.recall which hits the DB
        # (covers both session and long-term)
        db_results = self._mem.recall(
            query,
            user_id=self.user_id,
            agent_id=self.agent_id,
            limit=limit,
            min_score=min_score,
        )

        for sr in db_results:
            # Determine layer from stored session_layer field
            layer = self._get_session_layer(sr.memory)
            results.append({
                "content": sr.memory.content,
                "layer": layer,
                "score": sr.score,
                "memory": sr.memory,
            })

        # Sort by score descending
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    # ---- Session end: compress & promote ----

    def _compress_and_promote(self) -> None:
        """Compress session memories and promote important ones to long-term.

        - Memories with importance >= threshold are promoted (session_layer set
          to 'long_term').
        - Memories with importance < threshold are kept as-is but logged for
          future decay cleanup.
        """
        if not self._session_memory_ids:
            return

        promoted = 0
        kept = 0

        for mem_id in self._session_memory_ids:
            memory = self._mem.storage.get_memory(mem_id)
            if memory is None:
                continue

            if memory.importance >= self._promote_threshold:
                self._mem.storage.update_memory(
                    mem_id,
                    session_layer=SessionLayer.LONG_TERM.value,
                )
                promoted += 1
                _logger.debug(
                    "Promoted memory %s to long-term (importance=%.2f)",
                    mem_id[:8],
                    memory.importance,
                )
            else:
                kept += 1

        _logger.info(
            "Session %s: promoted %d, kept %d as session-layer",
            self.conversation_id,
            promoted,
            kept,
        )

    def _get_session_layer(self, memory: Memory) -> str:
        """Read the session_layer column for a memory from the DB."""
        try:
            cur = self._mem.storage._execute(
                "SELECT session_layer FROM memories WHERE id = ?",
                (memory.id,),
            )
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
        except StorageError:
            pass
        return SessionLayer.LONG_TERM.value


def _simple_match(query: str, text: str) -> bool:
    """Case-insensitive keyword containment check."""
    q_lower = query.lower()
    t_lower = text.lower()
    # Check if any word from the query appears in the text
    words = q_lower.split()
    if not words:
        return False
    return any(w in t_lower for w in words if len(w) > 2)
