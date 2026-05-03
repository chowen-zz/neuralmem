"""Async wrapper around NeuralMem for use with FastAPI, asyncio, and other async frameworks."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

from neuralmem.core.memory import NeuralMem
from neuralmem.core.types import Memory, MemoryHistoryEntry, MemoryType, SearchResult


class AsyncNeuralMem:
    """Async wrapper around NeuralMem for use with FastAPI, asyncio, etc.

    Offloads all synchronous NeuralMem operations to a thread pool so they
    can be awaited without blocking the event loop.

    Usage::

        from neuralmem import NeuralMem
        from neuralmem.core.async_memory import AsyncNeuralMem

        mem = NeuralMem()
        async with AsyncNeuralMem(mem) as amem:
            await amem.remember("The user likes Python")
            results = await amem.recall("What does the user like?")

    You can also instantiate directly without the context manager::

        amem = AsyncNeuralMem(mem)
        await amem.remember("some fact")
        await amem.close()
    """

    def __init__(self, neural_mem: NeuralMem, max_workers: int = 4) -> None:
        """Initialize the async wrapper.

        Args:
            neural_mem: The synchronous NeuralMem instance to wrap.
            max_workers: Maximum number of threads in the executor pool.
        """
        self._mem = neural_mem
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    # ==================== Core API ====================

    async def remember(
        self,
        content: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
        expires_at: datetime | None = None,
        expires_in: timedelta | None = None,
    ) -> list[Memory]:
        """Store a memory (async).

        Delegates to NeuralMem.remember() in a thread pool.

        Args:
            content: The content string to remember.
            user_id: User identifier for memory scoping.
            agent_id: Agent identifier.
            session_id: Session identifier.
            memory_type: Override automatic type detection.
            tags: Tags for categorization.
            importance: Importance score (0.0–1.0).
            expires_at: Absolute expiration time (UTC).
            expires_in: Relative expiration (e.g. timedelta(hours=1)).

        Returns:
            List of extracted and stored Memory objects.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._mem.remember(
                content,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                memory_type=memory_type,
                tags=tags,
                importance=importance,
                expires_at=expires_at,
                expires_in=expires_in,
            ),
        )

    async def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        tags: list[str] | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 10,
        min_score: float = 0.3,
    ) -> list[SearchResult]:
        """Retrieve relevant memories (async).

        Delegates to NeuralMem.recall() in a thread pool.

        Args:
            query: The search query string.
            user_id: Filter by user.
            agent_id: Filter by agent.
            memory_types: Filter by memory types.
            tags: Filter by tags.
            time_range: (start, end) datetime tuple.
            limit: Maximum results to return.
            min_score: Minimum similarity score threshold.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._mem.recall(
                query,
                user_id=user_id,
                agent_id=agent_id,
                memory_types=memory_types,
                tags=tags,
                time_range=time_range,
                limit=limit,
                min_score=min_score,
            ),
        )

    async def reflect(
        self,
        topic: str,
        *,
        user_id: str | None = None,
        depth: int = 2,
    ) -> str:
        """Reflect on a topic using memory retrieval and graph traversal (async).

        Delegates to NeuralMem.reflect() in a thread pool.

        Args:
            topic: The topic to reflect on.
            user_id: Filter by user.
            depth: Graph traversal depth.

        Returns:
            A structured reflection report string.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._mem.reflect(topic, user_id=user_id, depth=depth),
        )

    async def forget(
        self,
        memory_id: str | None = None,
        *,
        user_id: str | None = None,
        before: datetime | None = None,
        tags: list[str] | None = None,
    ) -> int:
        """Delete memories (async). Supports GDPR-compliant full deletion.

        Delegates to NeuralMem.forget() in a thread pool.

        Args:
            memory_id: Specific memory ID to delete.
            user_id: Delete all memories for this user.
            before: Delete memories created before this datetime.
            tags: Delete memories with any of these tags.

        Returns:
            Number of memories deleted.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._mem.forget(
                memory_id=memory_id,
                user_id=user_id,
                before=before,
                tags=tags,
            ),
        )

    async def remember_batch(
        self,
        contents: list[str],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        progress_callback: Any | None = None,
    ) -> list[Memory]:
        """Batch remember multiple items (async).

        Delegates to NeuralMem.remember_batch() in a thread pool.

        Args:
            contents: List of content strings to remember.
            user_id: User identifier for memory scoping.
            agent_id: Agent identifier.
            memory_type: Override memory type for all items.
            tags: Tags to apply to all items.
            progress_callback: Optional callback(current, total, content_preview).

        Returns:
            List of all stored memories across all items.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._mem.remember_batch(
                contents,
                user_id=user_id,
                agent_id=agent_id,
                memory_type=memory_type,
                tags=tags,
                progress_callback=progress_callback,
            ),
        )

    async def export_memories(
        self,
        *,
        user_id: str | None = None,
        format: str = "json",
        include_embeddings: bool = False,
    ) -> str:
        """Export memories as JSON, markdown, or CSV (async).

        Delegates to NeuralMem.export_memories() in a thread pool.

        Args:
            user_id: Filter memories by user. None = all users.
            format: Output format - "json", "markdown", or "csv".
            include_embeddings: If False (default), omit embedding vectors.

        Returns:
            Formatted string of exported memories.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._mem.export_memories(
                user_id=user_id,
                format=format,
                include_embeddings=include_embeddings,
            ),
        )

    async def import_memories(
        self,
        data: str,
        *,
        format: str = "json",
        user_id: str | None = None,
        skip_duplicates: bool = True,
    ) -> int:
        """Import memories from exported data (async).

        Delegates to NeuralMem.import_memories() in a thread pool.

        Args:
            data: The exported data string.
            format: "json", "markdown", or "csv".
            user_id: Override user_id for imported memories.
            skip_duplicates: If True, skip memories with similar content already stored.

        Returns:
            Number of memories imported.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._mem.import_memories(
                data,
                format=format,
                user_id=user_id,
                skip_duplicates=skip_duplicates,
            ),
        )

    async def consolidate(self, user_id: str | None = None) -> dict[str, int]:
        """Run memory consolidation: decay old memories, merge similar ones (async).

        Delegates to NeuralMem.consolidate() in a thread pool.

        Args:
            user_id: Limit consolidation to a specific user. None = all users.

        Returns:
            Dict with counts: decayed, forgotten, merged.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._mem.consolidate(user_id=user_id),
        )

    async def cleanup_expired(self) -> int:
        """Remove all expired memories (async).

        Returns:
            Number of expired memories deleted.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._mem.cleanup_expired,
        )

    async def get_stats(self) -> dict[str, object]:
        """Return memory store statistics (async).

        Returns:
            Dict containing storage and graph statistics.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._mem.get_stats,
        )

    async def resolve_conflict(
        self,
        memory_id: str,
        *,
        action: str = "reactivate",
    ) -> bool:
        """Resolve a memory conflict manually (async).

        Args:
            memory_id: Target memory ID.
            action: "reactivate" (re-enable) or "delete" (permanent removal).

        Returns:
            Whether the operation succeeded.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._mem.resolve_conflict(memory_id, action=action),
        )

    async def get(self, memory_id: str) -> Memory | None:
        """Retrieve a single memory by ID (async).

        Delegates to NeuralMem.get() in a thread pool.

        Args:
            memory_id: The memory identifier.

        Returns:
            Memory object if found, None otherwise.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._mem.get(memory_id),
        )

    async def update(
        self,
        memory_id: str,
        content: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> Memory | None:
        """Update a memory's content (async). Records version history automatically.

        Delegates to NeuralMem.update() in a thread pool.

        Args:
            memory_id: The memory to update.
            content: New content text.
            metadata: Optional metadata for the history entry.

        Returns:
            Updated Memory object, or None if not found.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._mem.update(memory_id, content, metadata=metadata),
        )

    async def history(self, memory_id: str) -> list[MemoryHistoryEntry]:
        """Retrieve the version history for a memory (async).

        Delegates to NeuralMem.history() in a thread pool.

        Args:
            memory_id: The memory to get history for.

        Returns:
            List of MemoryHistoryEntry objects, ordered chronologically.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._mem.history(memory_id),
        )

    async def forget_batch(
        self,
        memory_ids: list[str] | None = None,
        *,
        user_id: str | None = None,
        tags: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, object]:
        """Batch delete with dry_run preview (async).

        Args:
            memory_ids: List of specific memory IDs to delete.
            user_id: Delete all memories for this user.
            tags: Delete memories with any of these tags.
            dry_run: If True, return what would be deleted without actually deleting.

        Returns:
            Dict with 'count' (int), 'memory_ids' (list[str]), and 'dry_run' (bool).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._mem.forget_batch(
                memory_ids=memory_ids,
                user_id=user_id,
                tags=tags,
                dry_run=dry_run,
            ),
        )

    # ==================== Lifecycle ====================

    async def close(self) -> None:
        """Shut down the thread pool executor.

        Safe to call multiple times.
        """
        self._executor.shutdown(wait=False)

    # ==================== Context Manager ====================

    async def __aenter__(self) -> AsyncNeuralMem:
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the async context manager, shutting down the executor."""
        await self.close()
