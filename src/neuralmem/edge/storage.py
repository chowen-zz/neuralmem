"""KV-backed storage backend for Cloudflare Workers / edge runtimes.

Implements StorageBackend using Workers KV (or any dict-like KV interface)
for memory persistence, vector index, graph snapshots, and stats.

All operations are mock-testable: pass a mock ``kv`` dict in tests.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from neuralmem.core.types import Memory, MemoryType
from neuralmem.edge.config import EdgeConfig
from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)


class EdgeStorage(StorageBackend):
    """Storage backend backed by Workers KV (or any dict-like KV store).

    In production the ``kv`` object is a Cloudflare Workers KV namespace.
    In tests, pass a plain ``dict`` or a mock object implementing
    ``get(key, cache_ttl=None)`` and ``put(key, value)``.
    """

    def __init__(
        self,
        config: EdgeConfig | None = None,
        kv: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or EdgeConfig.from_env()
        # In Workers runtime this would be the KV namespace binding.
        # In tests, a plain dict acts as an in-memory KV store.
        self._kv = kv if kv is not None else {}
        self._memory_prefix = "mem:"
        self._index_prefix = "idx:"
        self._graph_prefix = "graph:"
        self._stats_prefix = "stats:"

    # --- KV helpers ---------------------------------------------------------

    def _kv_get(self, key: str) -> Any | None:
        """Read from KV. Returns None if key missing."""
        if hasattr(self._kv, "get"):
            # Workers KV returns a Promise; we assume resolved value here.
            val = self._kv.get(key)
            if hasattr(val, "__await__"):
                raise RuntimeError("EdgeStorage requires synchronous KV in tests; use await in Workers")
            return val
        return self._kv.get(key)

    def _kv_put(self, key: str, value: Any) -> None:
        """Write to KV."""
        if hasattr(self._kv, "put"):
            self._kv.put(key, value)
        else:
            self._kv[key] = value

    def _kv_delete(self, key: str) -> None:
        """Delete from KV."""
        if hasattr(self._kv, "delete"):
            self._kv.delete(key)
        else:
            self._kv.pop(key, None)

    def _kv_list(self, prefix: str) -> list[str]:
        """List keys with given prefix."""
        if hasattr(self._kv, "list"):
            raw = self._kv.list({"prefix": prefix})
            if isinstance(raw, dict):
                return [k["name"] for k in raw.get("keys", [])]
            return list(raw)
        return [k for k in self._kv if k.startswith(prefix)]

    # --- StorageBackend implementation --------------------------------------

    def save_memory(self, memory: Memory) -> str:
        """Persist a Memory to KV."""
        key = f"{self._memory_prefix}{memory.id}"
        # embedding is excluded from model_dump_json; store it separately
        data = memory.model_dump(mode="json")
        data["embedding"] = memory.embedding
        self._kv_put(key, json.dumps(data))
        # Update stats counter
        self._increment_stat("memory_count", 1)
        return memory.id

    def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a Memory by ID."""
        raw = self._kv_get(f"{self._memory_prefix}{memory_id}")
        if raw is None:
            return None
        if isinstance(raw, str):
            return Memory.model_validate_json(raw)
        if isinstance(raw, dict):
            return Memory.model_validate(raw)
        return None

    def update_memory(self, memory_id: str, **kwargs: object) -> None:
        """Partial update of a Memory."""
        mem = self.get_memory(memory_id)
        if mem is None:
            return
        # Build updated fields
        update_data = mem.model_dump()
        for k, v in kwargs.items():
            if k in update_data:
                update_data[k] = v
        update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        updated = Memory.model_validate(update_data)
        self._kv_put(f"{self._memory_prefix}{memory_id}", updated.model_dump_json())

    def delete_memories(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        before: object = None,
        tags: list[str] | None = None,
        max_importance: float | None = None,
    ) -> int:
        """Delete memories matching criteria. Returns count deleted."""
        keys = self._kv_list(self._memory_prefix)
        deleted = 0
        for key in keys:
            raw = self._kv_get(key)
            if raw is None:
                continue
            mem = (
                Memory.model_validate_json(raw)
                if isinstance(raw, str)
                else Memory.model_validate(raw)
            )
            # Match filters
            if memory_id is not None and mem.id != memory_id:
                continue
            if user_id is not None and mem.user_id != user_id:
                continue
            if max_importance is not None and mem.importance > max_importance:
                continue
            if tags is not None and not any(t in mem.tags for t in tags):
                continue
            self._kv_delete(key)
            deleted += 1
        self._increment_stat("memory_count", -deleted)
        return deleted

    def vector_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Approximate vector search using stored embeddings in KV.

        Note: Full brute-force scan for correctness; edge deployments
        with many memories should use a vector-specific KV index.
        """
        keys = self._kv_list(self._memory_prefix)
        scored: list[tuple[str, float]] = []
        for key in keys:
            raw = self._kv_get(key)
            if raw is None:
                continue
            mem = (
                Memory.model_validate_json(raw)
                if isinstance(raw, str)
                else Memory.model_validate(raw)
            )
            if user_id is not None and mem.user_id != user_id:
                continue
            if memory_types is not None and mem.memory_type not in memory_types:
                continue
            if mem.embedding is None:
                continue
            score = _cosine_similarity(vector, mem.embedding)
            scored.append((mem.id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def keyword_search(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Simple keyword search over memory content."""
        query_lower = query.lower()
        keys = self._kv_list(self._memory_prefix)
        scored: list[tuple[str, float]] = []
        for key in keys:
            raw = self._kv_get(key)
            if raw is None:
                continue
            mem = (
                Memory.model_validate_json(raw)
                if isinstance(raw, str)
                else Memory.model_validate(raw)
            )
            if user_id is not None and mem.user_id != user_id:
                continue
            if memory_types is not None and mem.memory_type not in memory_types:
                continue
            content_lower = mem.content.lower()
            if query_lower in content_lower:
                # Simple score: overlap ratio
                score = len(query_lower) / max(len(content_lower), 1)
                scored.append((mem.id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def temporal_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        time_range: tuple[object, object] | None = None,
        recency_weight: float = 0.3,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Temporal search combining vector similarity with recency."""
        vec_results = self.vector_search(vector, user_id=user_id, limit=limit * 2)
        now = datetime.now(timezone.utc)
        scored: list[tuple[str, float]] = []
        for mem_id, vec_score in vec_results:
            mem = self.get_memory(mem_id)
            if mem is None:
                continue
            if time_range is not None:
                created = mem.created_at
                if not (time_range[0] <= created <= time_range[1]):  # type: ignore[operator]
                    continue
            # Recency score: exponential decay
            age_seconds = (now - mem.created_at).total_seconds()
            recency_score = max(0.0, 1.0 - age_seconds / 86400.0)  # decay over 1 day
            combined = (1 - recency_weight) * vec_score + recency_weight * recency_score
            scored.append((mem_id, combined))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def find_similar(
        self, vector: list[float], user_id: str | None = None, threshold: float = 0.95
    ) -> list[Memory]:
        """Find memories with cosine similarity >= threshold."""
        results = self.vector_search(vector, user_id=user_id, limit=10_000)
        matches: list[Memory] = []
        for mem_id, score in results:
            if score >= threshold:
                mem = self.get_memory(mem_id)
                if mem is not None:
                    matches.append(mem)
        return matches

    def record_access(self, memory_id: str) -> None:
        """Record a single access event."""
        mem = self.get_memory(memory_id)
        if mem is None:
            return
        self.update_memory(
            memory_id,
            last_accessed=datetime.now(timezone.utc).isoformat(),
            access_count=mem.access_count + 1,
        )

    def save_history(
        self,
        memory_id: str,
        old_content: str | None,
        new_content: str,
        event: str,
        metadata: dict | None = None,
    ) -> None:
        """Append a history entry for a memory."""
        history_key = f"{self._memory_prefix}{memory_id}:history"
        entry = {
            "old_content": old_content,
            "new_content": new_content,
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        existing = self._kv_get(history_key)
        if existing is None:
            history = [entry]
        elif isinstance(existing, str):
            history = json.loads(existing)
            history.append(entry)
        elif isinstance(existing, list):
            history = existing + [entry]
        else:
            history = [entry]
        self._kv_put(history_key, json.dumps(history))

    def get_history(self, memory_id: str) -> list[dict]:
        """Retrieve history entries for a memory."""
        raw = self._kv_get(f"{self._memory_prefix}{memory_id}:history")
        if raw is None:
            return []
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, list):
            return raw
        return []

    def batch_record_access(self, memory_ids: list[str]) -> None:
        """Batch record access for multiple memories."""
        for mid in memory_ids:
            self.record_access(mid)

    def get_stats(self, user_id: str | None = None) -> dict[str, object]:
        """Return aggregate stats."""
        raw = self._kv_get(f"{self._stats_prefix}global")
        stats: dict[str, object] = {}
        if isinstance(raw, str):
            stats = json.loads(raw)
        elif isinstance(raw, dict):
            stats = dict(raw)
        if user_id is not None:
            # Count per-user
            keys = self._kv_list(self._memory_prefix)
            user_count = 0
            for key in keys:
                if ":history" in key:
                    continue
                mem = self.get_memory(key.replace(self._memory_prefix, ""))
                if mem is not None and mem.user_id == user_id:
                    user_count += 1
            stats["user_memory_count"] = user_count
        return stats

    def list_memories(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Memory]:
        """List all memories, optionally filtered by user."""
        keys = self._kv_list(self._memory_prefix)
        results: list[Memory] = []
        for key in keys:
            if ":history" in key:
                continue
            mem_id = key.replace(self._memory_prefix, "")
            mem = self.get_memory(mem_id)
            if mem is None:
                continue
            if user_id is not None and mem.user_id != user_id:
                continue
            results.append(mem)
            if len(results) >= limit:
                break
        return results

    def load_graph_snapshot(self) -> dict | None:
        """Load the full graph snapshot from KV."""
        raw = self._kv_get(f"{self._graph_prefix}snapshot")
        if raw is None:
            return None
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, dict):
            return raw
        return None

    def save_graph_snapshot(self, data: dict) -> None:
        """Save the full graph snapshot to KV."""
        self._kv_put(f"{self._graph_prefix}snapshot", json.dumps(data))

    # --- Internal helpers ---------------------------------------------------

    def _increment_stat(self, name: str, delta: int) -> None:
        key = f"{self._stats_prefix}global"
        raw = self._kv_get(key)
        stats: dict[str, Any] = {}
        if isinstance(raw, str):
            stats = json.loads(raw)
        elif isinstance(raw, dict):
            stats = dict(raw)
        current = stats.get(name, 0)
        if isinstance(current, int):
            stats[name] = current + delta
        else:
            stats[name] = delta
        self._kv_put(key, json.dumps(stats))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
