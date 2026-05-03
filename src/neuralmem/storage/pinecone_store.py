"""Pinecone vector store backend."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from neuralmem.core.exceptions import StorageError
from neuralmem.core.types import Memory, MemoryScope, MemoryType
from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)

try:
    from pinecone import Pinecone  # type: ignore[import-untyped]
except ImportError:
    Pinecone = None  # type: ignore[assignment,misc]


def _check_pinecone() -> None:
    if Pinecone is None:
        raise ImportError(
            "PineconeVectorStore requires the pinecone package. "
            "Install with: pip install neuralmem[pinecone]"
        )


class PineconeVectorStore(StorageBackend):
    """Pinecone-backed vector store.

    Uses upsert-based storage with metadata filtering.
    """

    def __init__(
        self, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        _check_pinecone()
        cfg = {**(config or {}), **kwargs}
        self._dim: int = cfg.get("embedding_dim", 384)
        self._index_name: str = cfg.get("index_name", "neuralmem")

        api_key = cfg.get("api_key", "")
        self._client = Pinecone(api_key=api_key)

        # Check if index exists; list_indexes returns names
        existing = [i.name for i in self._client.list_indexes()]
        if self._index_name not in existing:
            self._client.create_index(
                name=self._index_name,
                dimension=self._dim,
                metric="cosine",
            )

        self._index = self._client.Index(self._index_name)

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save_memory(self, memory: Memory) -> str:
        embedding = memory.embedding or [0.0] * self._dim
        metadata: dict[str, Any] = {
            "content": memory.content,
            "memory_type": memory.memory_type.value,
            "scope": memory.scope.value,
            "user_id": memory.user_id or "",
            "agent_id": memory.agent_id or "",
            "session_id": memory.session_id or "",
            "tags": json.dumps(list(memory.tags)),
            "source": memory.source or "",
            "importance": memory.importance,
            "entity_ids": json.dumps(list(memory.entity_ids)),
            "is_active": int(memory.is_active),
            "superseded_by": memory.superseded_by or "",
            "supersedes": json.dumps(list(memory.supersedes)),
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "last_accessed": memory.last_accessed.isoformat(),
            "access_count": memory.access_count,
            "expires_at": (
                memory.expires_at.isoformat()
                if memory.expires_at
                else ""
            ),
        }
        self._index.upsert(
            vectors=[
                {
                    "id": memory.id,
                    "values": embedding,
                    "metadata": metadata,
                }
            ]
        )
        return memory.id

    # ------------------------------------------------------------------
    # get / list
    # ------------------------------------------------------------------

    def get_memory(self, memory_id: str) -> Memory | None:
        result = self._index.fetch(ids=[memory_id])
        vectors = result.get("vectors", {})
        if memory_id not in vectors:
            return None
        vec = vectors[memory_id]
        return _memory_from_pinecone(vec, memory_id)

    def list_memories(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Memory]:
        # Pinecone list does not support filter directly;
        # use a dummy vector query with metadata filter
        filter_dict: dict[str, Any] | None = None
        if user_id:
            filter_dict = {"user_id": {"$eq": user_id}}

        results = self._index.query(
            vector=[0.0] * self._dim,
            top_k=limit,
            include_metadata=True,
            include_values=True,
            filter=filter_dict,
        )
        memories: list[Memory] = []
        for match in results.get("matches", []):
            mid = match["id"]
            vec_data = {
                "id": mid,
                "values": match.get("values"),
                "metadata": match.get("metadata", {}),
            }
            memories.append(_memory_from_pinecone(vec_data, mid))
        return memories

    # ------------------------------------------------------------------
    # update / delete
    # ------------------------------------------------------------------

    def update_memory(self, memory_id: str, **kwargs: object) -> None:
        result = self._index.fetch(ids=[memory_id])
        vectors = result.get("vectors", {})
        if memory_id not in vectors:
            raise StorageError(f"Memory {memory_id} not found")

        vec = vectors[memory_id]
        metadata = dict(vec.get("metadata", {}))
        for key, value in kwargs.items():
            if key == "tags":
                metadata["tags"] = json.dumps(
                    list(value)  # type: ignore[arg-type]
                )
            elif key == "entity_ids":
                metadata["entity_ids"] = json.dumps(
                    list(value)  # type: ignore[arg-type]
                )
            elif key == "supersedes":
                metadata["supersedes"] = json.dumps(
                    list(value)  # type: ignore[arg-type]
                )
            elif key == "is_active":
                metadata["is_active"] = int(bool(value))
            elif key == "memory_type" and isinstance(
                value, MemoryType
            ):
                metadata["memory_type"] = value.value
            elif key == "scope" and isinstance(value, MemoryScope):
                metadata["scope"] = value.value
            elif key == "expires_at":
                metadata["expires_at"] = (
                    value.isoformat()
                    if isinstance(value, datetime)
                    else str(value or "")
                )
            else:
                metadata[key] = value

        metadata["updated_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

        values = list(vec.get("values") or [0.0] * self._dim)
        if "embedding" in kwargs and kwargs["embedding"] is not None:
            values = list(
                kwargs["embedding"]  # type: ignore[arg-type]
            )

        self._index.upsert(
            vectors=[
                {
                    "id": memory_id,
                    "values": values,
                    "metadata": metadata,
                }
            ]
        )

    def delete_memories(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        before: object = None,
        tags: list[str] | None = None,
        max_importance: float | None = None,
    ) -> int:
        if memory_id is not None:
            self._index.delete(ids=[memory_id])
            return 1

        # Fetch all via dummy query and filter
        results = self._index.query(
            vector=[0.0] * self._dim,
            top_k=10_000,
            include_metadata=True,
        )
        ids_to_delete: list[str] = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            mid = match["id"]
            if user_id and metadata.get("user_id") != user_id:
                continue
            if before is not None:
                created = metadata.get("created_at", "")
                before_str = (
                    before.isoformat()
                    if isinstance(before, datetime)
                    else str(before)
                )
                if created >= before_str:
                    continue
            if tags:
                meta_tags = json.loads(
                    metadata.get("tags", "[]")
                )
                if not any(t in meta_tags for t in tags):
                    continue
            if max_importance is not None:
                if metadata.get("importance", 0.0) >= max_importance:
                    continue
            ids_to_delete.append(mid)

        if ids_to_delete:
            self._index.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    # ------------------------------------------------------------------
    # vector search
    # ------------------------------------------------------------------

    def vector_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        filter_dict = _build_pinecone_filter(
            user_id, memory_types
        )
        results = self._index.query(
            vector=vector,
            top_k=limit,
            include_metadata=False,
            filter=filter_dict,
        )
        pairs: list[tuple[str, float]] = []
        for match in results.get("matches", []):
            score = max(0.0, match.get("score", 0.0))
            pairs.append((match["id"], score))
        return pairs

    # ------------------------------------------------------------------
    # keyword search
    # ------------------------------------------------------------------

    def keyword_search(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        query_lower = query.lower()
        filter_dict: dict[str, Any] | None = None
        if user_id:
            filter_dict = {"user_id": {"$eq": user_id}}

        results = self._index.query(
            vector=[0.0] * self._dim,
            top_k=10_000,
            include_metadata=True,
            filter=filter_dict,
        )
        pairs: list[tuple[str, float]] = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            if memory_types:
                mt = metadata.get("memory_type", "")
                if mt not in [t.value for t in memory_types]:
                    continue
            content = metadata.get("content", "").lower()
            if query_lower in content:
                pairs.append((match["id"], 1.0))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:limit]

    # ------------------------------------------------------------------
    # temporal search
    # ------------------------------------------------------------------

    def temporal_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        time_range: tuple[object, object] | None = None,
        recency_weight: float = 0.3,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        results = self.vector_search(
            vector=vector, user_id=user_id, limit=limit * 2
        )
        if not results:
            return []

        now = datetime.now(timezone.utc)
        scored: list[tuple[str, float]] = []
        for mid, sim_score in results:
            mem = self.get_memory(mid)
            if mem is None:
                continue
            if time_range:
                start, end = time_range
                start_dt = (
                    start
                    if isinstance(start, datetime)
                    else datetime.fromisoformat(str(start))
                )
                end_dt = (
                    end
                    if isinstance(end, datetime)
                    else datetime.fromisoformat(str(end))
                )
                if not (start_dt <= mem.created_at <= end_dt):
                    continue
            age_hours = (
                now - mem.last_accessed
            ).total_seconds() / 3600.0
            recency = 1.0 / (1.0 + age_hours / 24.0)
            combined = (
                (1.0 - recency_weight) * sim_score
                + recency_weight * recency
            )
            scored.append((mid, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # find_similar
    # ------------------------------------------------------------------

    def find_similar(
        self,
        vector: list[float],
        user_id: str | None = None,
        threshold: float = 0.95,
    ) -> list[Memory]:
        results = self.vector_search(
            vector=vector, user_id=user_id, limit=50
        )
        similar_ids = [
            mid for mid, score in results if score >= threshold
        ]
        if not similar_ids:
            return []
        memories: list[Memory] = []
        for mid in similar_ids:
            mem = self.get_memory(mid)
            if mem is not None:
                memories.append(mem)
        return memories

    # ------------------------------------------------------------------
    # access tracking
    # ------------------------------------------------------------------

    def record_access(self, memory_id: str) -> None:
        result = self._index.fetch(ids=[memory_id])
        vectors = result.get("vectors", {})
        if memory_id not in vectors:
            return
        vec = vectors[memory_id]
        metadata = dict(vec.get("metadata", {}))
        metadata["access_count"] = (
            metadata.get("access_count", 0) + 1
        )
        metadata["last_accessed"] = (
            datetime.now(timezone.utc).isoformat()
        )
        values = list(vec.get("values") or [0.0] * self._dim)
        self._index.upsert(
            vectors=[
                {
                    "id": memory_id,
                    "values": values,
                    "metadata": metadata,
                }
            ]
        )

    def batch_record_access(
        self, memory_ids: list[str]
    ) -> None:
        for mid in memory_ids:
            self.record_access(mid)

    # ------------------------------------------------------------------
    # history (no-op)
    # ------------------------------------------------------------------

    def save_history(
        self,
        memory_id: str,
        old_content: str | None,
        new_content: str,
        event: str,
        metadata: dict | None = None,
    ) -> None:
        _logger.debug(
            "Pinecone backend: history saved for %s (event=%s)",
            memory_id,
            event,
        )

    def get_history(self, memory_id: str) -> list[dict]:
        return []

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def get_stats(
        self, user_id: str | None = None
    ) -> dict[str, object]:
        stats = self._index.describe_index_stats()
        return {
            "backend": "pinecone",
            "total_memories": stats.get(
                "total_vector_count", 0
            ),
            "user_filter": user_id or "all",
        }

    # ------------------------------------------------------------------
    # graph snapshots (no-op)
    # ------------------------------------------------------------------

    def load_graph_snapshot(self) -> dict | None:
        return None

    def save_graph_snapshot(self, data: dict) -> None:
        _logger.debug(
            "Pinecone backend: graph snapshot saved (no-op)"
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _memory_from_pinecone(
    vec_data: dict[str, Any], mid: str
) -> Memory:
    """Build a Memory from Pinecone vector data."""
    meta = vec_data.get("metadata", {})
    values = vec_data.get("values")
    return Memory(
        id=mid,
        content=meta.get("content", ""),
        memory_type=MemoryType(
            meta.get("memory_type", "semantic")
        ),
        scope=MemoryScope(meta.get("scope", "user")),
        user_id=meta.get("user_id") or None,
        agent_id=meta.get("agent_id") or None,
        session_id=meta.get("session_id") or None,
        tags=tuple(json.loads(meta.get("tags", "[]"))),
        source=meta.get("source") or None,
        importance=meta.get("importance", 0.5),
        entity_ids=tuple(
            json.loads(meta.get("entity_ids", "[]"))
        ),
        is_active=bool(meta.get("is_active", 1)),
        superseded_by=meta.get("superseded_by") or None,
        supersedes=tuple(
            json.loads(meta.get("supersedes", "[]"))
        ),
        created_at=datetime.fromisoformat(
            meta["created_at"]
        ),
        updated_at=datetime.fromisoformat(
            meta["updated_at"]
        ),
        last_accessed=datetime.fromisoformat(
            meta["last_accessed"]
        ),
        access_count=meta.get("access_count", 0),
        embedding=list(values) if values else None,
        expires_at=(
            datetime.fromisoformat(meta["expires_at"])
            if meta.get("expires_at")
            else None
        ),
    )


def _build_pinecone_filter(
    user_id: str | None,
    memory_types: list[MemoryType] | None,
) -> dict[str, Any] | None:
    """Build a Pinecone metadata filter."""
    conditions: list[dict[str, Any]] = []
    if user_id:
        conditions.append({"user_id": {"$eq": user_id}})
    if memory_types:
        types = [mt.value for mt in memory_types]
        if len(types) == 1:
            conditions.append(
                {"memory_type": {"$eq": types[0]}}
            )
        else:
            conditions.append(
                {"memory_type": {"$in": types}}
            )
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
