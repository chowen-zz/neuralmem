"""Qdrant vector store backend."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from neuralmem.core.exceptions import StorageError
from neuralmem.core.types import Memory, MemoryScope, MemoryType
from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient  # type: ignore[import-untyped]
    from qdrant_client.models import (  # type: ignore[import-untyped]
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointIdsList,
        PointStruct,
        VectorParams,
    )
except ImportError:
    QdrantClient = None  # type: ignore[assignment,misc]


def _check_qdrant() -> None:
    if QdrantClient is None:
        raise ImportError(
            "QdrantVectorStore requires the qdrant-client package. "
            "Install with: pip install neuralmem[qdrant]"
        )


class QdrantVectorStore(StorageBackend):
    """Qdrant-backed vector store."""

    def __init__(
        self, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        _check_qdrant()
        cfg = {**(config or {}), **kwargs}
        self._dim: int = cfg.get("embedding_dim", 384)
        self._collection: str = cfg.get("collection_name", "neuralmem")

        url = cfg.get("url", "http://localhost:6333")
        api_key = cfg.get("api_key")
        if cfg.get(":memory:") or url == ":memory:":
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(url=url, api_key=api_key)

        # Ensure collection exists
        existing = [
            c.name for c in self._client.get_collections().collections
        ]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dim, distance=Distance.COSINE
                ),
            )

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save_memory(self, memory: Memory) -> str:
        embedding = memory.embedding or [0.0] * self._dim
        payload: dict[str, Any] = {
            "content": memory.content,
            "memory_type": memory.memory_type.value,
            "scope": memory.scope.value,
            "user_id": memory.user_id or "",
            "agent_id": memory.agent_id or "",
            "session_id": memory.session_id or "",
            "tags": list(memory.tags),
            "source": memory.source or "",
            "importance": memory.importance,
            "entity_ids": list(memory.entity_ids),
            "is_active": memory.is_active,
            "superseded_by": memory.superseded_by or "",
            "supersedes": list(memory.supersedes),
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "last_accessed": memory.last_accessed.isoformat(),
            "access_count": memory.access_count,
            "expires_at": (
                memory.expires_at.isoformat() if memory.expires_at else ""
            ),
        }
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=memory.id, vector=embedding, payload=payload
                )
            ],
        )
        return memory.id

    # ------------------------------------------------------------------
    # get / list
    # ------------------------------------------------------------------

    def get_memory(self, memory_id: str) -> Memory | None:
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=[memory_id],
            with_vectors=True,
            with_payload=True,
        )
        if not points:
            return None
        return _memory_from_qdrant(points[0])

    def list_memories(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Memory]:
        from qdrant_client.models import Filter as _F
        from qdrant_client.models import MatchValue as _M

        scroll_filter = None
        if user_id:
            scroll_filter = _F(
                must=[_F(
                    must=[FieldCondition(
                        key="user_id", match=_M(value=user_id)
                    )]
                )]
            )
        result, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )
        return [_memory_from_qdrant(p) for p in result]

    # ------------------------------------------------------------------
    # update / delete
    # ------------------------------------------------------------------

    def update_memory(self, memory_id: str, **kwargs: object) -> None:
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=[memory_id],
            with_vectors=True,
            with_payload=True,
        )
        if not points:
            raise StorageError(f"Memory {memory_id} not found")

        payload = dict(points[0].payload or {})
        for key, value in kwargs.items():
            if key == "tags":
                payload["tags"] = list(value)  # type: ignore[arg-type]
            elif key == "entity_ids":
                payload["entity_ids"] = list(value)  # type: ignore[arg-type]
            elif key == "supersedes":
                payload["supersedes"] = list(
                    value  # type: ignore[arg-type]
                )
            elif key == "is_active":
                payload["is_active"] = bool(value)
            elif key == "memory_type" and isinstance(value, MemoryType):
                payload["memory_type"] = value.value
            elif key == "scope" and isinstance(value, MemoryScope):
                payload["scope"] = value.value
            elif key == "expires_at":
                payload["expires_at"] = (
                    value.isoformat()
                    if isinstance(value, datetime)
                    else str(value or "")
                )
            else:
                payload[key] = value

        payload["updated_at"] = datetime.now(timezone.utc).isoformat()

        vector = list(points[0].vector) if points[0].vector else None
        if "embedding" in kwargs and kwargs["embedding"] is not None:
            vector = list(kwargs["embedding"])  # type: ignore[arg-type]

        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=memory_id, vector=vector, payload=payload
                )
            ],
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
            self._client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=[memory_id]),
            )
            return 1

        # Collect matching points via scroll
        result, _ = self._client.scroll(
            collection_name=self._collection,
            limit=10_000,
            with_payload=True,
        )
        ids_to_delete: list[str] = []
        for point in result:
            payload = point.payload or {}
            if user_id and payload.get("user_id") != user_id:
                continue
            if before is not None:
                created = payload.get("created_at", "")
                before_str = (
                    before.isoformat()
                    if isinstance(before, datetime)
                    else str(before)
                )
                if created >= before_str:
                    continue
            if tags:
                ptags = payload.get("tags", [])
                if not any(t in ptags for t in tags):
                    continue
            if max_importance is not None:
                if payload.get("importance", 0.0) >= max_importance:
                    continue
            ids_to_delete.append(point.id)

        if ids_to_delete:
            self._client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=ids_to_delete),
            )
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
        query_filter = _build_qdrant_filter(user_id, memory_types)
        results = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
        )
        pairs: list[tuple[str, float]] = []
        for hit in results.points:
            score = max(0.0, hit.score)
            pairs.append((str(hit.id), score))
        return pairs

    # ------------------------------------------------------------------
    # keyword search (brute-force payload scan)
    # ------------------------------------------------------------------

    def keyword_search(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        query_lower = query.lower()
        result, _ = self._client.scroll(
            collection_name=self._collection,
            limit=10_000,
            with_payload=True,
        )
        pairs: list[tuple[str, float]] = []
        for point in result:
            payload = point.payload or {}
            if user_id and payload.get("user_id") != user_id:
                continue
            if memory_types:
                mt = payload.get("memory_type", "")
                if mt not in [t.value for t in memory_types]:
                    continue
            content = payload.get("content", "").lower()
            if query_lower in content:
                pairs.append((str(point.id), 1.0))
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
        similar: list[Memory] = []
        for mid, score in results:
            if score >= threshold:
                mem = self.get_memory(mid)
                if mem is not None:
                    similar.append(mem)
        return similar

    # ------------------------------------------------------------------
    # access tracking
    # ------------------------------------------------------------------

    def record_access(self, memory_id: str) -> None:
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=[memory_id],
            with_vectors=True,
            with_payload=True,
        )
        if not points:
            return
        payload = dict(points[0].payload or {})
        payload["access_count"] = payload.get("access_count", 0) + 1
        payload["last_accessed"] = (
            datetime.now(timezone.utc).isoformat()
        )
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=list(points[0].vector)
                    if points[0].vector
                    else None,
                    payload=payload,
                )
            ],
        )

    def batch_record_access(self, memory_ids: list[str]) -> None:
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
            "Qdrant backend: history saved for %s (event=%s)",
            memory_id, event,
        )

    def get_history(self, memory_id: str) -> list[dict]:
        return []

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def get_stats(self, user_id: str | None = None) -> dict[str, object]:
        info = self._client.get_collection(self._collection)
        return {
            "backend": "qdrant",
            "total_memories": info.points_count,
            "user_filter": user_id or "all",
        }

    # ------------------------------------------------------------------
    # graph snapshots (no-op)
    # ------------------------------------------------------------------

    def load_graph_snapshot(self) -> dict | None:
        return None

    def save_graph_snapshot(self, data: dict) -> None:
        _logger.debug("Qdrant backend: graph snapshot saved (no-op)")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _memory_from_qdrant(point: Any) -> Memory:
    """Build a Memory from a Qdrant point."""
    payload = point.payload or {}
    return Memory(
        id=str(point.id),
        content=payload.get("content", ""),
        memory_type=MemoryType(
            payload.get("memory_type", "semantic")
        ),
        scope=MemoryScope(payload.get("scope", "user")),
        user_id=payload.get("user_id") or None,
        agent_id=payload.get("agent_id") or None,
        session_id=payload.get("session_id") or None,
        tags=tuple(payload.get("tags", [])),
        source=payload.get("source") or None,
        importance=payload.get("importance", 0.5),
        entity_ids=tuple(payload.get("entity_ids", [])),
        is_active=bool(payload.get("is_active", True)),
        superseded_by=payload.get("superseded_by") or None,
        supersedes=tuple(payload.get("supersedes", [])),
        created_at=datetime.fromisoformat(payload["created_at"]),
        updated_at=datetime.fromisoformat(payload["updated_at"]),
        last_accessed=datetime.fromisoformat(
            payload["last_accessed"]
        ),
        access_count=payload.get("access_count", 0),
        embedding=list(point.vector) if point.vector else None,
        expires_at=(
            datetime.fromisoformat(payload["expires_at"])
            if payload.get("expires_at")
            else None
        ),
    )


def _build_qdrant_filter(
    user_id: str | None,
    memory_types: list[MemoryType] | None,
) -> Any:
    """Build a Qdrant Filter object."""
    conditions = []
    if user_id:
        conditions.append(
            FieldCondition(
                key="user_id", match=MatchValue(value=user_id)
            )
        )
    if memory_types:
        for mt in memory_types:
            conditions.append(
                FieldCondition(
                    key="memory_type",
                    match=MatchValue(value=mt.value),
                )
            )
    if not conditions:
        return None
    return Filter(must=conditions)
