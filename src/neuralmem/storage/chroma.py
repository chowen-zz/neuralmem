"""ChromaDB vector store backend."""
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
    import chromadb  # type: ignore[import-untyped]
except ImportError:
    chromadb = None  # type: ignore[assignment]


def _check_chroma() -> None:
    if chromadb is None:
        raise ImportError(
            "ChromaVectorStore requires the chromadb package. "
            "Install with: pip install neuralmem[chroma]"
        )


class ChromaVectorStore(StorageBackend):
    """ChromaDB-backed vector store.

    Accepts a ``config`` dict (or keyword arguments that will be forwarded
    to Chroma's ``Client`` or ``PersistentClient``).
    """

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        _check_chroma()
        cfg = {**(config or {}), **kwargs}
        self._dim: int = cfg.get("embedding_dim", 384)
        path: str | None = cfg.get("persist_directory")

        if path:
            self._client = chromadb.PersistentClient(path=path)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=cfg.get("collection_name", "neuralmem"),
            metadata={"hnsw:space": "cosine"},
        )

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
                memory.expires_at.isoformat() if memory.expires_at else ""
            ),
        }
        self._collection.upsert(
            ids=[memory.id],
            documents=[memory.content],
            embeddings=[embedding],
            metadatas=[metadata],
        )
        return memory.id

    # ------------------------------------------------------------------
    # get / list
    # ------------------------------------------------------------------

    def get_memory(self, memory_id: str) -> Memory | None:
        result = self._collection.get(ids=[memory_id], include=["embeddings"])
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        emb = result["embeddings"][0] if result["embeddings"] else None
        return _memory_from_chroma(meta, memory_id, emb)

    def list_memories(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Memory]:
        where = {"user_id": user_id} if user_id else None
        result = self._collection.get(
            where=where,  # type: ignore[arg-type]
            include=["embeddings"],
            limit=limit,
        )
        memories: list[Memory] = []
        for i, mid in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            emb = (
                result["embeddings"][i]
                if result["embeddings"]
                else None
            )
            memories.append(_memory_from_chroma(meta, mid, emb))
        return memories

    # ------------------------------------------------------------------
    # update / delete
    # ------------------------------------------------------------------

    def update_memory(self, memory_id: str, **kwargs: object) -> None:
        existing = self._collection.get(ids=[memory_id], include=["embeddings"])
        if not existing["ids"]:
            raise StorageError(f"Memory {memory_id} not found")

        meta = dict(existing["metadatas"][0])
        for key, value in kwargs.items():
            if key == "tags":
                meta["tags"] = json.dumps(list(value))  # type: ignore[arg-type]
            elif key == "entity_ids":
                meta["entity_ids"] = json.dumps(
                    list(value)  # type: ignore[arg-type]
                )
            elif key == "supersedes":
                meta["supersedes"] = json.dumps(
                    list(value)  # type: ignore[arg-type]
                )
            elif key == "is_active":
                meta["is_active"] = int(bool(value))
            elif key == "memory_type" and isinstance(value, MemoryType):
                meta["memory_type"] = value.value
            elif key == "scope" and isinstance(value, MemoryScope):
                meta["scope"] = value.value
            elif key == "expires_at":
                meta["expires_at"] = (
                    value.isoformat()
                    if isinstance(value, datetime)
                    else str(value or "")
                )
            else:
                meta[key] = value

        meta["updated_at"] = datetime.now(timezone.utc).isoformat()

        emb = (
            list(existing["embeddings"][0])
            if existing["embeddings"]
            else None
        )
        if "embedding" in kwargs and kwargs["embedding"] is not None:
            emb = list(kwargs["embedding"])  # type: ignore[arg-type]

        self._collection.update(
            ids=[memory_id],
            documents=[meta.get("content", "")],
            embeddings=[emb] if emb else None,
            metadatas=[meta],
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
            self._collection.delete(ids=[memory_id])
            return 1

        # Fetch all and filter
        all_data = self._collection.get(include=["metadatas"])
        ids_to_delete: list[str] = []
        for i, mid in enumerate(all_data["ids"]):
            meta = all_data["metadatas"][i]
            if user_id and meta.get("user_id") != user_id:
                continue
            if before is not None:
                created = meta.get("created_at", "")
                before_str = (
                    before.isoformat()
                    if isinstance(before, datetime)
                    else str(before)
                )
                if created >= before_str:
                    continue
            if tags:
                meta_tags = json.loads(meta.get("tags", "[]"))
                if not any(t in meta_tags for t in tags):
                    continue
            if max_importance is not None:
                if meta.get("importance", 0.0) >= max_importance:
                    continue
            ids_to_delete.append(mid)

        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
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
        where = _build_where(user_id, memory_types)
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=limit,
            where=where,  # type: ignore[arg-type]
            include=["distances"],
        )
        if not result["ids"] or not result["distances"]:
            return []
        pairs: list[tuple[str, float]] = []
        for i, mid in enumerate(result["ids"][0]):
            dist = result["distances"][0][i]
            score = max(0.0, 1.0 - dist)
            pairs.append((mid, score))
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
        where = _build_where(user_id, memory_types)
        result = self._collection.query(
            query_texts=[query],
            n_results=limit,
            where=where,  # type: ignore[arg-type]
            include=["distances"],
        )
        if not result["ids"] or not result["distances"]:
            return []
        pairs: list[tuple[str, float]] = []
        for i, mid in enumerate(result["ids"][0]):
            dist = result["distances"][0][i]
            score = max(0.0, 1.0 - dist)
            pairs.append((mid, score))
        return pairs

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
        all_mems = self.list_memories(user_id=user_id)
        mem_map = {m.id: m for m in all_mems}

        scored: list[tuple[str, float]] = []
        for mid, sim_score in results:
            mem = mem_map.get(mid)
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
        similar_ids = [mid for mid, score in results if score >= threshold]
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
        existing = self._collection.get(ids=[memory_id])
        if not existing["ids"]:
            return
        meta = dict(existing["metadatas"][0])
        meta["access_count"] = meta.get("access_count", 0) + 1
        meta["last_accessed"] = datetime.now(timezone.utc).isoformat()
        self._collection.update(
            ids=[memory_id],
            documents=[meta.get("content", "")],
            metadatas=[meta],
        )

    def batch_record_access(self, memory_ids: list[str]) -> None:
        for mid in memory_ids:
            self.record_access(mid)

    # ------------------------------------------------------------------
    # history
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
            "Chroma backend: history saved for %s (event=%s)",
            memory_id,
            event,
        )

    def get_history(self, memory_id: str) -> list[dict]:
        return []

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def get_stats(self, user_id: str | None = None) -> dict[str, object]:
        all_data = self._collection.get(include=["metadatas"])
        count = len(all_data["ids"])
        return {
            "backend": "chroma",
            "total_memories": count,
            "user_filter": user_id or "all",
        }

    # ------------------------------------------------------------------
    # graph snapshots (no-op)
    # ------------------------------------------------------------------

    def load_graph_snapshot(self) -> dict | None:
        return None

    def save_graph_snapshot(self, data: dict) -> None:
        _logger.debug("Chroma backend: graph snapshot saved (no-op)")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _memory_from_chroma(
    meta: dict[str, Any],
    mid: str,
    embedding: list[float] | None,
) -> Memory:
    """Build a Memory from Chroma metadata."""
    return Memory(
        id=mid,
        content=meta.get("content", ""),
        memory_type=MemoryType(meta.get("memory_type", "semantic")),
        scope=MemoryScope(meta.get("scope", "user")),
        user_id=meta.get("user_id") or None,
        agent_id=meta.get("agent_id") or None,
        session_id=meta.get("session_id") or None,
        tags=tuple(json.loads(meta.get("tags", "[]"))),
        source=meta.get("source") or None,
        importance=meta.get("importance", 0.5),
        entity_ids=tuple(json.loads(meta.get("entity_ids", "[]"))),
        is_active=bool(meta.get("is_active", 1)),
        superseded_by=meta.get("superseded_by") or None,
        supersedes=tuple(json.loads(meta.get("supersedes", "[]"))),
        created_at=datetime.fromisoformat(meta["created_at"]),
        updated_at=datetime.fromisoformat(meta["updated_at"]),
        last_accessed=datetime.fromisoformat(meta["last_accessed"]),
        access_count=meta.get("access_count", 0),
        embedding=embedding,
        expires_at=(
            datetime.fromisoformat(meta["expires_at"])
            if meta.get("expires_at")
            else None
        ),
    )


def _build_where(
    user_id: str | None,
    memory_types: list[MemoryType] | None,
) -> dict[str, Any] | None:
    """Build a Chroma ``where`` clause."""
    conditions: list[dict[str, Any]] = []
    if user_id:
        conditions.append({"user_id": user_id})
    if memory_types:
        types = [mt.value for mt in memory_types]
        if len(types) == 1:
            conditions.append({"memory_type": types[0]})
        else:
            conditions.append({"memory_type": {"$in": types}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
