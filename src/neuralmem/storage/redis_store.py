"""Redis vector store backend."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from neuralmem.core.exceptions import StorageError
from neuralmem.core.types import Memory, MemoryScope, MemoryType
from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)

try:
    import redis  # type: ignore[import-untyped]
    from redis.commands.search.field import (  # type: ignore[import-untyped]
        NumericField,
        TagField,
        TextField,
        VectorField,
    )
    from redis.commands.search.index_definition import (  # type: ignore[import-untyped]
        IndexDefinition,
    )
    from redis.commands.search.query import Query  # type: ignore[import-untyped]
except ImportError:
    redis = None  # type: ignore[assignment,misc]


def _check_redis() -> None:
    if redis is None:
        raise ImportError(
            "RedisVectorStore requires the redis package. "
            "Install with: pip install neuralmem[redis]"
        )


class RedisVectorStore(StorageBackend):
    """Redis-backed vector store using RediSearch module.

    Stores memories as Redis hashes with a vector index for ANN
    search via ``FT.SEARCH``.
    """

    def __init__(
        self, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        _check_redis()
        cfg = {**(config or {}), **kwargs}
        self._dim: int = cfg.get("embedding_dim", 384)
        self._prefix: str = cfg.get("key_prefix", "nm:")
        self._index_name: str = cfg.get("index_name", "idx:neuralmem")

        url = cfg.get("url", "redis://localhost:6379/0")
        self._client: Any = redis.Redis.from_url(url)

        self._ensure_index()

    # ------------------------------------------------------------------
    # index management
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """Create the RediSearch index if it doesn't already exist."""
        try:
            self._client.ft(self._index_name).info()
        except Exception:
            # Index does not exist — create it
            schema = [
                TextField("$.content", as_name="content"),
                TagField(
                    "$.memory_type", as_name="memory_type"
                ),
                TagField("$.scope", as_name="scope"),
                TagField("$.user_id", as_name="user_id"),
                TagField("$.agent_id", as_name="agent_id"),
                TagField("$.session_id", as_name="session_id"),
                NumericField("$.importance", as_name="importance"),
                NumericField(
                    "$.access_count", as_name="access_count"
                ),
                NumericField(
                    "$.is_active", as_name="is_active"
                ),
                TextField("$.created_at", as_name="created_at"),
                TextField("$.updated_at", as_name="updated_at"),
                VectorField(
                    "$.embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self._dim,
                        "DISTANCE_METRIC": "COSINE",
                    },
                    as_name="embedding",
                ),
            ]
            self._client.ft(self._index_name).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self._prefix], index_type="JSON"
                ),
            )

    def _key(self, memory_id: str) -> str:
        return f"{self._prefix}{memory_id}"

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save_memory(self, memory: Memory) -> str:
        embedding = memory.embedding or [0.0] * self._dim
        import numpy as np

        emb_bytes = np.array(embedding, dtype=np.float32).tobytes()

        doc: dict[str, Any] = {
            "id": memory.id,
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
            "is_active": int(memory.is_active),
            "superseded_by": memory.superseded_by or "",
            "supersedes": list(memory.supersedes),
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "last_accessed": memory.last_accessed.isoformat(),
            "access_count": memory.access_count,
            "expires_at": (
                memory.expires_at.isoformat()
                if memory.expires_at
                else ""
            ),
            "embedding": emb_bytes,
        }
        self._client.json().set(self._key(memory.id), "$", doc)
        return memory.id

    # ------------------------------------------------------------------
    # get / list
    # ------------------------------------------------------------------

    def get_memory(self, memory_id: str) -> Memory | None:
        data = self._client.json().get(self._key(memory_id))
        if data is None:
            return None
        return _memory_from_redis(data)

    def list_memories(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Memory]:
        q = "*"
        if user_id:
            q = f"@user_id:{{{user_id}}}"
        query = Query(q).paging(0, limit).no_content()
        try:
            result = self._client.ft(self._index_name).search(
                query
            )
        except Exception:
            return []
        memories: list[Memory] = []
        for doc in result.docs:
            data = self._client.json().get(doc.id)
            if data:
                memories.append(_memory_from_redis(data))
        return memories[:limit]

    # ------------------------------------------------------------------
    # update / delete
    # ------------------------------------------------------------------

    def update_memory(self, memory_id: str, **kwargs: object) -> None:
        key = self._key(memory_id)
        data = self._client.json().get(key)
        if data is None:
            raise StorageError(f"Memory {memory_id} not found")

        for k, v in kwargs.items():
            if k == "tags":
                self._client.json().set(
                    key, "$.tags", list(v)  # type: ignore[arg-type]
                )
            elif k == "entity_ids":
                self._client.json().set(
                    key,
                    "$.entity_ids",
                    list(v),  # type: ignore[arg-type]
                )
            elif k == "supersedes":
                self._client.json().set(
                    key,
                    "$.supersedes",
                    list(v),  # type: ignore[arg-type]
                )
            elif k == "is_active":
                self._client.json().set(
                    key, "$.is_active", int(bool(v))
                )
            elif k == "memory_type" and isinstance(v, MemoryType):
                self._client.json().set(
                    key, "$.memory_type", v.value
                )
            elif k == "scope" and isinstance(v, MemoryScope):
                self._client.json().set(key, "$.scope", v.value)
            elif k == "expires_at":
                val = (
                    v.isoformat()
                    if isinstance(v, datetime)
                    else str(v or "")
                )
                self._client.json().set(key, "$.expires_at", val)
            elif k == "embedding":
                import numpy as np

                if v is not None:
                    emb_bytes = np.array(
                        v,  # type: ignore[arg-type]
                        dtype=np.float32,
                    ).tobytes()
                    self._client.json().set(
                        key, "$.embedding", emb_bytes
                    )
            else:
                self._client.json().set(key, f"$.{k}", v)

        self._client.json().set(
            key,
            "$.updated_at",
            datetime.now(timezone.utc).isoformat(),
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
            self._client.delete(self._key(memory_id))
            return 1

        # Fetch all matching keys
        keys = list(
            self._client.scan_iter(match=f"{self._prefix}*", count=500)
        )
        deleted = 0
        for raw_key in keys:
            key = (
                raw_key.decode()
                if isinstance(raw_key, bytes)
                else raw_key
            )
            data = self._client.json().get(key)
            if data is None:
                continue
            if user_id and data.get("user_id") != user_id:
                continue
            if before is not None:
                created = data.get("created_at", "")
                before_str = (
                    before.isoformat()
                    if isinstance(before, datetime)
                    else str(before)
                )
                if created >= before_str:
                    continue
            if tags:
                mtags = data.get("tags", [])
                if not any(t in mtags for t in tags):
                    continue
            if max_importance is not None:
                if data.get("importance", 0.0) >= max_importance:
                    continue
            self._client.delete(key)
            deleted += 1
        return deleted

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
        import numpy as np

        emb_bytes = np.array(vector, dtype=np.float32).tobytes()
        filter_parts: list[str] = []
        if user_id:
            filter_parts.append(f"@user_id:{{{user_id}}}")
        if memory_types:
            type_str = "|".join(mt.value for mt in memory_types)
            filter_parts.append(f"@memory_type:{{{type_str}}}")
        filter_str = " ".join(filter_parts) if filter_parts else ""

        base = (
            f"({filter_str})=>[KNN {limit} "
            f"@embedding $vec AS score]"
        )
        query = (
            Query(base)
            .sort_by("score")
            .return_fields("id", "score")
            .dialect(2)
        )
        try:
            result = self._client.ft(self._index_name).search(
                query, query_params={"vec": emb_bytes}
            )
        except Exception:
            return []
        pairs: list[tuple[str, float]] = []
        for doc in result.docs:
            mid = doc.id.replace(self._prefix, "", 1)
            # Redis cosine distance → similarity
            dist = float(getattr(doc, "score", 1.0))
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
        filter_parts: list[str] = [f"@content:{query}"]
        if user_id:
            filter_parts.append(f"@user_id:{{{user_id}}}")
        if memory_types:
            type_str = "|".join(mt.value for mt in memory_types)
            filter_parts.append(f"@memory_type:{{{type_str}}}")
        q = Query(" ".join(filter_parts)).paging(0, limit)
        try:
            result = self._client.ft(self._index_name).search(q)
        except Exception:
            return []
        pairs: list[tuple[str, float]] = []
        for doc in result.docs:
            mid = doc.id.replace(self._prefix, "", 1)
            pairs.append((mid, 1.0))
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
        raw = self.vector_search(
            vector=vector, user_id=user_id, limit=limit * 2
        )
        now = datetime.now(timezone.utc)
        scored: list[tuple[str, float]] = []
        for mid, sim in raw:
            mem = self.get_memory(mid)
            if mem is None:
                continue
            if time_range:
                start, end = time_range
                s = (
                    start
                    if isinstance(start, datetime)
                    else datetime.fromisoformat(str(start))
                )
                e = (
                    end
                    if isinstance(end, datetime)
                    else datetime.fromisoformat(str(end))
                )
                if not (s <= mem.created_at <= e):
                    continue
            age_h = (
                (now - mem.last_accessed).total_seconds() / 3600.0
            )
            rec = 1.0 / (1.0 + age_h / 24.0)
            combined = (
                (1.0 - recency_weight) * sim
                + recency_weight * rec
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
        raw = self.vector_search(
            vector=vector, user_id=user_id, limit=50
        )
        similar: list[Memory] = []
        for mid, score in raw:
            if score >= threshold:
                mem = self.get_memory(mid)
                if mem is not None:
                    similar.append(mem)
        return similar

    # ------------------------------------------------------------------
    # access tracking
    # ------------------------------------------------------------------

    def record_access(self, memory_id: str) -> None:
        key = self._key(memory_id)
        data = self._client.json().get(key)
        if data is None:
            return
        self._client.json().set(
            key,
            "$.access_count",
            data.get("access_count", 0) + 1,
        )
        self._client.json().set(
            key,
            "$.last_accessed",
            datetime.now(timezone.utc).isoformat(),
        )

    def batch_record_access(self, memory_ids: list[str]) -> None:
        for mid in memory_ids:
            self.record_access(mid)

    # ------------------------------------------------------------------
    # history (no-op — relies on external persistence)
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
            "Redis backend: history saved for %s (event=%s)",
            memory_id, event,
        )

    def get_history(self, memory_id: str) -> list[dict]:
        return []

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def get_stats(self, user_id: str | None = None) -> dict[str, object]:
        try:
            info = self._client.ft(self._index_name).info()
            total = info.get("num_docs", 0)
        except Exception:
            total = 0
        return {
            "backend": "redis",
            "total_memories": total,
            "user_filter": user_id or "all",
        }

    # ------------------------------------------------------------------
    # graph snapshots
    # ------------------------------------------------------------------

    def load_graph_snapshot(self) -> dict | None:
        data = self._client.json().get(f"{self._prefix}graph")
        return data

    def save_graph_snapshot(self, data: dict) -> None:
        self._client.json().set(f"{self._prefix}graph", "$", data)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _memory_from_redis(data: dict[str, Any]) -> Memory:
    """Build a Memory from a Redis JSON document."""
    import numpy as np

    emb_raw = data.get("embedding")
    embedding: list[float] | None = None
    if emb_raw and isinstance(emb_raw, (bytes, bytearray)):
        embedding = list(np.frombuffer(emb_raw, dtype=np.float32))
    elif isinstance(emb_raw, list):
        embedding = emb_raw

    return Memory(
        id=data.get("id", ""),
        content=data.get("content", ""),
        memory_type=MemoryType(
            data.get("memory_type", "semantic")
        ),
        scope=MemoryScope(data.get("scope", "user")),
        user_id=data.get("user_id") or None,
        agent_id=data.get("agent_id") or None,
        session_id=data.get("session_id") or None,
        tags=tuple(data.get("tags", [])),
        source=data.get("source") or None,
        importance=data.get("importance", 0.5),
        entity_ids=tuple(data.get("entity_ids", [])),
        is_active=bool(data.get("is_active", True)),
        superseded_by=data.get("superseded_by") or None,
        supersedes=tuple(data.get("supersedes", [])),
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]),
        last_accessed=datetime.fromisoformat(
            data["last_accessed"]
        ),
        access_count=data.get("access_count", 0),
        embedding=embedding,
        expires_at=(
            datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None
        ),
    )
