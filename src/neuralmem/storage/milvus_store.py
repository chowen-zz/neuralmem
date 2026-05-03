"""Milvus vector store backend."""
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
    from pymilvus import (  # type: ignore[import-untyped]
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
    )
except ImportError:
    Collection = None  # type: ignore[assignment,misc]
    connections = None  # type: ignore[assignment]


def _check_milvus() -> None:
    if Collection is None:
        raise ImportError(
            "MilvusVectorStore requires the pymilvus package. "
            "Install with: pip install neuralmem[milvus]"
        )


class MilvusVectorStore(StorageBackend):
    """Milvus-backed vector store.

    Uses schema-based storage with auto-created collections.
    """

    def __init__(
        self, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        _check_milvus()
        cfg = {**(config or {}), **kwargs}
        self._dim: int = cfg.get("embedding_dim", 384)
        self._collection_name: str = cfg.get(
            "collection_name", "neuralmem"
        )

        host = cfg.get("host", "localhost")
        port = cfg.get("port", 19530)
        connections.connect(alias="default", host=host, port=port)

        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        """Get or create the Milvus collection."""
        fields = [
            FieldSchema(
                name="id", dtype=DataType.VARCHAR,
                is_primary=True, max_length=128,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._dim,
            ),
            FieldSchema(
                name="content", dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="memory_type", dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name="scope", dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name="user_id", dtype=DataType.VARCHAR,
                max_length=128,
            ),
            FieldSchema(
                name="agent_id", dtype=DataType.VARCHAR,
                max_length=128,
            ),
            FieldSchema(
                name="session_id", dtype=DataType.VARCHAR,
                max_length=128,
            ),
            FieldSchema(
                name="tags", dtype=DataType.VARCHAR,
                max_length=4096,
            ),
            FieldSchema(
                name="source", dtype=DataType.VARCHAR,
                max_length=512,
            ),
            FieldSchema(
                name="importance", dtype=DataType.FLOAT,
            ),
            FieldSchema(
                name="entity_ids", dtype=DataType.VARCHAR,
                max_length=4096,
            ),
            FieldSchema(
                name="is_active", dtype=DataType.INT64,
            ),
            FieldSchema(
                name="superseded_by", dtype=DataType.VARCHAR,
                max_length=128,
            ),
            FieldSchema(
                name="supersedes", dtype=DataType.VARCHAR,
                max_length=4096,
            ),
            FieldSchema(
                name="created_at", dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name="updated_at", dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name="last_accessed", dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name="access_count", dtype=DataType.INT64,
            ),
            FieldSchema(
                name="expires_at", dtype=DataType.VARCHAR,
                max_length=64,
            ),
        ]
        schema = CollectionSchema(fields=fields)
        col = Collection(
            name=self._collection_name, schema=schema,
        )
        # Create IVF_FLAT index for vector search
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        try:
            col.create_index(
                field_name="embedding",
                index_params=index_params,
            )
        except Exception:
            pass  # Index may already exist
        col.load()
        return col

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save_memory(self, memory: Memory) -> str:
        embedding = memory.embedding or [0.0] * self._dim
        data = [
            [memory.id],
            [embedding],
            [memory.content],
            [memory.memory_type.value],
            [memory.scope.value],
            [memory.user_id or ""],
            [memory.agent_id or ""],
            [memory.session_id or ""],
            [json.dumps(list(memory.tags))],
            [memory.source or ""],
            [float(memory.importance)],
            [json.dumps(list(memory.entity_ids))],
            [int(memory.is_active)],
            [memory.superseded_by or ""],
            [json.dumps(list(memory.supersedes))],
            [memory.created_at.isoformat()],
            [memory.updated_at.isoformat()],
            [memory.last_accessed.isoformat()],
            [memory.access_count],
            [
                memory.expires_at.isoformat()
                if memory.expires_at
                else ""
            ],
        ]
        self._collection.insert(data)
        self._collection.flush()
        return memory.id

    # ------------------------------------------------------------------
    # get / list
    # ------------------------------------------------------------------

    def get_memory(self, memory_id: str) -> Memory | None:
        result = self._collection.query(
            expr=f'id == "{memory_id}"',
            output_fields=["*"],
        )
        if not result:
            return None
        return _memory_from_milvus(result[0])

    def list_memories(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Memory]:
        expr = ""
        if user_id:
            expr = f'user_id == "{user_id}"'
        result = self._collection.query(
            expr=expr or "id != \"\"",
            output_fields=["*"],
            limit=limit,
        )
        return [_memory_from_milvus(r) for r in result]

    # ------------------------------------------------------------------
    # update / delete
    # ------------------------------------------------------------------

    def update_memory(
        self, memory_id: str, **kwargs: object
    ) -> None:
        existing = self._collection.query(
            expr=f'id == "{memory_id}"',
            output_fields=["*"],
        )
        if not existing:
            raise StorageError(f"Memory {memory_id} not found")

        row = existing[0]
        metadata: dict[str, Any] = dict(row)
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
            elif key == "scope" and isinstance(
                value, MemoryScope
            ):
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

        embedding = row.get("embedding")
        if "embedding" in kwargs and kwargs["embedding"] is not None:
            embedding = list(
                kwargs["embedding"]  # type: ignore[arg-type]
            )

        data = [
            [memory_id],
            [embedding or [0.0] * self._dim],
            [metadata.get("content", "")],
            [metadata.get("memory_type", "semantic")],
            [metadata.get("scope", "user")],
            [metadata.get("user_id", "")],
            [metadata.get("agent_id", "")],
            [metadata.get("session_id", "")],
            [metadata.get("tags", "[]")],
            [metadata.get("source", "")],
            [float(metadata.get("importance", 0.5))],
            [metadata.get("entity_ids", "[]")],
            [int(metadata.get("is_active", 1))],
            [metadata.get("superseded_by", "")],
            [metadata.get("supersedes", "[]")],
            [metadata.get("created_at", "")],
            [metadata["updated_at"]],
            [metadata.get("last_accessed", "")],
            [int(metadata.get("access_count", 0))],
            [metadata.get("expires_at", "")],
        ]
        # Milvus upsert = insert with same primary key
        self._collection.upsert(data)

    def delete_memories(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        before: object = None,
        tags: list[str] | None = None,
        max_importance: float | None = None,
    ) -> int:
        if memory_id is not None:
            self._collection.delete(
                expr=f'id == "{memory_id}"'
            )
            return 1

        # Fetch all and filter
        result = self._collection.query(
            expr="id != \"\"",
            output_fields=["*"],
            limit=10_000,
        )
        ids_to_delete: list[str] = []
        for row in result:
            if user_id and row.get("user_id") != user_id:
                continue
            if before is not None:
                created = row.get("created_at", "")
                before_str = (
                    before.isoformat()
                    if isinstance(before, datetime)
                    else str(before)
                )
                if created >= before_str:
                    continue
            if tags:
                meta_tags = json.loads(
                    row.get("tags", "[]")
                )
                if not any(t in meta_tags for t in tags):
                    continue
            if max_importance is not None:
                if row.get("importance", 0.0) >= max_importance:
                    continue
            ids_to_delete.append(row["id"])

        if ids_to_delete:
            id_list = ", ".join(
                f'"{i}"' for i in ids_to_delete
            )
            self._collection.delete(
                expr=f"id in [{id_list}]"
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
        expr = _build_milvus_filter(user_id, memory_types)
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        results = self._collection.search(
            data=[vector],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=["id"],
        )
        pairs: list[tuple[str, float]] = []
        if results:
            for hit in results[0]:
                score = max(0.0, hit.score)
                pairs.append((hit.entity.get("id"), score))
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
        expr = ""
        if user_id:
            expr = f'user_id == "{user_id}"'
        result = self._collection.query(
            expr=expr or "id != \"\"",
            output_fields=["*"],
            limit=10_000,
        )
        pairs: list[tuple[str, float]] = []
        for row in result:
            if memory_types:
                mt = row.get("memory_type", "")
                if mt not in [t.value for t in memory_types]:
                    continue
            content = row.get("content", "").lower()
            if query_lower in content:
                pairs.append((row["id"], 1.0))
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
        existing = self._collection.query(
            expr=f'id == "{memory_id}"',
            output_fields=["*"],
        )
        if not existing:
            return
        row = existing[0]
        metadata: dict[str, Any] = dict(row)
        metadata["access_count"] = (
            metadata.get("access_count", 0) + 1
        )
        metadata["last_accessed"] = (
            datetime.now(timezone.utc).isoformat()
        )

        embedding = row.get("embedding") or [
            0.0
        ] * self._dim
        data = [
            [memory_id],
            [embedding],
            [metadata.get("content", "")],
            [metadata.get("memory_type", "semantic")],
            [metadata.get("scope", "user")],
            [metadata.get("user_id", "")],
            [metadata.get("agent_id", "")],
            [metadata.get("session_id", "")],
            [metadata.get("tags", "[]")],
            [metadata.get("source", "")],
            [float(metadata.get("importance", 0.5))],
            [metadata.get("entity_ids", "[]")],
            [int(metadata.get("is_active", 1))],
            [metadata.get("superseded_by", "")],
            [metadata.get("supersedes", "[]")],
            [metadata.get("created_at", "")],
            [metadata.get("updated_at", "")],
            [metadata["last_accessed"]],
            [metadata["access_count"]],
            [metadata.get("expires_at", "")],
        ]
        self._collection.upsert(data)

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
            "Milvus backend: history saved for %s (event=%s)",
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
        self._collection.flush()
        return {
            "backend": "milvus",
            "total_memories": self._collection.num_entities,
            "user_filter": user_id or "all",
        }

    # ------------------------------------------------------------------
    # graph snapshots (no-op)
    # ------------------------------------------------------------------

    def load_graph_snapshot(self) -> dict | None:
        return None

    def save_graph_snapshot(self, data: dict) -> None:
        _logger.debug(
            "Milvus backend: graph snapshot saved (no-op)"
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _memory_from_milvus(row: dict[str, Any]) -> Memory:
    """Build a Memory from a Milvus result row."""
    return Memory(
        id=row.get("id", ""),
        content=row.get("content", ""),
        memory_type=MemoryType(
            row.get("memory_type", "semantic")
        ),
        scope=MemoryScope(row.get("scope", "user")),
        user_id=row.get("user_id") or None,
        agent_id=row.get("agent_id") or None,
        session_id=row.get("session_id") or None,
        tags=tuple(json.loads(row.get("tags", "[]"))),
        source=row.get("source") or None,
        importance=row.get("importance", 0.5),
        entity_ids=tuple(
            json.loads(row.get("entity_ids", "[]"))
        ),
        is_active=bool(row.get("is_active", 1)),
        superseded_by=row.get("superseded_by") or None,
        supersedes=tuple(
            json.loads(row.get("supersedes", "[]"))
        ),
        created_at=datetime.fromisoformat(
            row["created_at"]
        ),
        updated_at=datetime.fromisoformat(
            row["updated_at"]
        ),
        last_accessed=datetime.fromisoformat(
            row["last_accessed"]
        ),
        access_count=row.get("access_count", 0),
        embedding=row.get("embedding"),
        expires_at=(
            datetime.fromisoformat(row["expires_at"])
            if row.get("expires_at")
            else None
        ),
    )


def _build_milvus_filter(
    user_id: str | None,
    memory_types: list[MemoryType] | None,
) -> str | None:
    """Build a Milvus expression filter."""
    conditions: list[str] = []
    if user_id:
        conditions.append(f'user_id == "{user_id}"')
    if memory_types:
        types = [mt.value for mt in memory_types]
        if len(types) == 1:
            conditions.append(
                f'memory_type == "{types[0]}"'
            )
        else:
            type_list = ", ".join(
                f'"{t}"' for t in types
            )
            conditions.append(
                f"memory_type in [{type_list}]"
            )
    if not conditions:
        return None
    return " and ".join(conditions)
