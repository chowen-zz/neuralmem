"""Weaviate vector store backend."""
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
    import weaviate  # type: ignore[import-untyped]
except ImportError:
    weaviate = None  # type: ignore[assignment]


def _check_weaviate() -> None:
    if weaviate is None:
        raise ImportError(
            "WeaviateVectorStore requires the weaviate-client "
            "package. Install with: pip install "
            "neuralmem[weaviate]"
        )


class WeaviateVectorStore(StorageBackend):
    """Weaviate-backed vector store.

    Uses schema-less storage with auto-class creation.
    """

    def __init__(
        self, config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        _check_weaviate()
        cfg = {**(config or {}), **kwargs}
        self._dim: int = cfg.get("embedding_dim", 384)
        self._class_name: str = cfg.get(
            "class_name", "NeuralMem"
        )

        url = cfg.get("url", "http://localhost:8080")
        api_key = cfg.get("api_key")

        auth = None
        if api_key:
            auth = weaviate.auth.AuthApiKey(api_key)

        self._client = weaviate.Client(
            url=url, auth_client_secret=auth,
        )
        self._ensure_class()

    def _ensure_class(self) -> None:
        """Ensure the Weaviate class exists."""
        if self._client.schema.exists(self._class_name):
            return
        class_def = {
            "class": self._class_name,
            "vectorizer": "none",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                },
                {
                    "name": "memory_type",
                    "dataType": ["text"],
                },
                {
                    "name": "scope",
                    "dataType": ["text"],
                },
                {
                    "name": "user_id",
                    "dataType": ["text"],
                },
                {
                    "name": "agent_id",
                    "dataType": ["text"],
                },
                {
                    "name": "session_id",
                    "dataType": ["text"],
                },
                {
                    "name": "tags_json",
                    "dataType": ["text"],
                },
                {
                    "name": "source",
                    "dataType": ["text"],
                },
                {
                    "name": "importance",
                    "dataType": ["number"],
                },
                {
                    "name": "entity_ids_json",
                    "dataType": ["text"],
                },
                {
                    "name": "is_active",
                    "dataType": ["boolean"],
                },
                {
                    "name": "superseded_by",
                    "dataType": ["text"],
                },
                {
                    "name": "supersedes_json",
                    "dataType": ["text"],
                },
                {
                    "name": "created_at",
                    "dataType": ["text"],
                },
                {
                    "name": "updated_at",
                    "dataType": ["text"],
                },
                {
                    "name": "last_accessed",
                    "dataType": ["text"],
                },
                {
                    "name": "access_count",
                    "dataType": ["int"],
                },
                {
                    "name": "expires_at",
                    "dataType": ["text"],
                },
            ],
        }
        self._client.schema.create_class(class_def)

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save_memory(self, memory: Memory) -> str:
        embedding = memory.embedding or [0.0] * self._dim
        properties: dict[str, Any] = {
            "content": memory.content,
            "memory_type": memory.memory_type.value,
            "scope": memory.scope.value,
            "user_id": memory.user_id or "",
            "agent_id": memory.agent_id or "",
            "session_id": memory.session_id or "",
            "tags_json": json.dumps(list(memory.tags)),
            "source": memory.source or "",
            "importance": float(memory.importance),
            "entity_ids_json": json.dumps(
                list(memory.entity_ids)
            ),
            "is_active": memory.is_active,
            "superseded_by": memory.superseded_by or "",
            "supersedes_json": json.dumps(
                list(memory.supersedes)
            ),
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
        self._client.data_object.create(
            class_name=self._class_name,
            data_object=properties,
            uuid=memory.id,
            vector=embedding,
        )
        return memory.id

    # ------------------------------------------------------------------
    # get / list
    # ------------------------------------------------------------------

    def get_memory(self, memory_id: str) -> Memory | None:
        try:
            obj = self._client.data_object.get_by_id(
                class_name=self._class_name,
                uuid=memory_id,
                with_vector=True,
            )
        except Exception:
            return None
        if obj is None:
            return None
        return _memory_from_weaviate(obj, memory_id)

    def list_memories(
        self, user_id: str | None = None,
        limit: int = 10_000,
    ) -> list[Memory]:
        query = (
            self.client_query()
            .with_additional(["id", "vector"])
            .with_limit(limit)
        )
        if user_id:
            where_filter = {
                "path": ["user_id"],
                "operator": "Equal",
                "valueText": user_id,
            }
            query = query.with_where(where_filter)

        result = query.do()
        objects = (
            result.get("data", {})
            .get("Get", {})
            .get(self._class_name, [])
        )
        memories: list[Memory] = []
        for obj in objects:
            obj_id = obj.get("_additional", {}).get("id", "")
            memories.append(_memory_from_weaviate(obj, obj_id))
        return memories

    def client_query(self) -> Any:
        """Return a class-level GraphQL query builder."""
        return self._client.query.get(
            self._class_name,
            [
                "content", "memory_type", "scope",
                "user_id", "agent_id", "session_id",
                "tags_json", "source", "importance",
                "entity_ids_json", "is_active",
                "superseded_by", "supersedes_json",
                "created_at", "updated_at",
                "last_accessed", "access_count",
                "expires_at",
            ],
        )

    # ------------------------------------------------------------------
    # update / delete
    # ------------------------------------------------------------------

    def update_memory(
        self, memory_id: str, **kwargs: object
    ) -> None:
        try:
            obj = self._client.data_object.get_by_id(
                class_name=self._class_name,
                uuid=memory_id,
                with_vector=True,
            )
        except Exception:
            obj = None
        if obj is None:
            raise StorageError(
                f"Memory {memory_id} not found"
            )

        properties = dict(obj.get("properties", {}))
        for key, value in kwargs.items():
            if key == "tags":
                properties["tags_json"] = json.dumps(
                    list(value)  # type: ignore[arg-type]
                )
            elif key == "entity_ids":
                properties["entity_ids_json"] = json.dumps(
                    list(value)  # type: ignore[arg-type]
                )
            elif key == "supersedes":
                properties["supersedes_json"] = json.dumps(
                    list(value)  # type: ignore[arg-type]
                )
            elif key == "is_active":
                properties["is_active"] = bool(value)
            elif key == "memory_type" and isinstance(
                value, MemoryType
            ):
                properties["memory_type"] = value.value
            elif key == "scope" and isinstance(
                value, MemoryScope
            ):
                properties["scope"] = value.value
            elif key == "expires_at":
                properties["expires_at"] = (
                    value.isoformat()
                    if isinstance(value, datetime)
                    else str(value or "")
                )
            else:
                properties[key] = value

        properties["updated_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

        vector = obj.get("vector")
        if "embedding" in kwargs and kwargs["embedding"] is not None:
            vector = list(
                kwargs["embedding"]  # type: ignore[arg-type]
            )

        self._client.data_object.update(
            class_name=self._class_name,
            uuid=memory_id,
            data_object=properties,
            vector=vector,
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
            self._client.data_object.delete(
                class_name=self._class_name,
                uuid=memory_id,
            )
            return 1

        # Fetch all and filter
        result = (
            self._client.query.get(
                self._class_name,
                [
                    "content", "memory_type", "scope",
                    "user_id", "agent_id", "session_id",
                    "tags_json", "source", "importance",
                    "entity_ids_json", "is_active",
                    "superseded_by", "supersedes_json",
                    "created_at", "updated_at",
                    "last_accessed", "access_count",
                    "expires_at",
                ],
            )
            .with_additional(["id"])
            .with_limit(10_000)
            .do()
        )
        objects = (
            result.get("data", {})
            .get("Get", {})
            .get(self._class_name, [])
        )

        ids_to_delete: list[str] = []
        for obj in objects:
            obj_id = obj.get("_additional", {}).get("id", "")
            properties = obj
            if user_id and properties.get("user_id") != user_id:
                continue
            if before is not None:
                created = properties.get("created_at", "")
                before_str = (
                    before.isoformat()
                    if isinstance(before, datetime)
                    else str(before)
                )
                if created >= before_str:
                    continue
            if tags:
                meta_tags = json.loads(
                    properties.get("tags_json", "[]")
                )
                if not any(t in meta_tags for t in tags):
                    continue
            if max_importance is not None:
                if properties.get("importance", 0.0) >= max_importance:
                    continue
            ids_to_delete.append(obj_id)

        for obj_id in ids_to_delete:
            self._client.data_object.delete(
                class_name=self._class_name,
                uuid=obj_id,
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
        query = (
            self._client.query.get(
                self._class_name,
                ["content", "memory_type"],
            )
            .with_near_vector({"vector": vector})
            .with_additional(["id", "distance"])
            .with_limit(limit)
        )

        where_filter = _build_weaviate_filter(
            user_id, memory_types
        )
        if where_filter:
            query = query.with_where(where_filter)

        result = query.do()
        objects = (
            result.get("data", {})
            .get("Get", {})
            .get(self._class_name, [])
        )
        pairs: list[tuple[str, float]] = []
        for obj in objects:
            obj_id = obj.get("_additional", {}).get("id", "")
            dist = float(
                obj.get("_additional", {}).get(
                    "distance", 1.0
                )
            )
            score = max(0.0, 1.0 - dist)
            pairs.append((obj_id, score))
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
        # Use BM25-like approach: fetch all and filter
        all_mems = self.list_memories(user_id=user_id)
        query_lower = query.lower()
        pairs: list[tuple[str, float]] = []
        for mem in all_mems:
            if memory_types:
                if mem.memory_type not in memory_types:
                    continue
            if query_lower in mem.content.lower():
                pairs.append((mem.id, 1.0))
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
        try:
            obj = self._client.data_object.get_by_id(
                class_name=self._class_name,
                uuid=memory_id,
                with_vector=True,
            )
        except Exception:
            return
        if obj is None:
            return

        properties = dict(obj.get("properties", {}))
        properties["access_count"] = (
            properties.get("access_count", 0) + 1
        )
        properties["last_accessed"] = (
            datetime.now(timezone.utc).isoformat()
        )

        vector = obj.get("vector")
        self._client.data_object.update(
            class_name=self._class_name,
            uuid=memory_id,
            data_object=properties,
            vector=vector,
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
            "Weaviate backend: history saved for %s "
            "(event=%s)",
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
        result = self._client.query.aggregate(
            self._class_name
        ).with_meta_count().do()
        count = (
            result.get("data", {})
            .get("Aggregate", {})
            .get(self._class_name, [{}])[0]
            .get("meta", {})
            .get("count", 0)
        )
        return {
            "backend": "weaviate",
            "total_memories": count,
            "user_filter": user_id or "all",
        }

    # ------------------------------------------------------------------
    # graph snapshots (no-op)
    # ------------------------------------------------------------------

    def load_graph_snapshot(self) -> dict | None:
        return None

    def save_graph_snapshot(self, data: dict) -> None:
        _logger.debug(
            "Weaviate backend: graph snapshot saved (no-op)"
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _memory_from_weaviate(
    obj: dict[str, Any], mid: str
) -> Memory:
    """Build a Memory from a Weaviate object."""
    props = obj.get("properties", obj)
    additional = obj.get("_additional", {})
    vector = additional.get("vector")
    return Memory(
        id=mid,
        content=props.get("content", ""),
        memory_type=MemoryType(
            props.get("memory_type", "semantic")
        ),
        scope=MemoryScope(props.get("scope", "user")),
        user_id=props.get("user_id") or None,
        agent_id=props.get("agent_id") or None,
        session_id=props.get("session_id") or None,
        tags=tuple(
            json.loads(props.get("tags_json", "[]"))
        ),
        source=props.get("source") or None,
        importance=props.get("importance", 0.5),
        entity_ids=tuple(
            json.loads(
                props.get("entity_ids_json", "[]")
            )
        ),
        is_active=bool(props.get("is_active", True)),
        superseded_by=props.get("superseded_by") or None,
        supersedes=tuple(
            json.loads(
                props.get("supersedes_json", "[]")
            )
        ),
        created_at=datetime.fromisoformat(
            props["created_at"]
        ),
        updated_at=datetime.fromisoformat(
            props["updated_at"]
        ),
        last_accessed=datetime.fromisoformat(
            props["last_accessed"]
        ),
        access_count=props.get("access_count", 0),
        embedding=list(vector) if vector else None,
        expires_at=(
            datetime.fromisoformat(props["expires_at"])
            if props.get("expires_at")
            else None
        ),
    )


def _build_weaviate_filter(
    user_id: str | None,
    memory_types: list[MemoryType] | None,
) -> dict[str, Any] | None:
    """Build a Weaviate where filter."""
    conditions: list[dict[str, Any]] = []
    if user_id:
        conditions.append({
            "path": ["user_id"],
            "operator": "Equal",
            "valueText": user_id,
        })
    if memory_types:
        types = [mt.value for mt in memory_types]
        if len(types) == 1:
            conditions.append({
                "path": ["memory_type"],
                "operator": "Equal",
                "valueText": types[0],
            })
        else:
            conditions.append({
                "path": ["memory_type"],
                "operator": "ContainsAny",
                "valueTextArray": types,
            })
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {
        "operator": "And",
        "operands": conditions,
    }
