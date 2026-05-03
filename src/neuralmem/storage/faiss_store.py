"""FAISS vector store backend."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from neuralmem.core.exceptions import StorageError
from neuralmem.core.types import Memory, MemoryScope, MemoryType
from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import-untyped]
except ImportError:
    faiss = None  # type: ignore[assignment]


def _check_faiss() -> None:
    if faiss is None:
        raise ImportError(
            "FAISSVectorStore requires the faiss-cpu package. "
            "Install with: pip install neuralmem[faiss]"
        )


class FAISSVectorStore(StorageBackend):
    """FAISS-backed vector store.

    Stores vectors in a FAISS index and metadata in a plain dict.
    """

    def __init__(
        self, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        _check_faiss()
        cfg = {**(config or {}), **kwargs}
        self._dim: int = cfg.get("embedding_dim", 384)

        # FAISS index — flat L2 (cosine via normalisation)
        self._index = faiss.IndexFlatIP(self._dim)

        # id -> Memory  and  id -> faiss-ordinal mappings
        self._memories: dict[str, Memory] = {}
        self._id_list: list[str] = []  # ordinal -> memory_id
        self._history: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _normalise(self, vec: list[float]) -> np.ndarray:
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.reshape(1, -1)

    def _rebuild_index(self) -> None:
        self._index.reset()
        self._id_list.clear()
        if not self._memories:
            return
        vectors: list[np.ndarray] = []
        for mid, mem in self._memories.items():
            emb = mem.embedding or [0.0] * self._dim
            vectors.append(self._normalise(emb))
            self._id_list.append(mid)
        if vectors:
            matrix = np.vstack(vectors)
            self._index.add(matrix)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save_memory(self, memory: Memory) -> str:
        self._memories[memory.id] = memory
        emb = memory.embedding or [0.0] * self._dim
        vec = self._normalise(emb)
        self._index.add(vec)  # type: ignore[arg-type]
        self._id_list.append(memory.id)
        return memory.id

    # ------------------------------------------------------------------
    # get / list
    # ------------------------------------------------------------------

    def get_memory(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    def list_memories(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Memory]:
        results: list[Memory] = []
        for mem in self._memories.values():
            if user_id and mem.user_id != user_id:
                continue
            results.append(mem)
            if len(results) >= limit:
                break
        return results

    # ------------------------------------------------------------------
    # update / delete
    # ------------------------------------------------------------------

    def update_memory(self, memory_id: str, **kwargs: object) -> None:
        if memory_id not in self._memories:
            raise StorageError(f"Memory {memory_id} not found")
        mem = self._memories[memory_id]
        # Reconstruct with updated fields
        update_dict: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key == "tags":
                update_dict["tags"] = tuple(value)  # type: ignore[arg-type]
            elif key == "entity_ids":
                update_dict["entity_ids"] = tuple(
                    value  # type: ignore[arg-type]
                )
            elif key == "supersedes":
                update_dict["supersedes"] = tuple(
                    value  # type: ignore[arg-type]
                )
            elif key == "is_active":
                update_dict["is_active"] = bool(value)
            elif key == "memory_type" and isinstance(
                value, MemoryType
            ):
                update_dict["memory_type"] = value
            elif key == "scope" and isinstance(value, MemoryScope):
                update_dict["scope"] = value
            elif key == "expires_at":
                update_dict["expires_at"] = (
                    value
                    if isinstance(value, datetime)
                    else (
                        datetime.fromisoformat(str(value))
                        if value
                        else None
                    )
                )
            elif key == "embedding":
                update_dict["embedding"] = (
                    list(value)  # type: ignore[arg-type]
                    if value is not None
                    else None
                )
            else:
                update_dict[key] = value

        update_dict["updated_at"] = datetime.now(timezone.utc)
        self._memories[memory_id] = mem.model_copy(update=update_dict)

        # Rebuild the FAISS index (simplest correct approach)
        self._rebuild_index()

    def delete_memories(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        before: object = None,
        tags: list[str] | None = None,
        max_importance: float | None = None,
    ) -> int:
        ids_to_delete: list[str] = []

        if memory_id is not None:
            if memory_id in self._memories:
                ids_to_delete.append(memory_id)
        else:
            for mid, mem in self._memories.items():
                if user_id and mem.user_id != user_id:
                    continue
                if before is not None:
                    before_dt = (
                        before
                        if isinstance(before, datetime)
                        else datetime.fromisoformat(str(before))
                    )
                    if mem.created_at >= before_dt:
                        continue
                if tags:
                    if not any(t in mem.tags for t in tags):
                        continue
                if max_importance is not None:
                    if mem.importance >= max_importance:
                        continue
                ids_to_delete.append(mid)

        for mid in ids_to_delete:
            del self._memories[mid]

        if ids_to_delete:
            self._rebuild_index()

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
        if self._index.ntotal == 0:
            return []
        vec = self._normalise(vector)
        k = min(limit, self._index.ntotal)
        scores, indices = self._index.search(vec, k)  # type: ignore[arg-type]

        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_list):
                continue
            mid = self._id_list[idx]
            mem = self._memories.get(mid)
            if mem is None:
                continue
            if user_id and mem.user_id != user_id:
                continue
            if memory_types and mem.memory_type not in memory_types:
                continue
            # IP with normalised vectors → cosine similarity
            results.append((mid, float(max(0.0, score))))
        return results

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
        results: list[tuple[str, float]] = []
        for mid, mem in self._memories.items():
            if user_id and mem.user_id != user_id:
                continue
            if memory_types and mem.memory_type not in memory_types:
                continue
            if query_lower in mem.content.lower():
                results.append((mid, 1.0))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

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
            mem = self._memories.get(mid)
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
                (1.0 - recency_weight) * sim + recency_weight * rec
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
        return [
            self._memories[mid]
            for mid, score in raw
            if score >= threshold and mid in self._memories
        ]

    # ------------------------------------------------------------------
    # access tracking
    # ------------------------------------------------------------------

    def record_access(self, memory_id: str) -> None:
        mem = self._memories.get(memory_id)
        if mem is None:
            return
        self._memories[memory_id] = mem.model_copy(
            update={
                "access_count": mem.access_count + 1,
                "last_accessed": datetime.now(timezone.utc),
            }
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
        self._history.setdefault(memory_id, []).append(
            {
                "memory_id": memory_id,
                "old_content": old_content,
                "new_content": new_content,
                "event": event,
                "changed_at": (
                    datetime.now(timezone.utc).isoformat()
                ),
                "metadata": metadata or {},
            }
        )

    def get_history(self, memory_id: str) -> list[dict]:
        return self._history.get(memory_id, [])

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def get_stats(self, user_id: str | None = None) -> dict[str, object]:
        count = 0
        for mem in self._memories.values():
            if user_id and mem.user_id != user_id:
                continue
            count += 1
        return {
            "backend": "faiss",
            "total_memories": count,
            "faiss_index_size": self._index.ntotal,
            "user_filter": user_id or "all",
        }

    # ------------------------------------------------------------------
    # graph snapshots (in-memory only)
    # ------------------------------------------------------------------

    def load_graph_snapshot(self) -> dict | None:
        return getattr(self, "_graph_snapshot", None)

    def save_graph_snapshot(self, data: dict) -> None:
        self._graph_snapshot = data
