"""PostgreSQL + pgvector storage backend implementing StorageProtocol.

Requires psycopg v3 and the pgvector PostgreSQL extension.
```
pip install "psycopg[binary]"
# In PostgreSQL: CREATE EXTENSION IF NOT EXISTS vector;
```
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import StorageError
from neuralmem.core.types import Memory, MemoryScope, MemoryType
from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)

# ── SQL DDL ───────────────────────────────────────────────────────────────────

_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

_CREATE_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'semantic',
    scope TEXT NOT NULL DEFAULT 'user',
    user_id TEXT,
    agent_id TEXT,
    session_id TEXT,
    tags TEXT DEFAULT '[]',
    source TEXT,
    importance REAL DEFAULT 0.5,
    entity_ids TEXT DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    superseded_by TEXT,
    supersedes TEXT DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    last_accessed TIMESTAMPTZ NOT NULL,
    access_count INTEGER DEFAULT 0,
    embedding vector({dim}),
    session_layer TEXT DEFAULT 'long_term',
    conversation_id TEXT,
    expires_at TIMESTAMPTZ
)
"""

_CREATE_FTS = """
ALTER TABLE memories ADD COLUMN IF NOT EXISTS
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
"""

_CREATE_FTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_memories_content_tsv
    ON memories USING gin(content_tsv)
"""

_CREATE_GRAPH_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS graph_snapshots (
    key TEXT PRIMARY KEY,
    data JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
)
"""

_CREATE_MEMORY_HISTORY = """
CREATE TABLE IF NOT EXISTS memory_history (
    id SERIAL PRIMARY KEY,
    memory_id TEXT NOT NULL,
    old_content TEXT,
    new_content TEXT NOT NULL,
    event TEXT NOT NULL DEFAULT 'UPDATE',
    changed_at TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}'
)
"""

_CREATE_MEMORY_HISTORY_INDEX = """
CREATE INDEX IF NOT EXISTS idx_memory_history_id
    ON memory_history(memory_id)
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _embedding_to_list(embedding: list[float] | None) -> list[float] | None:
    """Return embedding as a plain list (for pgvector insertion)."""
    if embedding is None:
        return None
    return list(embedding)


def _row_to_memory(row: dict[str, Any]) -> Memory:
    """Convert a psycopg row dict to a Memory object."""
    expires_at_raw = row.get("expires_at")
    if isinstance(expires_at_raw, str):
        expires_at = datetime.fromisoformat(expires_at_raw)
    elif isinstance(expires_at_raw, datetime):
        expires_at = expires_at_raw
    else:
        expires_at = None

    # embedding: pgvector returns it as a string '[0.1,0.2,...]' or as list
    emb_raw = row.get("embedding")
    embedding: list[float] | None = None
    if emb_raw is not None:
        if isinstance(emb_raw, str):
            embedding = [float(x) for x in emb_raw.strip("[]").split(",")]
        elif isinstance(emb_raw, (list, tuple)):
            embedding = list(emb_raw)
        else:
            # pgvector type object
            try:
                embedding = list(emb_raw)
            except Exception:
                embedding = None

    def _parse_dt(val: Any) -> datetime:
        if isinstance(val, datetime):
            return val
        return datetime.fromisoformat(str(val))

    return Memory(
        id=row["id"],
        content=row["content"],
        memory_type=MemoryType(row["memory_type"]),
        scope=MemoryScope(row["scope"]),
        user_id=row.get("user_id"),
        agent_id=row.get("agent_id"),
        session_id=row.get("session_id"),
        tags=tuple(json.loads(row.get("tags") or "[]")),
        source=row.get("source"),
        importance=row.get("importance", 0.5),
        entity_ids=tuple(json.loads(row.get("entity_ids") or "[]")),
        is_active=bool(row.get("is_active", True)),
        superseded_by=row.get("superseded_by"),
        supersedes=tuple(json.loads(row.get("supersedes") or "[]")),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
        last_accessed=_parse_dt(row["last_accessed"]),
        access_count=row.get("access_count", 0),
        embedding=embedding,
        expires_at=expires_at,
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Backend ───────────────────────────────────────────────────────────────────

class PgVectorStorage(StorageBackend):
    """PostgreSQL + pgvector storage backend.

    Uses the pgvector extension for efficient vector similarity search and
    PostgreSQL full-text search (tsvector) for keyword matching.
    """

    def __init__(self, config: NeuralMemConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._dim = config.embedding_dim
        self._dsn = getattr(config, "pg_dsn", "postgresql://localhost:5432/neuralmem")
        self._conn = self._make_connection()
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_connection(self):  # type: ignore[no-untyped-def]
        """Create a psycopg connection (import psycopg lazily)."""
        try:
            import psycopg  # type: ignore[import]
        except ImportError as exc:
            raise StorageError(
                "psycopg is required for PgVectorStorage. "
                "Install it with: pip install 'psycopg[binary]'"
            ) from exc

        conn = psycopg.connect(self._dsn, autocommit=False)
        # Register pgvector type support if available
        try:
            import pgvector.psycopg  # type: ignore[import]
            pgvector.psycopg.register_vector(conn)
        except Exception:
            _logger.debug("pgvector.psycopg.register_vector not available; "
                          "embedding will be handled as text literals.")
        return conn

    def _reconnect(self) -> None:
        """Reconnect if the connection is closed."""
        try:
            if self._conn.closed:
                self._conn = self._make_connection()
        except Exception:
            try:
                self._conn = self._make_connection()
            except Exception as exc:
                raise StorageError(f"Reconnection failed: {exc}") from exc

    def _init_db(self) -> None:
        """Create tables, extension, indexes."""
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:
            raise StorageError(
                "psycopg is required for PgVectorStorage."
            ) from exc

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(_CREATE_EXTENSION)
            cur.execute(_CREATE_MEMORIES.format(dim=self._dim))
            cur.execute(_CREATE_FTS)
            cur.execute(_CREATE_FTS_INDEX)
            cur.execute(_CREATE_GRAPH_SNAPSHOTS)
            cur.execute(_CREATE_MEMORY_HISTORY)
            cur.execute(_CREATE_MEMORY_HISTORY_INDEX)
            # Indexes for common query patterns
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user_id "
                "ON memories(user_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_is_active "
                "ON memories(is_active)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_created_at "
                "ON memories(created_at)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_importance "
                "ON memories(importance)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_session_id "
                "ON memories(session_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_embedding "
                "ON memories USING ivfflat (embedding vector_cosine_ops)"
            )
            self._conn.commit()

    def _execute(self, sql: str, params: tuple[Any, ...] | list[Any] | None = None):  # type: ignore[no-untyped-def]
        """Execute a SQL statement, returning the cursor."""
        try:
            self._reconnect()
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(sql, params or ())
                return cur
        except Exception as exc:
            raise StorageError(f"PostgreSQL error: {exc}") from exc

    def _fetchall(
        self, sql: str, params: tuple[Any, ...] | list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute and return all rows as dicts."""
        cur = self._execute(sql, params)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def _fetchone(
        self, sql: str, params: tuple[Any, ...] | list[Any] | None = None
    ) -> dict[str, Any] | None:
        """Execute and return one row as dict."""
        cur = self._execute(sql, params)
        row = cur.fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in cur.description] if cur.description else []
        return dict(zip(columns, row))

    def __del__(self) -> None:
        """Close connection on garbage collection."""
        try:
            if hasattr(self, "_conn") and not self._conn.closed:
                self._conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # StorageBackend implementation
    # ------------------------------------------------------------------

    def save_memory(self, memory: Memory) -> str:
        """Insert or replace a memory record."""
        tags_json = json.dumps(list(memory.tags))
        entity_ids_json = json.dumps(list(memory.entity_ids))
        supersedes_json = json.dumps(list(memory.supersedes))
        emb = _embedding_to_list(memory.embedding)

        session_layer = getattr(memory, "_session_layer", "long_term")
        conversation_id = getattr(memory, "_conversation_id", None)
        expires_at_str = memory.expires_at.isoformat() if memory.expires_at else None

        with self._lock:
            self._execute(
                """
                INSERT INTO memories
                    (id, content, memory_type, scope, user_id, agent_id, session_id,
                     tags, source, importance, entity_ids,
                     is_active, superseded_by, supersedes,
                     created_at, updated_at, last_accessed, access_count, embedding,
                     session_layer, conversation_id, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                       %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    memory_type = EXCLUDED.memory_type,
                    scope = EXCLUDED.scope,
                    user_id = EXCLUDED.user_id,
                    agent_id = EXCLUDED.agent_id,
                    session_id = EXCLUDED.session_id,
                    tags = EXCLUDED.tags,
                    source = EXCLUDED.source,
                    importance = EXCLUDED.importance,
                    entity_ids = EXCLUDED.entity_ids,
                    is_active = EXCLUDED.is_active,
                    superseded_by = EXCLUDED.superseded_by,
                    supersedes = EXCLUDED.supersedes,
                    updated_at = EXCLUDED.updated_at,
                    last_accessed = EXCLUDED.last_accessed,
                    access_count = EXCLUDED.access_count,
                    embedding = EXCLUDED.embedding,
                    session_layer = EXCLUDED.session_layer,
                    conversation_id = EXCLUDED.conversation_id,
                    expires_at = EXCLUDED.expires_at
                """,
                (
                    memory.id,
                    memory.content,
                    memory.memory_type.value,
                    memory.scope.value,
                    memory.user_id,
                    memory.agent_id,
                    memory.session_id,
                    tags_json,
                    memory.source,
                    memory.importance,
                    entity_ids_json,
                    memory.is_active,
                    memory.superseded_by,
                    supersedes_json,
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                    memory.last_accessed.isoformat(),
                    memory.access_count,
                    emb,
                    session_layer,
                    conversation_id,
                    expires_at_str,
                ),
            )
            self._conn.commit()

        return memory.id

    def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID."""
        row = self._fetchone("SELECT * FROM memories WHERE id = %s", (memory_id,))
        if row is None:
            return None
        return _row_to_memory(row)

    def update_memory(self, memory_id: str, **kwargs: object) -> None:
        """Update specific fields of a memory."""
        allowed = {
            "content", "memory_type", "scope", "user_id", "agent_id", "session_id",
            "tags", "source", "importance", "entity_ids", "embedding",
            "is_active", "superseded_by", "supersedes",
            "session_layer", "conversation_id",
            "expires_at",
        }
        updates: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key not in allowed:
                raise StorageError(f"Unknown field for update: {key}")
            if key == "tags":
                updates["tags"] = json.dumps(list(value))  # type: ignore[arg-type]
            elif key == "entity_ids":
                updates["entity_ids"] = json.dumps(list(value))  # type: ignore[arg-type]
            elif key == "supersedes":
                updates["supersedes"] = json.dumps(list(value))  # type: ignore[arg-type]
            elif key == "is_active":
                updates["is_active"] = bool(value)
            elif key == "embedding":
                updates["embedding"] = _embedding_to_list(value)  # type: ignore[arg-type]
            elif key == "memory_type" and isinstance(value, MemoryType):
                updates["memory_type"] = value.value
            elif key == "scope" and isinstance(value, MemoryScope):
                updates["scope"] = value.value
            elif key == "expires_at":
                updates["expires_at"] = (
                    value.isoformat() if isinstance(value, datetime) else value
                )
            else:
                updates[key] = value

        updates["updated_at"] = _now_iso()
        set_clause = ", ".join(f"{k} = %s" for k in updates)
        values = list(updates.values()) + [memory_id]

        with self._lock:
            self._execute(f"UPDATE memories SET {set_clause} WHERE id = %s", tuple(values))
            self._conn.commit()

    def delete_memories(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        before: object = None,
        tags: list[str] | None = None,
        max_importance: float | None = None,
    ) -> int:
        """Delete memories matching filters. Returns count of deleted rows."""
        conditions: list[str] = []
        params: list[Any] = []

        if memory_id is not None:
            conditions.append("id = %s")
            params.append(memory_id)
        if user_id is not None:
            conditions.append("user_id = %s")
            params.append(user_id)
        if before is not None:
            if isinstance(before, datetime):
                conditions.append("created_at < %s")
                params.append(before.isoformat())
            else:
                conditions.append("created_at < %s")
                params.append(str(before))
        if tags:
            # Check if any tag is present in the JSON tags array
            tag_conditions = " OR ".join("tags ILIKE %s" for _ in tags)
            conditions.append(f"({tag_conditions})")
            for tag in tags:
                params.append(f'%"{tag}"%')
        if max_importance is not None:
            conditions.append("importance < %s")
            params.append(max_importance)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._lock:
            cur = self._execute(f"DELETE FROM memories {where}", tuple(params))
            self._conn.commit()
            return cur.rowcount

    def vector_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Search by vector similarity using pgvector cosine distance (<=>)."""
        conditions: list[str] = ["embedding IS NOT NULL"]
        conditions.append("(expires_at IS NULL OR expires_at > %s)")
        params: list[Any] = [_now_iso()]
        if user_id is not None:
            conditions.append("user_id = %s")
            params.append(user_id)
        if memory_types:
            placeholders = ", ".join("%s" for _ in memory_types)
            conditions.append(f"memory_type IN ({placeholders})")
            params.extend(t.value for t in memory_types)

        where = " AND ".join(conditions)
        # <=> is cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: 1 - distance
        params.append(limit)

        try:
            rows = self._fetchall(
                f"""
                SELECT id, 1 - (embedding <=> %s::vector) AS score
                FROM memories
                WHERE {where}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                [str(vector)] + params[:-1] + [str(vector)] + [limit],
            )
            results: list[tuple[str, float]] = []
            for row in rows:
                score = max(0.0, float(row["score"]))
                results.append((row["id"], score))
            return results
        except Exception as exc:
            _logger.warning("pgvector search failed (%s).", exc)
            return self._vector_search_numpy(vector, user_id, memory_types, limit)

    def _vector_search_numpy(
        self,
        vector: list[float],
        user_id: str | None,
        memory_types: list[MemoryType] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Fallback brute-force cosine search using numpy."""
        conditions = ["embedding IS NOT NULL"]
        now_str = _now_iso()
        conditions.append("(expires_at IS NULL OR expires_at > %s)")
        params: list[Any] = [now_str]
        if user_id is not None:
            conditions.append("user_id = %s")
            params.append(user_id)
        if memory_types:
            placeholders = ", ".join("%s" for _ in memory_types)
            conditions.append(f"memory_type IN ({placeholders})")
            params.extend(t.value for t in memory_types)

        where = " AND ".join(conditions)
        rows = self._fetchall(
            f"SELECT id, embedding FROM memories WHERE {where}",
            tuple(params),
        )

        query_vec = np.array(vector, dtype=np.float32)
        scored: list[tuple[str, float]] = []
        for row in rows:
            emb_raw = row["embedding"]
            if emb_raw is None:
                continue
            if isinstance(emb_raw, str):
                emb = np.array(
                    [float(x) for x in emb_raw.strip("[]").split(",")],
                    dtype=np.float32,
                )
            else:
                emb = np.array(list(emb_raw), dtype=np.float32)
            score = _cosine_similarity(query_vec, emb)
            scored.append((row["id"], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def keyword_search(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Full-text search using PostgreSQL tsvector/tsquery."""
        join_conditions: list[str] = [
            "(expires_at IS NULL OR expires_at > %s)"
        ]
        params: list[Any] = [_now_iso()]

        # Build tsquery: split on whitespace, join with &
        tsquery = " & ".join(query.split())
        params.append(tsquery)

        if user_id is not None:
            join_conditions.append("user_id = %s")
            params.append(user_id)
        if memory_types:
            placeholders = ", ".join("%s" for _ in memory_types)
            join_conditions.append(f"memory_type IN ({placeholders})")
            params.extend(t.value for t in memory_types)

        extra_where = f"AND {' AND '.join(join_conditions)}" if join_conditions else ""
        params.append(limit)

        try:
            rows = self._fetchall(
                f"""
                SELECT id,
                       ts_rank_cd(content_tsv, to_tsquery('english', %s)) AS rank
                FROM memories
                WHERE content_tsv @@ to_tsquery('english', %s) {extra_where}
                ORDER BY rank DESC
                LIMIT %s
                """,
                [tsquery] + params,
            )
            results: list[tuple[str, float]] = []
            for row in rows:
                rank = float(row["rank"])
                # Normalise rank to [0, 1]
                score = min(1.0, max(0.0, rank * 10))
                results.append((row["id"], score))
            return results
        except Exception as exc:
            _logger.warning("Full-text search failed (%s). Falling back to LIKE.", exc)
            return self._keyword_search_like(query, user_id, memory_types, limit)

    def _keyword_search_like(
        self,
        query: str,
        user_id: str | None,
        memory_types: list[MemoryType] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Fallback keyword search using ILIKE."""
        conditions = ["content ILIKE %s"]
        now_str = _now_iso()
        conditions.append("(expires_at IS NULL OR expires_at > %s)")
        params: list[Any] = [f"%{query}%", now_str]
        if user_id is not None:
            conditions.append("user_id = %s")
            params.append(user_id)
        if memory_types:
            placeholders = ", ".join("%s" for _ in memory_types)
            conditions.append(f"memory_type IN ({placeholders})")
            params.extend(t.value for t in memory_types)

        where = " AND ".join(conditions)
        params.append(limit)

        rows = self._fetchall(
            f"SELECT id FROM memories WHERE {where} LIMIT %s",
            tuple(params),
        )
        # Assign a simple 0.5 score for LIKE matches
        return [(row["id"], 0.5) for row in rows]

    def temporal_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        time_range: tuple[object, object] | None = None,
        recency_weight: float = 0.3,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Blend vector similarity with recency."""
        candidates = self.vector_search(vector, user_id=user_id, limit=limit * 3)
        if not candidates:
            return []

        candidate_ids = [cid for cid, _ in candidates]
        semantic_scores = {cid: score for cid, score in candidates}

        placeholders = ", ".join("%s" for _ in candidate_ids)
        params: list[Any] = list(candidate_ids)

        time_conditions: list[str] = []
        if time_range is not None:
            start, end = time_range
            if start is not None:
                time_conditions.append("created_at >= %s")
                params.append(start.isoformat() if isinstance(start, datetime) else str(start))
            if end is not None:
                time_conditions.append("created_at <= %s")
                params.append(end.isoformat() if isinstance(end, datetime) else str(end))

        extra_where = f"AND {' AND '.join(time_conditions)}" if time_conditions else ""

        rows = self._fetchall(
            f"SELECT id, created_at FROM memories WHERE id IN ({placeholders}) {extra_where}",
            tuple(params),
        )

        now = datetime.now(timezone.utc)
        scored: list[tuple[str, float]] = []
        for row in rows:
            mem_id = row["id"]
            created = row["created_at"]
            if isinstance(created, str):
                created = datetime.fromisoformat(created)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            days_ago = max(0.0, (now - created).total_seconds() / 86400)
            time_score = max(0.0, 1.0 - days_ago / 365)

            sem_score = semantic_scores.get(mem_id, 0.0)
            final_score = sem_score * (1.0 - recency_weight) + time_score * recency_weight
            scored.append((mem_id, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def find_similar(
        self, vector: list[float], user_id: str | None = None, threshold: float = 0.95
    ) -> list[Memory]:
        """Find memories with similarity >= threshold."""
        candidates = self.vector_search(vector, user_id=user_id, limit=100)
        similar_ids = [cid for cid, score in candidates if score >= threshold]
        if not similar_ids:
            return []

        placeholders = ", ".join("%s" for _ in similar_ids)
        rows = self._fetchall(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", tuple(similar_ids)
        )
        return [_row_to_memory(row) for row in rows]

    def record_access(self, memory_id: str) -> None:
        """Increment access_count and update last_accessed."""
        with self._lock:
            self._execute(
                "UPDATE memories SET last_accessed = %s, "
                "access_count = access_count + 1 WHERE id = %s",
                (_now_iso(), memory_id),
            )
            self._conn.commit()

    def batch_record_access(self, memory_ids: list[str]) -> None:
        """Record access for multiple memories in a single query."""
        if not memory_ids:
            return
        now = _now_iso()
        placeholders = ", ".join("%s" for _ in memory_ids)
        with self._lock:
            self._execute(
                f"UPDATE memories SET last_accessed = %s, "
                f"access_count = access_count + 1 WHERE id IN ({placeholders})",
                [now] + memory_ids,
            )
            self._conn.commit()

    def get_stats(self, user_id: str | None = None) -> dict[str, object]:
        """Return memory statistics."""
        params: list[Any] = []
        where = ""
        if user_id is not None:
            where = "WHERE user_id = %s"
            params.append(user_id)

        total_row = self._fetchone(
            f"SELECT COUNT(*) AS total FROM memories {where}", tuple(params)
        )
        total = total_row["total"] if total_row else 0

        type_rows = self._fetchall(
            f"SELECT memory_type, COUNT(*) AS cnt FROM memories {where} GROUP BY memory_type",
            tuple(params),
        )
        by_type = {row["memory_type"]: row["cnt"] for row in type_rows}

        entity_rows = self._fetchall(
            f"SELECT entity_ids FROM memories {where}", tuple(params)
        )
        all_entity_ids: set[str] = set()
        for row in entity_rows:
            ids = json.loads(row["entity_ids"] or "[]")
            all_entity_ids.update(ids)

        return {
            "total": total,
            "by_type": by_type,
            "entity_count": len(all_entity_ids),
        }

    def list_memories(
        self, user_id: str | None = None, limit: int = 10_000
    ) -> list[Memory]:
        """List memories, optionally filtered by user_id."""
        now_str = _now_iso()
        try:
            if user_id is not None:
                rows = self._fetchall(
                    "SELECT * FROM memories WHERE user_id = %s "
                    "AND (expires_at IS NULL OR expires_at > %s) LIMIT %s",
                    (user_id, now_str, limit),
                )
            else:
                rows = self._fetchall(
                    "SELECT * FROM memories "
                    "WHERE (expires_at IS NULL OR expires_at > %s) LIMIT %s",
                    (now_str, limit),
                )
            return [_row_to_memory(row) for row in rows]
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"list_memories failed: {e}") from e

    def load_graph_snapshot(self) -> dict | None:
        """Load graph snapshot (JSON)."""
        try:
            row = self._fetchone(
                "SELECT data FROM graph_snapshots WHERE key = %s",
                ("default",),
            )
            if row:
                data = row["data"]
                if isinstance(data, str):
                    return json.loads(data)
                return data  # Already a dict (from jsonb)
            return None
        except Exception as e:
            _logger.warning("Failed to load graph snapshot: %s", e)
            return None

    def save_graph_snapshot(self, data: dict) -> None:
        """Persist graph snapshot."""
        try:
            now = _now_iso()
            with self._lock:
                self._execute(
                    """
                    INSERT INTO graph_snapshots (key, data, updated_at)
                    VALUES (%s, %s::jsonb, %s)
                    ON CONFLICT (key) DO UPDATE SET
                        data = EXCLUDED.data,
                        updated_at = EXCLUDED.updated_at
                    """,
                    ("default", json.dumps(data), now),
                )
                self._conn.commit()
        except Exception as e:
            _logger.warning("Failed to save graph snapshot: %s", e)

    def save_history(
        self,
        memory_id: str,
        old_content: str | None,
        new_content: str,
        event: str = "UPDATE",
        metadata: dict | None = None,
    ) -> None:
        """Record a memory version change."""
        with self._lock:
            self._execute(
                "INSERT INTO memory_history"
                " (memory_id, old_content, new_content,"
                " event, changed_at, metadata)"
                " VALUES (%s, %s, %s, %s, %s, %s::jsonb)",
                (
                    memory_id, old_content, new_content,
                    event, _now_iso(), json.dumps(metadata or {}),
                ),
            )
            self._conn.commit()

    def get_history(self, memory_id: str) -> list[dict[str, object]]:
        """Retrieve the version history for a memory."""
        rows = self._fetchall(
            "SELECT id, memory_id, old_content,"
            " new_content, event, changed_at,"
            " metadata FROM memory_history"
            " WHERE memory_id = %s ORDER BY id ASC",
            (memory_id,),
        )
        return [
            {
                "id": row["id"],
                "memory_id": row["memory_id"],
                "old_content": row["old_content"],
                "new_content": row["new_content"],
                "event": row["event"],
                "changed_at": row["changed_at"],
                "metadata": (
                    row["metadata"]
                    if isinstance(row["metadata"], dict)
                    else json.loads(row["metadata"])
                    if row["metadata"]
                    else {}
                ),
            }
            for row in rows
        ]

    def cleanup_expired(self) -> int:
        """Delete memories whose expiration time has passed.

        Returns:
            Number of memories deleted.
        """
        now_str = _now_iso()
        with self._lock:
            cur = self._execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= %s",
                (now_str,),
            )
            self._conn.commit()
            return cur.rowcount

    def batch_find_similar(
        self,
        vectors: list[list[float]],
        user_id: str | None = None,
        threshold: float = 0.95,
    ) -> dict[int, list[Memory]]:
        """Find similar memories for multiple vectors in a single DB load.

        Loads all active memories for the user once, then computes cosine
        similarity for each input vector against all loaded memories.
        """
        conditions = ["embedding IS NOT NULL", "is_active = TRUE"]
        now_str = _now_iso()
        conditions.append("(expires_at IS NULL OR expires_at > %s)")
        params: list[Any] = [now_str]
        if user_id is not None:
            conditions.append("user_id = %s")
            params.append(user_id)

        where = " AND ".join(conditions)
        rows = self._fetchall(
            f"SELECT * FROM memories WHERE {where}",
            tuple(params),
        )
        if not rows:
            return {i: [] for i in range(len(vectors))}

        # Pre-compute memory embeddings as numpy arrays
        mem_embeddings: list[tuple[Memory, np.ndarray]] = []
        for row in rows:
            emb_raw = row.get("embedding")
            if emb_raw is None:
                continue
            mem = _row_to_memory(row)
            if isinstance(emb_raw, str):
                emb = np.array(
                    [float(x) for x in emb_raw.strip("[]").split(",")],
                    dtype=np.float32,
                )
            else:
                emb = np.array(list(emb_raw), dtype=np.float32)
            mem_embeddings.append((mem, emb))

        # Compute similarity for each input vector
        results: dict[int, list[Memory]] = {}
        for idx, vec in enumerate(vectors):
            query_vec = np.array(vec, dtype=np.float32)
            similar: list[Memory] = []
            for mem, mem_emb in mem_embeddings:
                score = _cosine_similarity(query_vec, mem_emb)
                if score >= threshold:
                    similar.append(mem)
            results[idx] = similar

        return results
