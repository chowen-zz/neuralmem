"""SQLite + sqlite-vec 存储后端实现（含 numpy 回退）"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import StorageError
from neuralmem.core.types import Memory, MemoryScope, MemoryType
from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)

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
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1,
    superseded_by TEXT,
    supersedes TEXT DEFAULT '[]',
    embedding BLOB,
    expires_at TEXT
)
"""

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(id UNINDEXED, content, tokenize='unicode61')
"""

_CREATE_VEC = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec
    USING vec0(embedding float[{dim}])
"""

_CREATE_GRAPH_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS graph_snapshots (
    key TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_CREATE_GRAPH_NODES = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'unknown',
    aliases TEXT DEFAULT '[]',
    attributes TEXT DEFAULT '{}',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    memory_ids TEXT DEFAULT '[]',
    updated_at TEXT NOT NULL
)
"""

_CREATE_GRAPH_EDGES = """
CREATE TABLE IF NOT EXISTS graph_edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    timestamp TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id)
)
"""

_CREATE_MEMORY_HISTORY = """
CREATE TABLE IF NOT EXISTS memory_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    old_content TEXT,
    new_content TEXT NOT NULL,
    event TEXT NOT NULL DEFAULT 'UPDATE',
    changed_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
)
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _embedding_to_blob(embedding: list[float]) -> bytes:
    return np.array(embedding, dtype=np.float32).tobytes()


def _blob_to_embedding(blob: bytes | None) -> np.ndarray:
    if blob is None:
        return np.array([], dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32)


def _memory_from_row(row: sqlite3.Row) -> Memory:
    expires_at_raw = row["expires_at"] if "expires_at" in row.keys() else None
    expires_at = datetime.fromisoformat(expires_at_raw) if expires_at_raw else None
    # 防御性解析：并发读写时 SQLite 返回空值/None，使用安全默认值
    raw_type = row["memory_type"]
    memory_type = MemoryType(raw_type) if raw_type else MemoryType.SEMANTIC
    raw_scope = row["scope"]
    scope = MemoryScope(raw_scope) if raw_scope else MemoryScope.USER
    return Memory(
        id=row["id"],
        content=row["content"],
        memory_type=memory_type,
        scope=scope,
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        session_id=row["session_id"],
        tags=tuple(json.loads(row["tags"] or "[]")),
        source=row["source"],
        importance=row["importance"],
        entity_ids=tuple(json.loads(row["entity_ids"] or "[]")),
        is_active=bool(row["is_active"]) if "is_active" in row.keys() else True,
        superseded_by=row["superseded_by"] if "superseded_by" in row.keys() else None,
        supersedes=(
            tuple(json.loads(row["supersedes"] or "[]"))
            if "supersedes" in row.keys()
            else ()
        ),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
        last_accessed=datetime.fromisoformat(str(row["last_accessed"])),
        access_count=row["access_count"],
        embedding=list(_blob_to_embedding(row["embedding"])) if row["embedding"] else None,
        expires_at=expires_at,
    )


class SQLiteStorage(StorageBackend):
    """SQLite + sqlite-vec 存储后端，sqlite-vec 不可用时自动降级为 numpy 暴力搜索"""

    def __init__(self, config: NeuralMemConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._vec_available = False
        self._dim = config.embedding_dim
        db_path = config.get_db_path()
        self._db_path = str(db_path)
        self._conn = self._make_connection()
        self._init_db()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _make_connection(self) -> sqlite3.Connection:
        # isolation_level=None disables Python's implicit transaction management.
        # This prevents "cannot commit - no transaction is active" errors in
        # multi-threaded scenarios (e.g. ThreadPoolExecutor in retrieval engine).
        # All statements auto-commit; multi-statement writes that need atomicity
        # use explicit BEGIN/COMMIT via _execute_tx().
        conn = sqlite3.connect(self._db_path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA mmap_size=268435456")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        # 尝试加载 sqlite-vec 扩展
        try:
            import sqlite_vec  # type: ignore[import]
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._vec_available = True
            _logger.info("sqlite-vec loaded successfully.")
        except Exception as exc:
            _logger.warning("sqlite-vec unavailable (%s). Using numpy fallback.", exc)
            self._vec_available = False

        with self._lock:
            self._conn.execute(_CREATE_MEMORIES)
            self._conn.execute(_CREATE_FTS)
            self._conn.execute(_CREATE_GRAPH_SNAPSHOTS)
            self._conn.execute(_CREATE_GRAPH_NODES)
            self._conn.execute(_CREATE_GRAPH_EDGES)
            # Migration: add new columns if they don't exist
            self._migrate_schema()
            if self._vec_available:
                try:
                    self._conn.execute(_CREATE_VEC.format(dim=self._dim))
                except Exception as exc:
                    _logger.warning("Failed to create vec0 table (%s). Falling back to numpy.", exc)
                    self._vec_available = False
            # Create indexes for common query patterns
            self._conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)'
            )
            self._conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_memories_is_active ON memories(is_active)'
            )
            self._conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)'
            )
            self._conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)'
            )
            self._conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id)'
            )
            self._conn.execute(_CREATE_MEMORY_HISTORY)
            self._conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_memory_history_id ON memory_history(memory_id)'
            )
            self._conn.commit()

    def _migrate_schema(self) -> None:
        """Add columns introduced in later versions if they don't exist."""
        cur = self._execute("PRAGMA table_info(memories)")
        existing = {row["name"] for row in cur.fetchall()}
        migrations = [
            ("is_active", "INTEGER DEFAULT 1"),
            ("superseded_by", "TEXT"),
            ("supersedes", "TEXT DEFAULT '[]'"),
            ("session_layer", "TEXT DEFAULT 'long_term'"),
            ("conversation_id", "TEXT"),
            ("expires_at", "TEXT"),
        ]
        for col, col_def in migrations:
            if col not in existing:
                try:
                    self._execute(f"ALTER TABLE memories ADD COLUMN {col} {col_def}")
                    _logger.info("Migration: added column %s", col)
                except Exception as exc:
                    _logger.warning("Migration failed for column %s: %s", col, exc)
        self._conn.commit()

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._lock:
                    return self._conn.execute(sql, params)
            except sqlite3.OperationalError as exc:
                if "cannot commit" in str(exc) and attempt < max_retries - 1:
                    import time
                    time.sleep(0.01 * (attempt + 1))
                    continue
                raise StorageError(f"SQLite error: {exc}") from exc
            except sqlite3.Error as exc:
                raise StorageError(f"SQLite error: {exc}") from exc
        raise StorageError("SQLite error: max retries exceeded")  # unreachable

    def _fetchall(
        self, sql: str, params: tuple[Any, ...] = ()
    ) -> list[sqlite3.Row]:
        """Execute and fetchall under the same lock to prevent cursor corruption.

        Includes retry logic for concurrent write errors
        (e.g. 'cannot commit - no transaction is active').
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._lock:
                    cur = self._conn.execute(sql, params)
                    return cur.fetchall()
            except sqlite3.OperationalError as exc:
                if "cannot commit" in str(exc) and attempt < max_retries - 1:
                    import time
                    time.sleep(0.01 * (attempt + 1))
                    continue
                raise StorageError(f"SQLite error: {exc}") from exc
            except sqlite3.Error as exc:
                raise StorageError(f"SQLite error: {exc}") from exc
        return []  # unreachable, but satisfies type checker

    # ------------------------------------------------------------------
    # StorageBackend 实现
    # ------------------------------------------------------------------

    def save_memory(self, memory: Memory) -> str:
        tags_json = json.dumps(list(memory.tags))
        entity_ids_json = json.dumps(list(memory.entity_ids))
        embedding_blob = _embedding_to_blob(memory.embedding) if memory.embedding else None

        # Derive session_layer and conversation_id from memory metadata
        session_layer = getattr(memory, '_session_layer', 'long_term')
        conversation_id = getattr(memory, '_conversation_id', None)

        # Convert expires_at to ISO string
        expires_at_str = memory.expires_at.isoformat() if memory.expires_at else None

        with self._lock:
            supersedes_json = json.dumps(list(memory.supersedes))
            self._execute(
                """
                INSERT OR REPLACE INTO memories
                    (id, content, memory_type, scope, user_id, agent_id, session_id,
                     tags, source, importance, entity_ids,
                     is_active, superseded_by, supersedes,
                     created_at, updated_at, last_accessed, access_count, embedding,
                     session_layer, conversation_id, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    int(memory.is_active),
                    memory.superseded_by,
                    supersedes_json,
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                    memory.last_accessed.isoformat(),
                    memory.access_count,
                    embedding_blob,
                    session_layer,
                    conversation_id,
                    expires_at_str,
                ),
            )
            # 同步 FTS
            self._execute("DELETE FROM memories_fts WHERE id = ?", (memory.id,))
            self._execute(
                "INSERT INTO memories_fts (id, content) VALUES (?, ?)",
                (memory.id, memory.content),
            )
            # 同步 vec0（如果可用且有 embedding）
            if self._vec_available and memory.embedding:
                try:
                    self._execute(
                        "DELETE FROM memories_vec WHERE rowid = "
                        "(SELECT rowid FROM memories WHERE id = ?)",
                        (memory.id,),
                    )
                    cur = self._execute("SELECT rowid FROM memories WHERE id = ?", (memory.id,))
                    row = cur.fetchone()
                    if row:
                        self._execute(
                            "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                            (row["rowid"], embedding_blob),
                        )
                except Exception as exc:
                    _logger.warning("vec0 insert failed (%s). Vector search may be degraded.", exc)
            self._conn.commit()

        return memory.id

    def get_memory(self, memory_id: str) -> Memory | None:
        rows = self._fetchall(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        )
        if not rows:
            return None
        return _memory_from_row(rows[0])

    def update_memory(self, memory_id: str, **kwargs: object) -> None:
        allowed = {
            "content", "memory_type", "scope", "user_id", "agent_id", "session_id",
            "tags", "source", "importance", "entity_ids", "embedding",
            "is_active", "superseded_by", "supersedes",
            "session_layer", "conversation_id",
            "expires_at", "access_count", "last_accessed", "updated_at",
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
                updates["is_active"] = int(bool(value))
            elif key == "embedding":
                updates["embedding"] = _embedding_to_blob(value) if value else None  # type: ignore[arg-type]
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
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [memory_id]

        with self._lock:
            self._execute(f"UPDATE memories SET {set_clause} WHERE id = ?", tuple(values))
            if "content" in updates:
                self._execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
                cur = self._execute("SELECT content FROM memories WHERE id = ?", (memory_id,))
                row = cur.fetchone()
                if row:
                    self._execute(
                        "INSERT INTO memories_fts (id, content) VALUES (?, ?)",
                        (memory_id, row["content"]),
                    )
            self._conn.commit()

    def delete_memories(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        before: object = None,
        tags: list[str] | None = None,
        max_importance: float | None = None,
    ) -> int:
        conditions: list[str] = []
        params: list[Any] = []

        if memory_id is not None:
            conditions.append("id = ?")
            params.append(memory_id)
        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)
        if before is not None:
            if isinstance(before, datetime):
                conditions.append("created_at < ?")
                params.append(before.isoformat())
            else:
                conditions.append("created_at < ?")
                params.append(str(before))
        if tags:
            # 逐一检查 tags JSON 字段包含任意一个 tag
            tag_conditions = " OR ".join("tags LIKE ?" for _ in tags)
            conditions.append(f"({tag_conditions})")
            for tag in tags:
                params.append(f'%"{tag}"%')
        if max_importance is not None:
            conditions.append("importance < ?")
            params.append(max_importance)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._lock:
            # 先找到要删除的 id 列表，用于清理 FTS 和 vec
            id_cur = self._execute(f"SELECT id FROM memories {where}", tuple(params))
            ids_to_delete = [r["id"] for r in id_cur.fetchall()]

            if not ids_to_delete:
                return 0

            placeholders = ",".join("?" for _ in ids_to_delete)
            self._execute(
                f"DELETE FROM memories_fts WHERE id IN ({placeholders})", tuple(ids_to_delete)
            )
            if self._vec_available:
                try:
                    self._execute(
                        f"""DELETE FROM memories_vec WHERE rowid IN (
                            SELECT rowid FROM memories WHERE id IN ({placeholders})
                        )""",
                        tuple(ids_to_delete),
                    )
                except Exception as exc:
                    _logger.warning("vec0 delete failed (%s).", exc)

            cur = self._execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", tuple(ids_to_delete)
            )
            self._conn.commit()
            return cur.rowcount

    def vector_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        if self._vec_available:
            return self._vector_search_vec0(vector, user_id, memory_types, limit)
        return self._vector_search_numpy(vector, user_id, memory_types, limit)

    def _vector_search_vec0(
        self,
        vector: list[float],
        user_id: str | None,
        memory_types: list[MemoryType] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        # vec0 KNN 子查询 + JOIN memories 过滤条件
        conditions: list[str] = ["m.embedding IS NOT NULL"]
        conditions.append("(m.expires_at IS NULL OR m.expires_at > ?)")
        params: list[Any] = []
        params.append(_now_iso())
        if user_id is not None:
            conditions.append("m.user_id = ?")
            params.append(user_id)
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            conditions.append(f"m.memory_type IN ({placeholders})")
            params.extend(t.value for t in memory_types)

        where = " AND ".join(conditions)
        try:
            # vec0 要求 MATCH + LIMIT 在同一查询中，使用子查询模式
            rows = self._fetchall(
                f"""
                SELECT m.id, v.distance
                FROM (
                    SELECT rowid, distance FROM memories_vec
                    WHERE embedding MATCH ?
                    ORDER BY distance LIMIT ?
                ) v
                JOIN memories m ON m.rowid = v.rowid
                WHERE {where}
                LIMIT ?
                """,
                (_embedding_to_blob(vector), limit * 3) + tuple(params) + (limit,),
            )
            # distance 越小越相似，转换为相似度分数（余弦距离 = 1 - 余弦相似度）
            results = []
            for row in rows:
                dist = row["distance"]
                if dist is not None:
                    score = max(0.0, 1.0 - float(dist))
                    results.append((row["id"], score))
            return results
        except Exception as exc:
            _logger.debug("vec0 search unavailable (%s). Using numpy fallback.", exc)
            return self._vector_search_numpy(vector, user_id, memory_types, limit)

    def _vector_search_numpy(
        self,
        vector: list[float],
        user_id: str | None,
        memory_types: list[MemoryType] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        conditions = ["embedding IS NOT NULL"]
        now_str = _now_iso()
        conditions.append("(expires_at IS NULL OR expires_at > ?)")
        params: list[Any] = []
        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            conditions.append(f"memory_type IN ({placeholders})")
            params.extend(t.value for t in memory_types)

        where = " AND ".join(conditions)
        rows = self._fetchall(
            f"SELECT id, embedding FROM memories WHERE {where}",
            (now_str,) + tuple(params),
        )

        query_vec = np.array(vector, dtype=np.float32)
        scored: list[tuple[str, float]] = []
        for row in rows:
            emb = _blob_to_embedding(row["embedding"])
            if emb.size == 0:
                continue
            score = _cosine_similarity(query_vec, emb)
            scored.append((row["id"], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Sanitize a user query for FTS5 MATCH.

        FTS5 treats ``?``, ``*``, ``"``, ``OR``, ``AND``, ``NOT``, ``NEAR``
        and column filters as operators.  We want plain full-text matching, so
        we extract alphanumeric tokens, quote each one, and join with implicit
        OR (space-separated).
        """
        # Extract word-like tokens (alphanumeric + CJK)
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", query)
        if not tokens:
            # Fallback: strip anything that isn't a word char
            cleaned = re.sub(r"[^\w\s]", " ", query).strip()
            tokens = cleaned.split()
        if not tokens:
            return '""'  # empty match → no results
        # Quote each token so FTS5 treats it as a literal term
        return " OR ".join(f'"{t}"' for t in tokens)

    def keyword_search(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        # FTS5 BM25（rank 值为负数，越小越相关）
        fts_query = self._sanitize_fts5_query(query)
        join_conditions: list[str] = []
        join_conditions.append("(m.expires_at IS NULL OR m.expires_at > ?)")
        params: list[Any] = [fts_query]
        params.append(_now_iso())
        if user_id is not None:
            join_conditions.append("m.user_id = ?")
            params.append(user_id)
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            join_conditions.append(f"m.memory_type IN ({placeholders})")
            params.extend(t.value for t in memory_types)

        extra_where = f"AND {' AND '.join(join_conditions)}" if join_conditions else ""
        params.append(limit)

        try:
            rows = self._fetchall(
                f"""
                SELECT f.id, f.rank
                FROM memories_fts f
                JOIN memories m ON m.id = f.id
                WHERE memories_fts MATCH ? {extra_where}
                ORDER BY f.rank
                LIMIT ?
                """,
                tuple(params),
            )
        except (sqlite3.OperationalError, StorageError):
            # FTS 语法错误等，返回空结果
            return []

        results: list[tuple[str, float]] = []
        for row in rows:
            # rank 为负值，将其归一化为 [0, 1] 分数
            raw_rank = float(row["rank"])
            score = 1.0 / (1.0 + abs(raw_rank))
            results.append((row["id"], score))
        return results

    def temporal_search(
        self,
        vector: list[float],
        user_id: str | None = None,
        time_range: tuple[object, object] | None = None,
        recency_weight: float = 0.3,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        # 先做向量搜索（多取一些候选）
        candidates = self.vector_search(vector, user_id=user_id, limit=limit * 3)
        if not candidates:
            return []

        candidate_ids = [cid for cid, _ in candidates]
        semantic_scores = {cid: score for cid, score in candidates}

        placeholders = ",".join("?" for _ in candidate_ids)
        params: list[Any] = list(candidate_ids)

        # 时间范围过滤
        time_conditions: list[str] = []
        if time_range is not None:
            start, end = time_range
            if start is not None:
                time_conditions.append("created_at >= ?")
                params.append(start.isoformat() if isinstance(start, datetime) else str(start))
            if end is not None:
                time_conditions.append("created_at <= ?")
                params.append(end.isoformat() if isinstance(end, datetime) else str(end))

        extra_where = (
            f"AND {' AND '.join(time_conditions)}" if time_conditions else ""
        )

        rows = self._fetchall(
            f"SELECT id, created_at FROM memories WHERE id IN ({placeholders}) {extra_where}",
            tuple(params),
        )

        now = datetime.now(timezone.utc)
        scored: list[tuple[str, float]] = []
        for row in rows:
            mem_id = row["id"]
            created = datetime.fromisoformat(row["created_at"])
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
        candidates = self.vector_search(vector, user_id=user_id, limit=100)
        similar_ids = [cid for cid, score in candidates if score >= threshold]
        if not similar_ids:
            return []

        placeholders = ",".join("?" for _ in similar_ids)
        rows = self._fetchall(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", tuple(similar_ids)
        )
        return [_memory_from_row(row) for row in rows]

    def record_access(self, memory_id: str) -> None:
        with self._lock:
            self._execute(
                "UPDATE memories SET last_accessed = ?, "
                "access_count = access_count + 1 WHERE id = ?",
                (_now_iso(), memory_id),
            )
            self._conn.commit()

    def batch_record_access(self, memory_ids: list[str]) -> None:
        """Record access for multiple memories in a single DB call."""
        if not memory_ids:
            return
        now = _now_iso()
        placeholders = ",".join("?" * len(memory_ids))
        self._conn.execute(
            "UPDATE memories SET last_accessed = ?,"
            " access_count = access_count + 1"
            f" WHERE id IN ({placeholders})",
            [now] + memory_ids,
        )
        self._conn.commit()

    def get_stats(self, user_id: str | None = None) -> dict[str, object]:
        params: list[Any] = []
        where = ""
        if user_id is not None:
            where = "WHERE user_id = ?"
            params.append(user_id)

        cur = self._execute(
            f"SELECT COUNT(*) AS total FROM memories {where}", tuple(params)
        )
        total = cur.fetchone()["total"]

        cur = self._execute(
            f"SELECT memory_type, COUNT(*) AS cnt FROM memories {where} GROUP BY memory_type",
            tuple(params),
        )
        by_type = {row["memory_type"]: row["cnt"] for row in cur.fetchall()}

        # entity_count: 所有 entity_ids JSON 数组中唯一实体数
        cur = self._execute(f"SELECT entity_ids FROM memories {where}", tuple(params))
        all_entity_ids: set[str] = set()
        for row in cur.fetchall():
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
        """获取所有记忆，可按 user_id 过滤。"""
        now_str = _now_iso()
        try:
            if user_id is not None:
                rows = self._fetchall(
                    "SELECT * FROM memories WHERE user_id = ? "
                    "AND (expires_at IS NULL OR expires_at > ?) LIMIT ?",
                    (user_id, now_str, limit),
                )
            else:
                rows = self._fetchall(
                    "SELECT * FROM memories "
                    "WHERE (expires_at IS NULL OR expires_at > ?) LIMIT ?",
                    (now_str, limit),
                )
            return [_memory_from_row(row) for row in rows]
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"list_memories failed: {e}") from e

    def load_graph_snapshot(self) -> dict | None:
        """加载图谱快照（JSON 格式）"""
        try:
            with self._lock:
                cursor = self._execute(
                    "SELECT data FROM graph_snapshots WHERE key = 'default'", ()
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None
        except Exception as e:
            _logger.warning("Failed to load graph snapshot: %s", e)
            return None

    def save_graph_snapshot(self, data: dict) -> None:
        """持久化图谱快照"""
        try:
            now = _now_iso()
            with self._lock:
                self._execute(
                    "INSERT OR REPLACE INTO graph_snapshots "
                    "(key, data, updated_at) VALUES (?, ?, ?)",
                    ("default", json.dumps(data), now),
                )
                self._conn.commit()
        except Exception as e:
            _logger.warning("Failed to save graph snapshot: %s", e)

    def save_graph_nodes_incremental(self, nodes: list[dict]) -> None:
        """Incrementally save dirty graph nodes (INSERT OR REPLACE)."""
        if not nodes:
            return
        try:
            now = _now_iso()
            with self._lock:
                self._conn.executemany(
                    """INSERT OR REPLACE INTO graph_nodes
                       (id, name, entity_type, aliases, attributes,
                        first_seen, last_seen, memory_ids, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            n["id"], n["name"], n["entity_type"],
                            json.dumps(n.get("aliases", [])),
                            json.dumps(n.get("attributes", {})),
                            n["first_seen"], n["last_seen"],
                            json.dumps(n.get("memory_ids", [])),
                            now,
                        )
                        for n in nodes
                    ],
                )
                self._conn.commit()
        except Exception as e:
            _logger.warning("Failed to save graph nodes incrementally: %s", e)

    def save_graph_edges_incremental(self, edges: list[dict]) -> None:
        """Incrementally save dirty graph edges (INSERT OR REPLACE)."""
        if not edges:
            return
        try:
            now = _now_iso()
            with self._lock:
                self._conn.executemany(
                    """INSERT OR REPLACE INTO graph_edges
                       (source_id, target_id, relation_type, weight,
                        timestamp, metadata, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            e["source_id"], e["target_id"],
                            e["relation_type"], e["weight"],
                            e["timestamp"],
                            json.dumps(e.get("metadata", {})),
                            now,
                        )
                        for e in edges
                    ],
                )
                self._conn.commit()
        except Exception as e:
            _logger.warning("Failed to save graph edges incrementally: %s", e)

    def load_graph_nodes(self) -> list[dict] | None:
        """Load all graph nodes from the graph_nodes table."""
        try:
            with self._lock:
                cursor = self._execute(
                    "SELECT id, name, entity_type, aliases, attributes, "
                    "first_seen, last_seen, memory_ids FROM graph_nodes",
                    (),
                )
                rows = cursor.fetchall()
                if not rows:
                    return None
                return [
                    {
                        "id": r["id"],
                        "name": r["name"],
                        "entity_type": r["entity_type"],
                        "aliases": json.loads(r["aliases"] or "[]"),
                        "attributes": json.loads(r["attributes"] or "{}"),
                        "first_seen": r["first_seen"],
                        "last_seen": r["last_seen"],
                        "memory_ids": json.loads(r["memory_ids"] or "[]"),
                    }
                    for r in rows
                ]
        except Exception as e:
            _logger.warning("Failed to load graph nodes: %s", e)
            return None

    def load_graph_edges(self) -> list[dict] | None:
        """Load all graph edges from the graph_edges table."""
        try:
            with self._lock:
                cursor = self._execute(
                    "SELECT source_id, target_id, relation_type, weight, "
                    "timestamp, metadata FROM graph_edges",
                    (),
                )
                rows = cursor.fetchall()
                if not rows:
                    return None
                return [
                    {
                        "source_id": r["source_id"],
                        "target_id": r["target_id"],
                        "relation_type": r["relation_type"],
                        "weight": r["weight"],
                        "timestamp": r["timestamp"],
                        "metadata": json.loads(r["metadata"] or "{}"),
                    }
                    for r in rows
                ]
        except Exception as e:
            _logger.warning("Failed to load graph edges: %s", e)
            return None

    def cleanup_expired(self) -> int:
        """Delete all memories whose expiration time has passed.

        Returns:
            Number of memories deleted.
        """
        now_str = _now_iso()
        with self._lock:
            id_cur = self._execute(
                "SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now_str,),
            )
            ids_to_delete = [r["id"] for r in id_cur.fetchall()]

            if not ids_to_delete:
                return 0

            placeholders = ",".join("?" for _ in ids_to_delete)
            self._execute(
                f"DELETE FROM memories_fts WHERE id IN ({placeholders})",
                tuple(ids_to_delete),
            )
            if self._vec_available:
                try:
                    self._execute(
                        f"""DELETE FROM memories_vec WHERE rowid IN (
                            SELECT rowid FROM memories WHERE id IN ({placeholders})
                        )""",
                        tuple(ids_to_delete),
                    )
                except Exception as exc:
                    _logger.warning("vec0 delete during cleanup failed (%s).", exc)
            cur = self._execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",
                tuple(ids_to_delete),
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

        Args:
            vectors: List of embedding vectors to check.
            user_id: Filter memories by user. None = all users.
            threshold: Minimum cosine similarity to include.

        Returns:
            Dict mapping input vector index to list of similar Memory objects.
        """
        # Load all active memories with embeddings in a single query
        conditions = ["embedding IS NOT NULL", "is_active = 1"]
        now_str = _now_iso()
        conditions.append("(expires_at IS NULL OR expires_at > ?)")
        params: list[Any] = []
        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)

        where = " AND ".join(conditions)
        cur = self._execute(
            f"SELECT * FROM memories WHERE {where}",
            tuple(params) + (now_str,),
        )
        rows = cur.fetchall()
        if not rows:
            return {i: [] for i in range(len(vectors))}

        # Pre-compute memory embeddings as numpy arrays
        mem_embeddings: list[tuple[Memory, np.ndarray]] = []
        for row in rows:
            emb_blob = row["embedding"]
            if emb_blob:
                mem = _memory_from_row(row)
                mem_embeddings.append((mem, _blob_to_embedding(emb_blob)))

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

    def save_history(
        self,
        memory_id: str,
        old_content: str | None,
        new_content: str,
        event: str = 'UPDATE',
        metadata: dict | None = None,
    ) -> None:
        """Record a memory version change."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO memory_history"
                " (memory_id, old_content, new_content,"
                " event, changed_at, metadata)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    memory_id, old_content, new_content,
                    event, _now_iso(), json.dumps(metadata or {}),
                ),
            )
            self._conn.commit()

    def get_history(self, memory_id: str) -> list[dict[str, object]]:
        """Retrieve the version history for a memory."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, memory_id, old_content,"
                " new_content, event, changed_at,"
                " metadata FROM memory_history"
                " WHERE memory_id = ? ORDER BY id ASC",
                (memory_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "id": row["id"],
                "memory_id": row["memory_id"],
                "old_content": row["old_content"],
                "new_content": row["new_content"],
                "event": row["event"],
                "changed_at": row["changed_at"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            }
            for row in rows
        ]
