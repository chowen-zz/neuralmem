"""SQLite + sqlite-vec 存储后端实现（含 numpy 回退）"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
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
    embedding BLOB
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


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _memory_from_row(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        content=row["content"],
        memory_type=MemoryType(row["memory_type"]),
        scope=MemoryScope(row["scope"]),
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        session_id=row["session_id"],
        tags=tuple(json.loads(row["tags"] or "[]")),
        source=row["source"],
        importance=row["importance"],
        entity_ids=tuple(json.loads(row["entity_ids"] or "[]")),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        last_accessed=datetime.fromisoformat(row["last_accessed"]),
        access_count=row["access_count"],
        embedding=list(_blob_to_embedding(row["embedding"])) if row["embedding"] else None,
    )


class SQLiteStorage(StorageBackend):
    """SQLite + sqlite-vec 存储后端，sqlite-vec 不可用时自动降级为 numpy 暴力搜索"""

    def __init__(self, config: NeuralMemConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
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
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
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
            if self._vec_available:
                try:
                    self._conn.execute(_CREATE_VEC.format(dim=self._dim))
                except Exception as exc:
                    _logger.warning("Failed to create vec0 table (%s). Falling back to numpy.", exc)
                    self._vec_available = False
            self._conn.commit()

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        try:
            return self._conn.execute(sql, params)
        except sqlite3.Error as exc:
            raise StorageError(f"SQLite error: {exc}") from exc

    # ------------------------------------------------------------------
    # StorageBackend 实现
    # ------------------------------------------------------------------

    def save_memory(self, memory: Memory) -> str:
        tags_json = json.dumps(list(memory.tags))
        entity_ids_json = json.dumps(list(memory.entity_ids))
        embedding_blob = _embedding_to_blob(memory.embedding) if memory.embedding else None

        with self._lock:
            self._execute(
                """
                INSERT OR REPLACE INTO memories
                    (id, content, memory_type, scope, user_id, agent_id, session_id,
                     tags, source, importance, entity_ids,
                     created_at, updated_at, last_accessed, access_count, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                    memory.last_accessed.isoformat(),
                    memory.access_count,
                    embedding_blob,
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
                        "DELETE FROM memories_vec WHERE rowid = (SELECT rowid FROM memories WHERE id = ?)",
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
        cur = self._execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return _memory_from_row(row)

    def update_memory(self, memory_id: str, **kwargs: object) -> None:
        allowed = {
            "content", "memory_type", "scope", "user_id", "agent_id", "session_id",
            "tags", "source", "importance", "entity_ids", "embedding",
        }
        updates: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key not in allowed:
                raise StorageError(f"Unknown field for update: {key}")
            if key == "tags":
                updates["tags"] = json.dumps(list(value))  # type: ignore[arg-type]
            elif key == "entity_ids":
                updates["entity_ids"] = json.dumps(list(value))  # type: ignore[arg-type]
            elif key == "embedding":
                updates["embedding"] = _embedding_to_blob(value) if value else None  # type: ignore[arg-type]
            elif key == "memory_type" and isinstance(value, MemoryType):
                updates["memory_type"] = value.value
            elif key == "scope" and isinstance(value, MemoryScope):
                updates["scope"] = value.value
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
        blob = _embedding_to_blob(vector)
        # vec0 KNN 查询，再 JOIN memories 过滤条件
        conditions: list[str] = ["m.embedding IS NOT NULL"]
        params: list[Any] = []
        if user_id is not None:
            conditions.append("m.user_id = ?")
            params.append(user_id)
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            conditions.append(f"m.memory_type IN ({placeholders})")
            params.extend(t.value for t in memory_types)

        where = " AND ".join(conditions)
        try:
            cur = self._execute(
                f"""
                SELECT m.id, v.distance
                FROM memories_vec v
                JOIN memories m ON m.rowid = v.rowid
                WHERE {where}
                ORDER BY v.distance
                LIMIT ?
                """,
                tuple(params) + (limit,),
            )
            rows = cur.fetchall()
            # distance 越小越相似，转换为相似度分数（余弦距离 = 1 - 余弦相似度）
            results = []
            for row in rows:
                dist = row["distance"]
                score = max(0.0, 1.0 - float(dist))
                results.append((row["id"], score))
            return results
        except Exception as exc:
            _logger.warning("vec0 search failed (%s). Falling back to numpy.", exc)
            return self._vector_search_numpy(vector, user_id, memory_types, limit)

    def _vector_search_numpy(
        self,
        vector: list[float],
        user_id: str | None,
        memory_types: list[MemoryType] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        conditions = ["embedding IS NOT NULL"]
        params: list[Any] = []
        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            conditions.append(f"memory_type IN ({placeholders})")
            params.extend(t.value for t in memory_types)

        where = " AND ".join(conditions)
        cur = self._execute(
            f"SELECT id, embedding FROM memories WHERE {where}", tuple(params)
        )
        rows = cur.fetchall()

        query_vec = np.array(vector, dtype=np.float32)
        scored: list[tuple[str, float]] = []
        for row in rows:
            emb = _blob_to_embedding(row["embedding"])
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
        # FTS5 BM25（rank 值为负数，越小越相关）
        join_conditions: list[str] = []
        params: list[Any] = [query]
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
            cur = self._execute(
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
            rows = cur.fetchall()
        except sqlite3.OperationalError:
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

        cur = self._execute(
            f"SELECT id, created_at FROM memories WHERE id IN ({placeholders}) {extra_where}",
            tuple(params),
        )
        rows = cur.fetchall()

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
        cur = self._execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", tuple(similar_ids)
        )
        return [_memory_from_row(row) for row in cur.fetchall()]

    def record_access(self, memory_id: str) -> None:
        with self._lock:
            self._execute(
                "UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
                (_now_iso(), memory_id),
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
            now = datetime.utcnow().isoformat()
            with self._lock:
                self._execute(
                    "INSERT OR REPLACE INTO graph_snapshots (key, data, updated_at) VALUES (?, ?, ?)",
                    ("default", json.dumps(data), now),
                )
                self._conn.commit()
        except Exception as e:
            _logger.warning("Failed to save graph snapshot: %s", e)
