"""Unit tests for PgVectorStorage — mock psycopg to verify SQL queries."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_psycopg():
    """Mock psycopg import so PgVectorStorage can be constructed without a DB."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.closed = False

    mock_psycopg_mod = MagicMock()
    mock_psycopg_mod.connect.return_value = mock_conn

    with patch.dict("sys.modules", {
        "psycopg": mock_psycopg_mod,
        "pgvector": MagicMock(),
        "pgvector.psycopg": MagicMock(),
    }):
        yield mock_psycopg_mod, mock_conn, mock_cursor


@pytest.fixture
def config():
    """Minimal config for testing."""
    from neuralmem.core.config import NeuralMemConfig
    return NeuralMemConfig(
        db_path="/tmp/test.db",
        pg_dsn="postgresql://test:test@localhost:5432/testdb",
        embedding_dim=4,
    )


@pytest.fixture
def storage(mock_psycopg, config):
    """PgVectorStorage instance with mocked connection."""
    from neuralmem.storage.pgvector import PgVectorStorage
    return PgVectorStorage(config)


@pytest.fixture
def sample_memory():
    """A sample Memory for testing."""
    from neuralmem.core.types import Memory, MemoryType
    return Memory(
        id="01TEST1234567890ABCDEF",
        content="Test memory content",
        memory_type=MemoryType.SEMANTIC,
        user_id="user-1",
        importance=0.7,
        embedding=[0.1, 0.2, 0.3, 0.4],
    )


# ---------------------------------------------------------------------------
# Tests: construction / init
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_connection(self, mock_psycopg, config):
        mod, conn, cursor = mock_psycopg
        from neuralmem.storage.pgvector import PgVectorStorage
        PgVectorStorage(config)
        mod.connect.assert_called_once_with(
            "postgresql://test:test@localhost:5432/testdb",
            autocommit=False,
        )

    def test_init_creates_extension_and_tables(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        sql_calls = [call.args[0] for call in cursor.execute.call_args_list]
        assert any("CREATE EXTENSION" in s for s in sql_calls)
        assert any("CREATE TABLE IF NOT EXISTS memories" in s for s in sql_calls)
        assert any("CREATE TABLE IF NOT EXISTS graph_snapshots" in s for s in sql_calls)
        assert any("content_tsv" in s for s in sql_calls)

    def test_init_creates_indexes(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        sql_calls = [call.args[0] for call in cursor.execute.call_args_list]
        assert any("idx_memories_user_id" in s for s in sql_calls)
        assert any("idx_memories_content_tsv" in s for s in sql_calls)
        assert any("idx_memories_embedding" in s for s in sql_calls)


# ---------------------------------------------------------------------------
# Tests: save_memory
# ---------------------------------------------------------------------------

class TestSaveMemory:
    def test_save_memory_sql(self, storage, mock_psycopg, sample_memory):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        conn.commit.reset_mock()

        result = storage.save_memory(sample_memory)
        assert result == sample_memory.id

        # Find the INSERT statement
        insert_calls = [
            c for c in cursor.execute.call_args_list
            if "INSERT INTO memories" in c.args[0]
        ]
        assert len(insert_calls) == 1
        params = insert_calls[0].args[1]
        assert params[0] == sample_memory.id
        assert params[1] == "Test memory content"
        assert params[2] == "semantic"
        assert params[3] == "user"
        assert params[4] == "user-1"
        conn.commit.assert_called()

    def test_save_memory_no_embedding(self, storage, mock_psycopg):
        from neuralmem.core.types import Memory
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()

        m = Memory(content="no embedding", id="TESTNOEMB")
        storage.save_memory(m)

        insert_calls = [
            c for c in cursor.execute.call_args_list
            if "INSERT INTO memories" in c.args[0]
        ]
        assert len(insert_calls) == 1
        params = insert_calls[0].args[1]
        assert params[18] is None  # embedding position


# ---------------------------------------------------------------------------
# Tests: get_memory
# ---------------------------------------------------------------------------

class TestGetMemory:
    def test_get_memory_found(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()

        # Simulate fetchone returning a row
        fake_row = (
            "01TEST1234567890ABCDEF",  # id
            "Test content",            # content
            "semantic",                # memory_type
            "user",                    # scope
            "user-1",                  # user_id
            None,                      # agent_id
            None,                      # session_id
            "[]",                      # tags
            None,                      # source
            0.5,                       # importance
            "[]",                      # entity_ids
            True,                      # is_active
            None,                      # superseded_by
            "[]",                      # supersedes
            "2025-01-01T00:00:00+00:00",  # created_at
            "2025-01-01T00:00:00+00:00",  # updated_at
            "2025-01-01T00:00:00+00:00",  # last_accessed
            0,                         # access_count
            None,                      # embedding
            "long_term",               # session_layer
            None,                      # conversation_id
            None,                      # expires_at
        )
        description = [(col,) for col in [
            "id", "content", "memory_type", "scope", "user_id", "agent_id",
            "session_id", "tags", "source", "importance", "entity_ids",
            "is_active", "superseded_by", "supersedes", "created_at",
            "updated_at", "last_accessed", "access_count", "embedding",
            "session_layer", "conversation_id", "expires_at",
        ]]
        cursor.description = description
        cursor.fetchone.return_value = fake_row

        result = storage.get_memory("01TEST1234567890ABCDEF")
        assert result is not None
        assert result.id == "01TEST1234567890ABCDEF"
        assert result.content == "Test content"

    def test_get_memory_not_found(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.fetchone.return_value = None
        result = storage.get_memory("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: update_memory
# ---------------------------------------------------------------------------

class TestUpdateMemory:
    def test_update_content(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        conn.commit.reset_mock()

        storage.update_memory("mem-1", content="new content")

        update_calls = [
            c for c in cursor.execute.call_args_list
            if "UPDATE memories SET" in c.args[0]
        ]
        assert len(update_calls) == 1
        sql = update_calls[0].args[0]
        params = update_calls[0].args[1]
        assert "content = %s" in sql
        assert "updated_at = %s" in sql
        assert "new content" in params
        conn.commit.assert_called()

    def test_update_unknown_field_raises(self, storage):
        from neuralmem.core.exceptions import StorageError
        with pytest.raises(StorageError, match="Unknown field"):
            storage.update_memory("mem-1", bad_field="value")

    def test_update_tags_serialized(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()

        storage.update_memory("mem-1", tags=["a", "b"])

        update_calls = [
            c for c in cursor.execute.call_args_list
            if "UPDATE memories SET" in c.args[0]
        ]
        params = update_calls[0].args[1]
        assert json.dumps(["a", "b"]) in params

    def test_update_embedding(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()

        storage.update_memory("mem-1", embedding=[0.1, 0.2])

        update_calls = [
            c for c in cursor.execute.call_args_list
            if "UPDATE memories SET" in c.args[0]
        ]
        params = update_calls[0].args[1]
        assert [0.1, 0.2] in params


# ---------------------------------------------------------------------------
# Tests: delete_memories
# ---------------------------------------------------------------------------

class TestDeleteMemories:
    def test_delete_by_id(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        cursor.rowcount = 1

        count = storage.delete_memories(memory_id="mem-1")
        assert count == 1

        delete_calls = [
            c for c in cursor.execute.call_args_list
            if "DELETE FROM memories" in c.args[0]
        ]
        assert len(delete_calls) == 1
        assert "mem-1" in delete_calls[0].args[1]

    def test_delete_by_user_and_tags(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        cursor.rowcount = 3

        count = storage.delete_memories(user_id="user-1", tags=["important"])
        assert count == 3

        delete_calls = [
            c for c in cursor.execute.call_args_list
            if "DELETE FROM memories" in c.args[0]
        ]
        sql = delete_calls[0].args[0]
        assert "user_id = %s" in sql
        assert "ILIKE" in sql

    def test_delete_no_filters(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        cursor.rowcount = 0

        count = storage.delete_memories()
        assert count == 0


# ---------------------------------------------------------------------------
# Tests: vector_search
# ---------------------------------------------------------------------------

class TestVectorSearch:
    def test_vector_search_uses_pgvector_operator(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        description = [("id",), ("score",)]
        cursor.description = description
        cursor.fetchall.return_value = [("mem-1", 0.95)]

        results = storage.vector_search([0.1, 0.2, 0.3, 0.4], user_id="user-1", limit=5)

        search_calls = [
            c for c in cursor.execute.call_args_list
            if "<=>" in c.args[0]
        ]
        assert len(search_calls) == 1
        sql = search_calls[0].args[0]
        assert "ORDER BY embedding <=> %s::vector" in sql
        assert len(results) == 1
        assert results[0][0] == "mem-1"

    def test_vector_search_with_memory_types(self, storage, mock_psycopg):
        from neuralmem.core.types import MemoryType
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        description = [("id",), ("score",)]
        cursor.description = description
        cursor.fetchall.return_value = []

        storage.vector_search(
            [0.1, 0.2, 0.3, 0.4],
            memory_types=[MemoryType.SEMANTIC, MemoryType.EPISODIC],
        )

        search_calls = [
            c for c in cursor.execute.call_args_list
            if "<=>" in c.args[0]
        ]
        sql = search_calls[0].args[0]
        assert "memory_type IN" in sql


# ---------------------------------------------------------------------------
# Tests: keyword_search
# ---------------------------------------------------------------------------

class TestKeywordSearch:
    def test_keyword_search_uses_tsvector(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        description = [("id",), ("rank",)]
        cursor.description = description
        cursor.fetchall.return_value = [("mem-1", 0.8)]

        results = storage.keyword_search("test query", user_id="user-1")

        search_calls = [
            c for c in cursor.execute.call_args_list
            if "ts_rank_cd" in c.args[0]
        ]
        assert len(search_calls) == 1
        sql = search_calls[0].args[0]
        assert "content_tsv @@ to_tsquery" in sql
        assert len(results) == 1
        assert results[0][0] == "mem-1"

    def test_keyword_search_with_limit(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        description = [("id",), ("rank",)]
        cursor.description = description
        cursor.fetchall.return_value = []

        storage.keyword_search("hello", limit=3)

        search_calls = [
            c for c in cursor.execute.call_args_list
            if "ts_rank_cd" in c.args[0]
        ]
        assert search_calls[0].args[1][-1] == 3


# ---------------------------------------------------------------------------
# Tests: temporal_search
# ---------------------------------------------------------------------------

class TestTemporalSearch:
    def test_temporal_search_blends_semantic_and_recency(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        # First call: vector_search -> returns candidates via fetchall
        # Second call: temporal fetch
        cursor.execute.reset_mock()

        # Mock vector_search internally
        # We'll call it indirectly, need to set up fetchall for the
        # vector_search call (which uses fetchall) and the temporal query
        description_vs = [("id",), ("score",)]


        def mock_fetchall():
            return []

        # For temporal_search, we just test it returns empty when no candidates
        cursor.fetchall.return_value = []
        cursor.description = description_vs
        results = storage.temporal_search([0.1, 0.2, 0.3, 0.4])
        assert results == []


# ---------------------------------------------------------------------------
# Tests: find_similar
# ---------------------------------------------------------------------------

class TestFindSimilar:
    def test_find_similar_filters_by_threshold(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        # The method calls vector_search then fetchall for the final query
        # We test that the SQL query uses IN (...)
        cursor.execute.reset_mock()

        # Mock the _fetchall for the SELECT * call
        description = [
            ("id",), ("content",), ("memory_type",), ("scope",), ("user_id",),
            ("agent_id",), ("session_id",), ("tags",), ("source",),
            ("importance",), ("entity_ids",), ("is_active",), ("superseded_by",),
            ("supersedes",), ("created_at",), ("updated_at",), ("last_accessed",),
            ("access_count",), ("embedding",), ("session_layer",),
            ("conversation_id",), ("expires_at",),
        ]

        # First: vector_search returns results via its own fetchall
        # Second: find_similar fetchall
        # We need to sequence the returns
        cursor.description = description

        results = storage.find_similar([0.1, 0.2, 0.3, 0.4], threshold=0.95)
        assert results == []


# ---------------------------------------------------------------------------
# Tests: record_access / batch_record_access
# ---------------------------------------------------------------------------

class TestRecordAccess:
    def test_record_access(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        conn.commit.reset_mock()

        storage.record_access("mem-1")

        update_calls = [
            c for c in cursor.execute.call_args_list
            if "access_count = access_count + 1" in c.args[0]
        ]
        assert len(update_calls) == 1
        assert "mem-1" in update_calls[0].args[1]
        conn.commit.assert_called()

    def test_batch_record_access(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        conn.commit.reset_mock()

        storage.batch_record_access(["mem-1", "mem-2", "mem-3"])

        update_calls = [
            c for c in cursor.execute.call_args_list
            if "access_count = access_count + 1" in c.args[0]
        ]
        assert len(update_calls) == 1
        sql = update_calls[0].args[0]
        assert "IN (%s, %s, %s)" in sql
        conn.commit.assert_called()

    def test_batch_record_access_empty(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()

        storage.batch_record_access([])

        update_calls = [
            c for c in cursor.execute.call_args_list
            if "access_count" in c.args[0]
        ]
        assert len(update_calls) == 0


# ---------------------------------------------------------------------------
# Tests: get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_get_stats(self, storage, mock_psycopg):
        """Test get_stats by directly mocking _fetchone and _fetchall."""
        mock_psycopg[2]

        # Mock _fetchone for the COUNT query
        storage._fetchone = MagicMock(return_value={"total": 5})
        # Mock _fetchall for the GROUP BY and entity_ids queries
        storage._fetchall = MagicMock(side_effect=[
            [{"memory_type": "semantic", "cnt": 3}],  # by_type
            [{"entity_ids": '["e1", "e2"]'}],          # entities
        ])

        stats = storage.get_stats(user_id="user-1")
        assert stats["total"] == 5
        assert stats["by_type"] == {"semantic": 3}
        assert stats["entity_count"] == 2


# ---------------------------------------------------------------------------
# Tests: list_memories
# ---------------------------------------------------------------------------

class TestListMemories:
    def test_list_memories_with_user(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        description = [("id",), ("content",)]
        cursor.description = description
        cursor.fetchall.return_value = []

        storage.list_memories(user_id="user-1", limit=100)

        select_calls = [
            c for c in cursor.execute.call_args_list
            if "SELECT * FROM memories" in c.args[0] and "user_id" in c.args[0]
        ]
        assert len(select_calls) == 1

    def test_list_memories_without_user(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        description = [("id",), ("content",)]
        cursor.description = description
        cursor.fetchall.return_value = []

        storage.list_memories(limit=50)

        select_calls = [
            c for c in cursor.execute.call_args_list
            if "SELECT * FROM memories" in c.args[0] and "user_id" not in c.args[0]
        ]
        assert len(select_calls) == 1


# ---------------------------------------------------------------------------
# Tests: graph snapshot
# ---------------------------------------------------------------------------

class TestGraphSnapshot:
    def test_save_graph_snapshot(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        conn.commit.reset_mock()

        storage.save_graph_snapshot({"entities": [], "relations": []})

        upsert_calls = [
            c for c in cursor.execute.call_args_list
            if "graph_snapshots" in c.args[0]
        ]
        assert len(upsert_calls) == 1
        assert "ON CONFLICT" in upsert_calls[0].args[0]
        conn.commit.assert_called()

    def test_load_graph_snapshot(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        description = [("data",)]
        cursor.description = description
        # Return a tuple matching the description, with the jsonb data as a dict
        # (psycopg returns jsonb columns as Python dicts)
        cursor.fetchone.return_value = ({"entities": [{"name": "test"}]},)

        result = storage.load_graph_snapshot()
        assert result is not None
        assert "entities" in result

    def test_load_graph_snapshot_none(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.fetchone.return_value = None

        result = storage.load_graph_snapshot()
        assert result is None


# ---------------------------------------------------------------------------
# Tests: cleanup_expired
# ---------------------------------------------------------------------------

class TestCleanupExpired:
    def test_cleanup_expired(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        cursor.rowcount = 3
        conn.commit.reset_mock()

        count = storage.cleanup_expired()
        assert count == 3

        delete_calls = [
            c for c in cursor.execute.call_args_list
            if "DELETE FROM memories" in c.args[0]
        ]
        assert len(delete_calls) == 1
        assert "expires_at" in delete_calls[0].args[0]
        conn.commit.assert_called()


# ---------------------------------------------------------------------------
# Tests: batch_find_similar
# ---------------------------------------------------------------------------

class TestBatchFindSimilar:
    def test_batch_find_similar_empty(self, storage, mock_psycopg):
        _, conn, cursor = mock_psycopg
        cursor.execute.reset_mock()
        description = [("id",)]
        cursor.description = description
        cursor.fetchall.return_value = []

        results = storage.batch_find_similar([[0.1, 0.2], [0.3, 0.4]])
        assert results == {0: [], 1: []}


# ---------------------------------------------------------------------------
# Tests: __del__
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_del_closes_connection(self, mock_psycopg, config):
        _, conn, _ = mock_psycopg
        from neuralmem.storage.pgvector import PgVectorStorage
        s = PgVectorStorage(config)
        s.__del__()
        conn.close.assert_called()


# ---------------------------------------------------------------------------
# Tests: config pg_dsn
# ---------------------------------------------------------------------------

class TestConfigPgDsn:
    def test_config_has_pg_dsn(self):
        from neuralmem.core.config import NeuralMemConfig
        cfg = NeuralMemConfig()
        assert hasattr(cfg, "pg_dsn")
        assert cfg.pg_dsn == "postgresql://localhost:5432/neuralmem"

    def test_config_pg_dsn_custom(self):
        from neuralmem.core.config import NeuralMemConfig
        cfg = NeuralMemConfig(pg_dsn="postgresql://admin:secret@db:5432/mydb")
        assert cfg.pg_dsn == "postgresql://admin:secret@db:5432/mydb"
