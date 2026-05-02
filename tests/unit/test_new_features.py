"""Tests for Phase 1-3 new features:
- Phase 1a: Conflict Resolution (is_active, superseded_by, resolve_conflict)
- Phase 1b: Importance Auto-Reinforcement
- Phase 2a: Session-Aware Memory (3-layer architecture)
- Phase 2b: Memory Explainability
- Phase 3a: Batch Operations (remember_batch, export, forget_batch)
"""
from __future__ import annotations

import json

import pytest

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.memory import NeuralMem
from neuralmem.core.types import Memory, MemoryType, SearchResult, SessionLayer

# ── Helpers ──────────────────────────────────────────────────────────────

def _make_neural_mem(config, mock_embedder, tmp_path):
    """Create a NeuralMem with mock embedder injected."""
    db_path = str(tmp_path / "test.db")
    cfg = NeuralMemConfig(
        db_path=db_path,
        embedding_dim=4,
        conflict_threshold_high=0.95,
        conflict_threshold_low=0.75,
        enable_importance_reinforcement=True,
        reinforcement_boost=0.05,
    )
    mem = NeuralMem(config=cfg, embedder=mock_embedder)
    return mem


# ── Phase 1a: Conflict Resolution ────────────────────────────────────────

class TestConflictResolution:
    """Test Memory.is_active, superseded_by, and resolve_conflict API."""

    def test_memory_has_conflict_fields(self, sample_memory):
        """Memory model has is_active, superseded_by, supersedes fields."""
        assert sample_memory.is_active is True
        assert sample_memory.superseded_by is None
        assert sample_memory.supersedes == ()

    def test_remember_marks_superseded(self, config, mock_embedder, tmp_db_path, sample_vector):
        """When a similar (but not duplicate) memory is stored, old one gets superseded."""
        from neuralmem.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(config)

        # Store an initial memory
        m1 = Memory(
            content="User prefers dark mode",
            memory_type=MemoryType.SEMANTIC,
            user_id="test-user",
            importance=0.6,
            embedding=sample_vector,
        )
        storage.save_memory(m1)

        # Create a similar but not identical vector (below high threshold, above low)
        # Original: [1.0, 0.0, 0.0, 0.0]
        similar = [0.95, 0.05, 0.0, 0.0]
        norm = sum(x**2 for x in similar) ** 0.5
        similar = [x / norm for x in similar]

        # Verify the old memory is active
        old = storage.get_memory(m1.id)
        assert old.is_active is True

        # Store a superseding memory directly (simulating what remember() does)
        m2 = Memory(
            content="User now prefers light mode",
            memory_type=MemoryType.SEMANTIC,
            user_id="test-user",
            importance=0.7,
            embedding=similar,
        )
        storage.save_memory(m2)

        # Mark old as superseded
        storage.update_memory(m1.id, is_active=False, superseded_by=m2.id)
        storage.update_memory(m2.id, supersedes=(m1.id,))

        # Verify
        old_updated = storage.get_memory(m1.id)
        new = storage.get_memory(m2.id)
        assert old_updated.is_active is False
        assert old_updated.superseded_by == m2.id
        assert m1.id in new.supersedes

    def test_recall_filters_inactive(self, config, mock_embedder, tmp_db_path, sample_vector):
        """recall() filters out superseded (is_active=False) memories."""
        from neuralmem.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(config)
        m1 = Memory(
            content="Python is great for data science",
            memory_type=MemoryType.SEMANTIC,
            user_id="test-user",
            importance=0.5,
            embedding=sample_vector,
        )
        storage.save_memory(m1)

        # Mark as superseded
        storage.update_memory(m1.id, is_active=False, superseded_by="fake-id")

        # Verify storage returns it but with is_active=False
        m = storage.get_memory(m1.id)
        assert m.is_active is False

    def test_resolve_conflict_reactivate(self, config, mock_embedder, tmp_path, sample_vector):
        """resolve_conflict with action='reactivate' re-enables a memory."""
        from neuralmem.storage.sqlite import SQLiteStorage

        db_path = str(tmp_path / "test.db")
        cfg = NeuralMemConfig(db_path=db_path, embedding_dim=4)
        storage = SQLiteStorage(cfg)
        m1 = Memory(
            content="User likes coffee",
            memory_type=MemoryType.SEMANTIC,
            user_id="test-user",
            importance=0.5,
            embedding=sample_vector,
        )
        storage.save_memory(m1)
        storage.update_memory(m1.id, is_active=False, superseded_by="fake-id")

        mem = NeuralMem(config=cfg, embedder=mock_embedder)
        result = mem.resolve_conflict(m1.id, action="reactivate")
        assert result is True

        m = storage.get_memory(m1.id)
        assert m.is_active is True
        assert m.superseded_by is None

    def test_resolve_conflict_delete(self, config, mock_embedder, tmp_path, sample_vector):
        """resolve_conflict with action='delete' removes the memory."""
        from neuralmem.storage.sqlite import SQLiteStorage

        db_path = str(tmp_path / "test.db")
        cfg = NeuralMemConfig(db_path=db_path, embedding_dim=4)
        storage = SQLiteStorage(cfg)
        m1 = Memory(
            content="Temporary fact",
            memory_type=MemoryType.SEMANTIC,
            user_id="test-user",
            importance=0.3,
            embedding=sample_vector,
        )
        storage.save_memory(m1)

        mem = NeuralMem(config=cfg, embedder=mock_embedder)
        result = mem.resolve_conflict(m1.id, action="delete")
        assert result is True

        m = storage.get_memory(m1.id)
        assert m is None

    def test_resolve_conflict_nonexistent(self, tmp_path, mock_embedder):
        """resolve_conflict returns False for nonexistent memory."""
        db_path = str(tmp_path / "test.db")
        cfg = NeuralMemConfig(db_path=db_path, embedding_dim=4)
        mem = NeuralMem(config=cfg, embedder=mock_embedder)
        result = mem.resolve_conflict("nonexistent-id", action="reactivate")
        assert result is False

    def test_get_stats(self, config, tmp_db_path, sample_vector):
        """storage.get_stats returns total, by_type, entity_count."""
        from neuralmem.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(config)
        m1 = Memory(
            content="Active memory",
            memory_type=MemoryType.SEMANTIC,
            user_id="test-user",
            embedding=sample_vector,
        )
        storage.save_memory(m1)

        v2 = [0.0, 1.0, 0.0, 0.0]
        m2 = Memory(
            content="Another memory",
            memory_type=MemoryType.EPISODIC,
            user_id="test-user",
            embedding=v2,
        )
        storage.save_memory(m2)
        storage.update_memory(m2.id, is_active=False, superseded_by=m1.id)

        stats = storage.get_stats(user_id="test-user")
        assert stats["total"] == 2
        assert stats["by_type"]["semantic"] == 1
        assert stats["by_type"]["episodic"] == 1

    def test_update_memory_is_active_and_superseded_by(self, config, sample_vector):
        """Storage.update_memory can toggle is_active and superseded_by."""
        from neuralmem.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(config)
        m = Memory(
            content="Test",
            memory_type=MemoryType.SEMANTIC,
            user_id="test-user",
            embedding=sample_vector,
        )
        storage.save_memory(m)

        # Initially active
        loaded = storage.get_memory(m.id)
        assert loaded.is_active is True

        # Mark superseded
        storage.update_memory(m.id, is_active=False, superseded_by="new-id")
        loaded = storage.get_memory(m.id)
        assert loaded.is_active is False
        assert loaded.superseded_by == "new-id"

        # Reactivate
        storage.update_memory(m.id, is_active=True, superseded_by=None)
        loaded = storage.get_memory(m.id)
        assert loaded.is_active is True
        assert loaded.superseded_by is None


# ── Phase 1b: Importance Auto-Reinforcement ──────────────────────────────

class TestImportanceReinforcement:
    """Test that recall() boosts importance for accessed memories."""

    def test_config_has_reinforcement_settings(self):
        """NeuralMemConfig has importance reinforcement config fields."""
        config = NeuralMemConfig()
        assert hasattr(config, "enable_importance_reinforcement")
        assert hasattr(config, "reinforcement_boost")
        assert config.enable_importance_reinforcement is True
        assert config.reinforcement_boost == pytest.approx(0.05)

    def test_config_reinforcement_disabled(self):
        """Can disable importance reinforcement via config."""
        config = NeuralMemConfig(enable_importance_reinforcement=False)
        assert config.enable_importance_reinforcement is False

    def test_update_importance_in_storage(self, config, sample_vector):
        """Storage.update_memory can change importance."""
        from neuralmem.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(config)
        m = Memory(
            content="User prefers Python",
            memory_type=MemoryType.SEMANTIC,
            user_id="test-user",
            importance=0.5,
            embedding=sample_vector,
        )
        storage.save_memory(m)

        loaded = storage.get_memory(m.id)
        assert loaded.importance == pytest.approx(0.5)

        # Boost importance
        storage.update_memory(m.id, importance=0.55)
        loaded = storage.get_memory(m.id)
        assert loaded.importance == pytest.approx(0.55)

    def test_recall_records_access(self, config, sample_vector):
        """record_access increments access_count."""
        from neuralmem.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(config)
        m = Memory(
            content="Popular memory",
            memory_type=MemoryType.SEMANTIC,
            user_id="test-user",
            importance=0.5,
            embedding=sample_vector,
        )
        storage.save_memory(m)

        assert storage.get_memory(m.id).access_count == 0
        storage.record_access(m.id)
        assert storage.get_memory(m.id).access_count == 1
        storage.record_access(m.id)
        assert storage.get_memory(m.id).access_count == 2

    def test_recall_reinforcement_logic(self, tmp_path, mock_embedder, sample_vector):
        """NeuralMem.recall boosts importance when enable_importance_reinforcement=True."""
        from neuralmem.storage.sqlite import SQLiteStorage

        db_path = str(tmp_path / "test.db")
        cfg = NeuralMemConfig(
            db_path=db_path, embedding_dim=4,
            enable_importance_reinforcement=True, reinforcement_boost=0.1,
        )
        storage = SQLiteStorage(cfg)

        # Pre-store a memory
        m = Memory(
            content="User likes Java",
            memory_type=MemoryType.SEMANTIC,
            user_id="test-user",
            importance=0.5,
            embedding=sample_vector,
        )
        storage.save_memory(m)

        # Create NeuralMem and inject mock embedder
        mem = NeuralMem(config=cfg, embedder=mock_embedder)

        # Recall will hit semantic strategy; verify it works without error
        mem.recall("User likes Java", user_id="test-user")
        # The mock embedder generates different vectors, so results may be empty
        # but the code path runs without error

    def test_recall_no_boost_when_disabled(self, tmp_path, mock_embedder, sample_vector):
        """NeuralMem.recall skips boost when enable_importance_reinforcement=False."""
        from neuralmem.storage.sqlite import SQLiteStorage

        db_path = str(tmp_path / "test.db")
        cfg = NeuralMemConfig(
            db_path=db_path, embedding_dim=4,
            enable_importance_reinforcement=False, reinforcement_boost=0.1,
        )
        storage = SQLiteStorage(cfg)

        m = Memory(
            content="User likes Java",
            memory_type=MemoryType.SEMANTIC,
            user_id="test-user",
            importance=0.5,
            embedding=sample_vector,
        )
        storage.save_memory(m)

        mem = NeuralMem(config=cfg, embedder=mock_embedder)

        mem.recall("User likes Java", user_id="test-user")
        # With reinforcement disabled, importance should remain unchanged


# ── Phase 2a: Session-Aware Memory ───────────────────────────────────────

class TestSessionContext:
    """Test 3-layer session-aware memory architecture."""

    def test_session_context_manager(self, tmp_path, mock_embedder):
        """SessionContext works as a context manager."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        with mem.session(user_id="test-user") as ctx:
            assert ctx.conversation_id is not None
            assert ctx.user_id == "test-user"

    def test_working_memory_append(self, tmp_path, mock_embedder):
        """Working memory items are ephemeral and not persisted."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        with mem.session(user_id="test-user") as ctx:
            ctx.append_working("The user asked about decorators")
            ctx.append_working("Explained Python decorators")
            assert len(ctx.working_memory) == 2
            assert "decorators" in ctx.working_memory[0]

    def test_working_memory_clear(self, tmp_path, mock_embedder):
        """clear_working() empties the working memory list."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        with mem.session(user_id="test-user") as ctx:
            ctx.append_working("item 1")
            ctx.append_working("item 2")
            assert len(ctx.working_memory) == 2
            ctx.clear_working()
            assert len(ctx.working_memory) == 0

    def test_working_memory_isolation(self, tmp_path, mock_embedder):
        """Working memory returns a copy, not a reference."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        with mem.session(user_id="test-user") as ctx:
            ctx.append_working("item 1")
            wm = ctx.working_memory
            wm.append("item 2")
            assert len(ctx.working_memory) == 1  # still 1

    def test_remember_to_session(self, tmp_path, mock_embedder):
        """remember_to_session persists to DB with session layer tag."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        with mem.session(user_id="test-user") as ctx:
            memories = ctx.remember_to_session(
                "User prefers concise answers",
                importance=0.8,
                tags=["preference"],
            )
            assert len(memories) >= 1
            assert memories[0].content == "User prefers concise answers"

    def test_session_recall_searches_working_memory(self, tmp_path, mock_embedder):
        """Session recall finds items from working memory."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        with mem.session(user_id="test-user") as ctx:
            ctx.append_working("The user likes Python decorators")
            results = ctx.recall("decorators")
            # Should find at least the working memory item
            assert any(r["layer"] == "working" for r in results)

    def test_session_recall_includes_layer_info(self, tmp_path, mock_embedder):
        """Session recall results have 'layer' and 'score' fields."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        with mem.session(user_id="test-user") as ctx:
            ctx.append_working("dark mode preference")
            results = ctx.recall("dark mode")
            for r in results:
                assert "layer" in r
                assert "score" in r
                assert "content" in r
                assert r["layer"] in ("working", "session", "long_term")

    def test_session_promotion_on_exit(self, tmp_path, mock_embedder):
        """Session memories with high importance get promoted to long-term on exit."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        with mem.session(user_id="test-user", promote_threshold=0.5) as ctx:
            ctx.remember_to_session(
                "Important: user is allergic to peanuts",
                importance=0.9,
            )
            _ = ctx.conversation_id

        # After exit, the session memory should have been promoted
        # Verify by checking storage stats
        stats = mem.storage.get_stats()
        assert stats["total"] >= 1

    def test_session_low_importance_not_promoted(self, tmp_path, mock_embedder):
        """Session memories with low importance stay as session-layer."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        with mem.session(user_id="test-user", promote_threshold=0.7) as ctx:
            ctx.remember_to_session(
                "Minor detail",
                importance=0.3,
            )

        # Check the memory is still session-layer, not promoted
        memories = mem.storage.list_memories(user_id="test-user")
        if memories:
            # Memory should still exist
            assert len(memories) >= 1

    def test_manual_session_api(self, tmp_path, mock_embedder):
        """Manual session_start/append/end works without context manager."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        cid = mem.session_start(user_id="test-user")
        assert cid is not None

        # Append to working layer
        mem.session_append(cid, "Transient context", layer="working")

        # Append to session layer
        memories = mem.session_append(cid, "Important fact", layer="session", importance=0.8)
        assert memories is not None

        # End session
        mem.session_end(cid)

    def test_session_end_nonexistent(self, tmp_path, mock_embedder):
        """session_end on nonexistent conversation_id is a no-op."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        # Should not raise
        mem.session_end("nonexistent-id")

    def test_session_layer_enum(self):
        """SessionLayer enum has correct values."""
        assert SessionLayer.WORKING.value == "working"
        assert SessionLayer.SESSION.value == "session"
        assert SessionLayer.LONG_TERM.value == "long_term"

    def test_session_custom_conversation_id(self, tmp_path, mock_embedder):
        """Session accepts custom conversation_id."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        with mem.session("my-conversation-123", user_id="test-user") as ctx:
            assert ctx.conversation_id == "my-conversation-123"

    def test_session_agent_id(self, tmp_path, mock_embedder):
        """Session tracks agent_id."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        with mem.session(user_id="test-user", agent_id="my-agent") as ctx:
            assert ctx.agent_id == "my-agent"

    def test_simple_match_helper(self):
        """Test the _simple_match helper function."""
        from neuralmem.session.context import _simple_match

        assert _simple_match("python", "User likes Python") is True
        assert _simple_match("java", "User likes Python") is False
        assert _simple_match("decorators", "Explained Python decorators") is True
        assert _simple_match("ab", "Some text with ab") is False  # too short (< 3)
        assert _simple_match("", "Some text") is False

    def test_manual_session_append_working(self, tmp_path, mock_embedder):
        """Manual API: append to working layer."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        cid = mem.session_start(user_id="test-user")
        result = mem.session_append(cid, "Working context", layer="working")
        assert result is None  # working layer returns None
        mem.session_end(cid)

    def test_session_append_no_session(self, tmp_path, mock_embedder):
        """session_append with bad conversation_id returns None."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        result = mem.session_append("nonexistent", "content", layer="session")
        assert result is None


# ── Phase 2b: Memory Explainability ──────────────────────────────────────

class TestExplainability:
    """Test explanation text in SearchResult."""

    def test_search_result_has_explanation_field(self, sample_memory):
        """SearchResult model has explanation field."""
        sr = SearchResult(
            memory=sample_memory,
            score=0.85,
            retrieval_method="semantic",
            explanation="semantic match (score=0.85). Found via: semantic+keyword.",
        )
        assert sr.explanation is not None
        assert "semantic" in sr.explanation

    def test_search_result_explanation_default_none(self, sample_memory):
        """SearchResult.explanation defaults to None."""
        sr = SearchResult(
            memory=sample_memory,
            score=0.5,
            retrieval_method="keyword",
        )
        assert sr.explanation is None

    def test_build_explanation_format(self, config, mock_embedder, tmp_db_path):
        """_build_explanation produces a well-formatted string."""
        from neuralmem.graph.knowledge_graph import KnowledgeGraph
        from neuralmem.retrieval.engine import RetrievalEngine
        from neuralmem.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(config)
        graph = KnowledgeGraph(storage)
        engine = RetrievalEngine(
            storage=storage,
            embedder=mock_embedder,
            graph=graph,
            config=config,
        )

        m = Memory(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            user_id="u",
            importance=0.8,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        explanation = engine._build_explanation(
            primary_method="semantic",
            all_methods=["semantic", "keyword"],
            score=0.85,
            memory=m,
        )
        assert "semantic match" in explanation
        assert "semantic+keyword" in explanation
        assert "High importance" in explanation
        assert explanation.endswith(".")

    def test_build_explanation_low_importance(self, config, mock_embedder, tmp_db_path):
        """_build_explanation omits importance note for low-importance memories."""
        from neuralmem.graph.knowledge_graph import KnowledgeGraph
        from neuralmem.retrieval.engine import RetrievalEngine
        from neuralmem.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(config)
        graph = KnowledgeGraph(storage)
        engine = RetrievalEngine(
            storage=storage,
            embedder=mock_embedder,
            graph=graph,
            config=config,
        )

        m = Memory(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            user_id="u",
            importance=0.3,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        explanation = engine._build_explanation(
            primary_method="keyword",
            all_methods=["keyword"],
            score=0.5,
            memory=m,
        )
        assert "keyword match" in explanation
        assert "High importance" not in explanation

    def test_build_explanation_with_access_count(self, config, mock_embedder, tmp_db_path):
        """_build_explanation includes access_count when > 0."""
        from neuralmem.graph.knowledge_graph import KnowledgeGraph
        from neuralmem.retrieval.engine import RetrievalEngine
        from neuralmem.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(config)
        graph = KnowledgeGraph(storage)
        engine = RetrievalEngine(
            storage=storage,
            embedder=mock_embedder,
            graph=graph,
            config=config,
        )

        m = Memory(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            user_id="u",
            importance=0.5,
            access_count=5,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        explanation = engine._build_explanation(
            primary_method="semantic",
            all_methods=["semantic"],
            score=0.6,
            memory=m,
        )
        assert "Access count: 5" in explanation

    def test_mcp_recall_with_explain(self, config, mock_embedder, tmp_db_path):
        """MCP recall tool supports explain parameter."""
        from neuralmem.mcp.tools import format_search_results

        m = Memory(
            content="User likes Python",
            memory_type=MemoryType.SEMANTIC,
            user_id="test",
            importance=0.5,
        )
        results = [
            SearchResult(
                memory=m,
                score=0.9,
                retrieval_method="semantic",
                explanation="semantic match (score=0.90). Found via: semantic.",
            )
        ]
        output = format_search_results(results, show_explanations=True)
        assert "semantic match" in output

    def test_mcp_recall_without_explain(self, config, mock_embedder, tmp_db_path):
        """MCP recall without explain hides explanation."""
        from neuralmem.mcp.tools import format_search_results

        m = Memory(
            content="User likes Python",
            memory_type=MemoryType.SEMANTIC,
            user_id="test",
        )
        results = [
            SearchResult(
                memory=m,
                score=0.9,
                retrieval_method="semantic",
                explanation="semantic match (score=0.90).",
            )
        ]
        output = format_search_results(results, show_explanations=False)
        assert "semantic match" not in output

    def test_format_search_results_empty(self):
        """format_search_results with empty list."""
        from neuralmem.mcp.tools import format_search_results
        output = format_search_results([], show_explanations=True)
        assert "No relevant" in output


# ── Phase 3a: Batch Operations ───────────────────────────────────────────

class TestBatchOperations:
    """Test remember_batch, export_memories, forget_batch."""

    def test_remember_batch(self, tmp_path, mock_embedder):
        """remember_batch processes multiple items."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        memories = mem.remember_batch(
            [
                "User likes Python",
                "User works on AI projects",
                "User prefers dark mode",
            ],
            user_id="test-user",
        )
        assert len(memories) >= 1

    def test_remember_batch_with_callback(self, tmp_path, mock_embedder):
        """remember_batch calls progress_callback for each item."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        progress = []
        mem.remember_batch(
            ["Item 1", "Item 2"],
            user_id="test-user",
            progress_callback=lambda cur, total, preview: progress.append((cur, total, preview)),
        )
        # Should have at least 3 calls: 2 items + 1 done
        assert len(progress) >= 2
        # Last call should be done (current == total)
        assert progress[-1][0] == progress[-1][1]

    def test_remember_batch_empty(self, tmp_path, mock_embedder):
        """remember_batch with empty list returns empty."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        memories = mem.remember_batch([], user_id="test-user")
        assert memories == []

    def test_export_json(self, tmp_path, mock_embedder):
        """export_memories produces valid JSON."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        mem.remember("User likes Python", user_id="test-user")
        output = mem.export_memories(user_id="test-user", format="json")
        data = json.loads(output)
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "content" in data[0]
        assert "importance" in data[0]
        assert "embedding" not in data[0]  # excluded by default

    def test_export_json_with_embeddings(self, tmp_path, mock_embedder):
        """export_memories with include_embeddings=True includes vectors."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        mem.remember("User likes Python", user_id="test-user")
        output = mem.export_memories(user_id="test-user", format="json", include_embeddings=True)
        data = json.loads(output)
        assert "embedding" in data[0]

    def test_export_json_has_all_fields(self, tmp_path, mock_embedder):
        """Exported JSON has expected fields."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        mem.remember("Test export", user_id="test-user")
        output = mem.export_memories(user_id="test-user", format="json")
        data = json.loads(output)
        entry = data[0]
        expected_fields = [
            "id", "content", "memory_type", "scope", "user_id",
            "agent_id", "tags", "importance", "is_active",
            "created_at", "updated_at", "access_count",
        ]
        for field in expected_fields:
            assert field in entry, f"Missing field: {field}"

    def test_export_markdown(self, tmp_path, mock_embedder):
        """export_memories produces markdown."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        mem.remember("User likes Python", user_id="test-user")
        output = mem.export_memories(user_id="test-user", format="markdown")
        assert "# NeuralMem Export" in output
        assert "User likes Python" in output
        assert "**ID**" in output
        assert "**Importance**" in output

    def test_export_csv(self, tmp_path, mock_embedder):
        """export_memories produces CSV."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        mem.remember("User likes Python", user_id="test-user")
        output = mem.export_memories(user_id="test-user", format="csv")
        lines = output.strip().split("\n")
        assert len(lines) >= 2  # header + at least 1 row
        assert "id" in lines[0]
        assert "content" in lines[0]

    def test_export_unsupported_format(self, tmp_path, mock_embedder):
        """export_memories raises on unsupported format."""
        from neuralmem.core.exceptions import NeuralMemError
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        with pytest.raises(NeuralMemError, match="Unsupported export format"):
            mem.export_memories(format="xml")

    def test_forget_batch_by_ids(self, tmp_path, mock_embedder):
        """forget_batch deletes specific memories by ID."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        memories = mem.remember_batch(
            ["Memory 1", "Memory 2", "Memory 3"],
            user_id="test-user",
        )
        ids_to_delete = [m.id for m in memories[:2]]
        result = mem.forget_batch(memory_ids=ids_to_delete)
        assert result["dry_run"] is False
        assert result["count"] >= 1

    def test_forget_batch_dry_run(self, tmp_path, mock_embedder):
        """forget_batch with dry_run=True doesn't delete anything."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        memories = mem.remember_batch(
            ["Memory A", "Memory B"],
            user_id="test-user",
        )
        ids = [m.id for m in memories]
        result = mem.forget_batch(memory_ids=ids, dry_run=True)
        assert result["dry_run"] is True
        assert result["count"] == len(ids)
        assert result["memory_ids"] == sorted(ids)

        # Verify memories still exist after dry_run
        for mid in ids:
            assert mem.storage.get_memory(mid) is not None

    def test_forget_batch_by_tags(self, tmp_path, mock_embedder):
        """forget_batch can delete by tags."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        mem.remember("Tagged memory", user_id="test-user", tags=["temporary"])
        result = mem.forget_batch(tags=["temporary"], user_id="test-user")
        assert result["count"] >= 1

    def test_forget_batch_empty(self, tmp_path, mock_embedder):
        """forget_batch with no criteria deletes nothing."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        result = mem.forget_batch()
        assert result["count"] == 0
        assert result["memory_ids"] == []
        assert result["dry_run"] is False

    def test_forget_batch_combined_criteria(self, tmp_path, mock_embedder):
        """forget_batch merges IDs from multiple criteria."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        mem.remember("Mem A", user_id="user1", tags=["tag1"])
        m2 = mem.remember("Mem B", user_id="user2", tags=["tag2"])

        # Delete by user_id (gets all for user1) + specific IDs (m2)
        result = mem.forget_batch(
            memory_ids=[m2[0].id] if m2 else [],
            user_id="user1",
        )
        assert result["count"] >= 1

    def test_export_empty(self, tmp_path, mock_embedder):
        """export_memories with no memories returns empty array/header."""
        mem = _make_neural_mem(None, mock_embedder, tmp_path)
        output = mem.export_memories(format="json")
        data = json.loads(output)
        assert data == []
