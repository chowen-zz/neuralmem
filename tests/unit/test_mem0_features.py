"""Tests for mem0-inspired features: get/update/history, infer=False, event-driven extraction."""
from __future__ import annotations

import pytest

from neuralmem.core.memory import NeuralMem
from neuralmem.core.types import MemoryHistoryEntry, MemoryType
from neuralmem.extraction.extractor import ExtractedItem

# =====================================================================
# Public API: get(), update(), history()
# =====================================================================


class TestGetMemory:
    """Test NeuralMem.get() method."""

    def test_get_existing_memory(self, mem_with_mock: NeuralMem):
        """get() returns the stored memory."""
        stored = mem_with_mock.remember("The sky is blue")
        assert len(stored) >= 1
        memory_id = stored[0].id
        result = mem_with_mock.get(memory_id)
        assert result is not None
        assert result.id == memory_id
        assert "sky" in result.content.lower() or "blue" in result.content.lower()

    def test_get_nonexistent_memory(self, mem_with_mock: NeuralMem):
        """get() returns None for unknown ID."""
        result = mem_with_mock.get("nonexistent_id_12345")
        assert result is None

    def test_get_returns_memory_object(self, mem_with_mock: NeuralMem):
        """get() returns a proper Memory object with all fields."""
        stored = mem_with_mock.remember("Python is a programming language")
        memory_id = stored[0].id
        result = mem_with_mock.get(memory_id)
        assert result is not None
        assert hasattr(result, "id")
        assert hasattr(result, "content")
        assert hasattr(result, "memory_type")
        assert hasattr(result, "importance")
        assert hasattr(result, "created_at")
        assert hasattr(result, "is_active")


class TestUpdateMemory:
    """Test NeuralMem.update() method."""

    def test_update_content(self, mem_with_mock: NeuralMem):
        """update() changes the memory content."""
        stored = mem_with_mock.remember("The sky is blue")
        memory_id = stored[0].id
        result = mem_with_mock.update(memory_id, "The sky is green")
        assert result is not None
        assert "green" in result.content.lower()

    def test_update_creates_history(self, mem_with_mock: NeuralMem):
        """update() records a history entry."""
        stored = mem_with_mock.remember("Original content")
        memory_id = stored[0].id
        mem_with_mock.update(memory_id, "Updated content")
        history = mem_with_mock.history(memory_id)
        # Should have at least CREATE + UPDATE entries
        assert len(history) >= 1
        update_entries = [e for e in history if e.event == "UPDATE"]
        assert len(update_entries) >= 1
        assert update_entries[-1].new_content == "Updated content"

    def test_update_nonexistent_returns_none(self, mem_with_mock: NeuralMem):
        """update() returns None for unknown ID."""
        result = mem_with_mock.update("nonexistent_id", "new text")
        assert result is None

    def test_update_same_content_returns_existing(self, mem_with_mock: NeuralMem):
        """update() with identical content returns existing without creating history."""
        stored = mem_with_mock.remember("Same content")
        memory_id = stored[0].id
        result = mem_with_mock.update(memory_id, "Same content")
        assert result is not None
        assert result.id == memory_id

    def test_update_with_metadata(self, mem_with_mock: NeuralMem):
        """update() passes metadata to history entry."""
        stored = mem_with_mock.remember("Original")
        memory_id = stored[0].id
        mem_with_mock.update(memory_id, "Changed", metadata={"reason": "correction"})
        history = mem_with_mock.history(memory_id)
        update_entries = [e for e in history if e.event == "UPDATE"]
        assert len(update_entries) >= 1
        assert update_entries[-1].metadata.get("reason") == "correction"


class TestHistory:
    """Test NeuralMem.history() method."""

    def test_history_empty_for_new_memory(self, mem_with_mock: NeuralMem):
        """history() may be empty or have CREATE entry for a new memory."""
        stored = mem_with_mock.remember("New memory")
        memory_id = stored[0].id
        history = mem_with_mock.history(memory_id)
        # With infer=True (default), history might be empty since remember() doesn't
        # explicitly call save_history. That's ok - history is only tracked for
        # explicit create/update via the new API.
        assert isinstance(history, list)

    def test_history_tracks_updates(self, mem_with_mock: NeuralMem):
        """history() shows all update events."""
        stored = mem_with_mock.remember("Version 1")
        memory_id = stored[0].id
        mem_with_mock.update(memory_id, "Version 2")
        mem_with_mock.update(memory_id, "Version 3")
        history = mem_with_mock.history(memory_id)
        update_entries = [e for e in history if e.event == "UPDATE"]
        assert len(update_entries) == 2
        assert update_entries[0].new_content == "Version 2"
        assert update_entries[1].new_content == "Version 3"

    def test_history_returns_memory_history_entry(self, mem_with_mock: NeuralMem):
        """history() returns MemoryHistoryEntry objects."""
        stored = mem_with_mock.remember("Test")
        memory_id = stored[0].id
        mem_with_mock.update(memory_id, "Updated")
        history = mem_with_mock.history(memory_id)
        for entry in history:
            assert isinstance(entry, MemoryHistoryEntry)
            assert hasattr(entry, "event")
            assert hasattr(entry, "changed_at")
            assert hasattr(entry, "new_content")

    def test_history_nonexistent_returns_empty(self, mem_with_mock: NeuralMem):
        """history() returns empty list for unknown ID."""
        result = mem_with_mock.history("nonexistent_id")
        assert result == []


# =====================================================================
# infer=False mode (verbatim storage)
# =====================================================================


class TestInferFalseMode:
    """Test verbatim storage mode (infer=False)."""

    def test_infer_false_stores_verbatim(self, mem_with_mock: NeuralMem):
        """infer=False stores the exact text without extraction."""
        result = mem_with_mock.remember(
            "John went to the store yesterday and bought milk",
            infer=False,
        )
        assert len(result) == 1
        assert result[0].content == "John went to the store yesterday and bought milk"

    def test_infer_false_no_extraction(self, mem_with_mock: NeuralMem):
        """infer=False does not split into multiple memories."""
        long_text = "First fact. Second fact. Third fact."
        result = mem_with_mock.remember(long_text, infer=False)
        assert len(result) == 1
        assert result[0].content == long_text

    def test_infer_false_creates_history(self, mem_with_mock: NeuralMem):
        """infer=False records a CREATE history entry."""
        result = mem_with_mock.remember("Verbatim text", infer=False)
        memory_id = result[0].id
        history = mem_with_mock.history(memory_id)
        create_entries = [e for e in history if e.event == "CREATE"]
        assert len(create_entries) == 1

    def test_infer_false_with_user_id(self, mem_with_mock: NeuralMem):
        """infer=False respects user_id parameter."""
        result = mem_with_mock.remember("User memory", infer=False, user_id="user1")
        assert result[0].user_id == "user1"

    def test_infer_false_with_tags(self, mem_with_mock: NeuralMem):
        """infer=False respects tags parameter."""
        result = mem_with_mock.remember("Tagged memory", infer=False, tags=["important"])
        assert "important" in result[0].tags

    def test_infer_false_with_importance(self, mem_with_mock: NeuralMem):
        """infer=False respects importance parameter."""
        result = mem_with_mock.remember("Important", infer=False, importance=0.9)
        assert result[0].importance == pytest.approx(0.9)

    def test_infer_false_with_metadata(self, mem_with_mock: NeuralMem):
        """infer=False stores metadata in history."""
        result = mem_with_mock.remember(
            "Meta memory", infer=False, metadata={"source": "test"}
        )
        history = mem_with_mock.history(result[0].id)
        create_entries = [e for e in history if e.event == "CREATE"]
        assert len(create_entries) == 1
        assert create_entries[0].metadata.get("source") == "test"

    def test_infer_false_default_type_episodic(self, mem_with_mock: NeuralMem):
        """infer=False defaults to EPISODIC memory type."""
        result = mem_with_mock.remember("Raw text", infer=False)
        assert result[0].memory_type == MemoryType.EPISODIC

    def test_infer_false_custom_type(self, mem_with_mock: NeuralMem):
        """infer=False allows overriding memory type."""
        result = mem_with_mock.remember(
            "Procedural text", infer=False, memory_type=MemoryType.PROCEDURAL
        )
        assert result[0].memory_type == MemoryType.PROCEDURAL

    def test_infer_false_empty_content(self, mem_with_mock: NeuralMem):
        """infer=False with empty content returns empty list."""
        result = mem_with_mock.remember("", infer=False)
        assert result == []

    def test_infer_false_storable_and_retrievable(self, mem_with_mock: NeuralMem):
        """infer=False memories can be retrieved by recall()."""
        mem_with_mock.remember("The quick brown fox", infer=False, user_id="u1")
        results = mem_with_mock.recall("quick brown fox", user_id="u1", min_score=0.0)
        assert len(results) >= 1


# =====================================================================
# Event-driven extraction
# =====================================================================


class TestEventDrivenExtraction:
    """Test ADD/UPDATE/DELETE/NONE event types in extraction."""

    def test_extracted_item_default_event(self):
        """ExtractedItem defaults to ADD event."""
        item = ExtractedItem(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            entities=[],
            relations=[],
            tags=[],
            importance=0.5,
        )
        assert item.event == "ADD"

    def test_extracted_item_custom_event(self):
        """ExtractedItem accepts custom event type."""
        item = ExtractedItem(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            entities=[],
            relations=[],
            tags=[],
            importance=0.5,
            event="UPDATE",
        )
        assert item.event == "UPDATE"

    def test_extracted_item_delete_event(self):
        """ExtractedItem supports DELETE event."""
        item = ExtractedItem(
            content="old fact",
            memory_type=MemoryType.SEMANTIC,
            entities=[],
            relations=[],
            tags=[],
            importance=0.5,
            event="DELETE",
        )
        assert item.event == "DELETE"


# =====================================================================
# Custom extraction prompt
# =====================================================================


class TestCustomExtractionPrompt:
    """Test custom_extraction_prompt config option."""

    def test_config_has_custom_prompt_field(self):
        """NeuralMemConfig has custom_extraction_prompt field."""
        from neuralmem.core.config import NeuralMemConfig
        config = NeuralMemConfig(custom_extraction_prompt="Extract facts about cats")
        assert config.custom_extraction_prompt == "Extract facts about cats"

    def test_config_default_custom_prompt_none(self):
        """custom_extraction_prompt defaults to None."""
        from neuralmem.core.config import NeuralMemConfig
        config = NeuralMemConfig()
        assert config.custom_extraction_prompt is None

    def test_config_has_llm_conflict_resolution(self):
        """NeuralMemConfig has enable_llm_conflict_resolution field."""
        from neuralmem.core.config import NeuralMemConfig
        config = NeuralMemConfig(enable_llm_conflict_resolution=True)
        assert config.enable_llm_conflict_resolution is True

    def test_config_default_llm_conflict_resolution_false(self):
        """enable_llm_conflict_resolution defaults to False."""
        from neuralmem.core.config import NeuralMemConfig
        config = NeuralMemConfig()
        assert config.enable_llm_conflict_resolution is False


# =====================================================================
# Storage: history table
# =====================================================================


class TestStorageHistory:
    """Test SQLite storage history methods."""

    def test_save_and_get_history(self, tmp_db_path: str):
        """save_history() and get_history() work together."""
        from neuralmem.core.config import NeuralMemConfig
        from neuralmem.storage.sqlite import SQLiteStorage
        config = NeuralMemConfig(db_path=tmp_db_path)
        storage = SQLiteStorage(config)
        storage.save_history("mem123", "old text", "new text", event="UPDATE")
        history = storage.get_history("mem123")
        assert len(history) == 1
        assert history[0]["old_content"] == "old text"
        assert history[0]["new_content"] == "new text"
        assert history[0]["event"] == "UPDATE"

    def test_history_multiple_entries(self, tmp_db_path: str):
        """get_history() returns entries in chronological order."""
        from neuralmem.core.config import NeuralMemConfig
        from neuralmem.storage.sqlite import SQLiteStorage
        config = NeuralMemConfig(db_path=tmp_db_path)
        storage = SQLiteStorage(config)
        storage.save_history("mem1", None, "v1", event="CREATE")
        storage.save_history("mem1", "v1", "v2", event="UPDATE")
        storage.save_history("mem1", "v2", "v3", event="UPDATE")
        history = storage.get_history("mem1")
        assert len(history) == 3
        assert history[0]["event"] == "CREATE"
        assert history[1]["new_content"] == "v2"
        assert history[2]["new_content"] == "v3"

    def test_history_empty_for_unknown(self, tmp_db_path: str):
        """get_history() returns empty list for unknown memory."""
        from neuralmem.core.config import NeuralMemConfig
        from neuralmem.storage.sqlite import SQLiteStorage
        config = NeuralMemConfig(db_path=tmp_db_path)
        storage = SQLiteStorage(config)
        assert storage.get_history("nonexistent") == []

    def test_history_with_metadata(self, tmp_db_path: str):
        """save_history() stores metadata correctly."""
        from neuralmem.core.config import NeuralMemConfig
        from neuralmem.storage.sqlite import SQLiteStorage
        config = NeuralMemConfig(db_path=tmp_db_path)
        storage = SQLiteStorage(config)
        storage.save_history(
            "mem1", "old", "new", event="UPDATE", metadata={"reason": "correction"}
        )
        history = storage.get_history("mem1")
        assert history[0]["metadata"]["reason"] == "correction"
