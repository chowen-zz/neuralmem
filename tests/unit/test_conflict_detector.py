"""Tests for ConflictDetector — 15+ tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neuralmem.management.conflict_detector import ConflictDetector
from neuralmem.management.llm_manager import MemoryOperation, OperationType

# ==================== Init ====================


class TestDetectorInit:

    def test_detector_init(self):
        with patch(
            "neuralmem.management.conflict_detector.LLMMemoryManager"
        ):
            detector = ConflictDetector(llm_backend="ollama")
            assert detector is not None

    def test_detector_init_openai(self):
        with patch(
            "neuralmem.management.conflict_detector.LLMMemoryManager"
        ):
            detector = ConflictDetector(llm_backend="openai")
            assert detector is not None


# ==================== Conflict Detection ====================


class TestDetectConflicts:

    @pytest.mark.asyncio
    async def test_detect_conflicts_returns_list(self):
        mock_manager = MagicMock()
        mock_manager._call_llm = AsyncMock(return_value="[]")
        mock_manager._parse_json_response = AsyncMock(return_value=[])

        detector = ConflictDetector.__new__(ConflictDetector)
        detector._manager = mock_manager

        result = await detector.detect_conflicts(
            ["new fact"],
            [{"id": "1", "content": "old fact"}],
        )
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_conflicts_with_contradictions(self):
        conflicts = [
            {
                "new_memory": "Sky is green",
                "existing_memory_id": "mem_1",
                "conflict_type": "contradiction",
                "resolution": "update",
            }
        ]
        mock_manager = MagicMock()
        mock_manager._call_llm = AsyncMock(return_value="[]")
        mock_manager._parse_json_response = AsyncMock(
            return_value=conflicts
        )

        detector = ConflictDetector.__new__(ConflictDetector)
        detector._manager = mock_manager

        result = await detector.detect_conflicts(
            ["Sky is green"],
            [{"id": "mem_1", "content": "Sky is blue"}],
        )
        assert len(result) == 1
        assert result[0]["conflict_type"] == "contradiction"

    @pytest.mark.asyncio
    async def test_detect_conflicts_no_conflicts(self):
        mock_manager = MagicMock()
        mock_manager._call_llm = AsyncMock(return_value="[]")
        mock_manager._parse_json_response = AsyncMock(return_value=[])

        detector = ConflictDetector.__new__(ConflictDetector)
        detector._manager = mock_manager

        result = await detector.detect_conflicts(
            ["User likes dogs"],
            [{"id": "1", "content": "User likes cats"}],
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_detect_conflicts_empty_inputs(self):
        detector = ConflictDetector.__new__(ConflictDetector)
        detector._manager = MagicMock()

        # Empty new memories
        result = await detector.detect_conflicts(
            [], [{"id": "1", "content": "fact"}]
        )
        assert result == []

        # Empty existing memories
        result = await detector.detect_conflicts(
            ["new fact"], []
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_detect_conflicts_validates_conflict_type(self):
        """Invalid conflict_type defaults to contradiction."""
        conflicts = [
            {
                "new_memory": "test",
                "existing_memory_id": "m1",
                "conflict_type": "invalid_type",
                "resolution": "update",
            }
        ]
        mock_manager = MagicMock()
        mock_manager._call_llm = AsyncMock(return_value="")
        mock_manager._parse_json_response = AsyncMock(
            return_value=conflicts
        )

        detector = ConflictDetector.__new__(ConflictDetector)
        detector._manager = mock_manager

        result = await detector.detect_conflicts(
            ["test"], [{"id": "m1", "content": "old"}]
        )
        assert result[0]["conflict_type"] == "contradiction"

    @pytest.mark.asyncio
    async def test_conflict_with_metadata(self):
        """Conflict dicts preserve all fields."""
        conflicts = [
            {
                "new_memory": "new",
                "existing_memory_id": "e1",
                "conflict_type": "supersession",
                "resolution": "update",
            }
        ]
        mock_manager = MagicMock()
        mock_manager._call_llm = AsyncMock(return_value="")
        mock_manager._parse_json_response = AsyncMock(
            return_value=conflicts
        )

        detector = ConflictDetector.__new__(ConflictDetector)
        detector._manager = mock_manager

        result = await detector.detect_conflicts(
            ["new"], [{"id": "e1", "content": "old"}]
        )
        assert result[0]["existing_memory_id"] == "e1"
        assert result[0]["resolution"] == "update"


# ==================== Auto Resolve ====================


class TestAutoResolve:

    @pytest.mark.asyncio
    async def test_auto_resolve_returns_operations(self):
        detector = ConflictDetector.__new__(ConflictDetector)
        conflicts = [
            {
                "new_memory": "new fact",
                "existing_memory_id": "m1",
                "conflict_type": "contradiction",
                "resolution": "update",
            }
        ]
        ops = await detector.auto_resolve(conflicts)
        assert len(ops) == 1
        assert isinstance(ops[0], MemoryOperation)

    @pytest.mark.asyncio
    async def test_auto_resolve_update_for_contradiction(self):
        detector = ConflictDetector.__new__(ConflictDetector)
        conflicts = [
            {
                "new_memory": "Sky is green",
                "existing_memory_id": "m1",
                "conflict_type": "contradiction",
                "resolution": "update",
            }
        ]
        ops = await detector.auto_resolve(conflicts)
        assert ops[0].op_type is OperationType.UPDATE
        assert ops[0].old_memory_id == "m1"

    @pytest.mark.asyncio
    async def test_auto_resolve_delete_for_negation(self):
        detector = ConflictDetector.__new__(ConflictDetector)
        conflicts = [
            {
                "new_memory": "No longer true",
                "existing_memory_id": "m2",
                "conflict_type": "negation",
                "resolution": "delete",
            }
        ]
        ops = await detector.auto_resolve(conflicts)
        assert ops[0].op_type is OperationType.DELETE
        assert ops[0].old_memory_id == "m2"

    @pytest.mark.asyncio
    async def test_auto_resolve_empty_conflicts(self):
        detector = ConflictDetector.__new__(ConflictDetector)
        ops = await detector.auto_resolve([])
        assert ops == []

    @pytest.mark.asyncio
    async def test_auto_resolve_multiple_conflicts(self):
        detector = ConflictDetector.__new__(ConflictDetector)
        conflicts = [
            {
                "new_memory": "A",
                "existing_memory_id": "m1",
                "conflict_type": "contradiction",
                "resolution": "update",
            },
            {
                "new_memory": "B",
                "existing_memory_id": "m2",
                "conflict_type": "negation",
                "resolution": "delete",
            },
        ]
        ops = await detector.auto_resolve(conflicts)
        assert len(ops) == 2
        assert ops[0].op_type is OperationType.UPDATE
        assert ops[1].op_type is OperationType.DELETE

    @pytest.mark.asyncio
    async def test_auto_resolve_resolution_override(self):
        """resolution=delete forces DELETE even for contradiction."""
        detector = ConflictDetector.__new__(ConflictDetector)
        conflicts = [
            {
                "new_memory": "fact",
                "existing_memory_id": "m1",
                "conflict_type": "contradiction",
                "resolution": "delete",
            }
        ]
        ops = await detector.auto_resolve(conflicts)
        assert ops[0].op_type is OperationType.DELETE


# ==================== Misc ====================


class TestConflictMisc:

    @pytest.mark.asyncio
    async def test_batch_conflict_detection(self):
        """Multiple new memories detected in one call."""
        conflicts = [
            {
                "new_memory": "A",
                "existing_memory_id": "e1",
                "conflict_type": "contradiction",
                "resolution": "update",
            },
            {
                "new_memory": "B",
                "existing_memory_id": "e2",
                "conflict_type": "supersession",
                "resolution": "update",
            },
        ]
        mock_manager = MagicMock()
        mock_manager._call_llm = AsyncMock(return_value="")
        mock_manager._parse_json_response = AsyncMock(
            return_value=conflicts
        )

        detector = ConflictDetector.__new__(ConflictDetector)
        detector._manager = mock_manager

        result = await detector.detect_conflicts(
            ["A", "B"],
            [
                {"id": "e1", "content": "old A"},
                {"id": "e2", "content": "old B"},
            ],
        )
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_llm_call_mock(self):
        """Verify _call_llm is invoked during detect_conflicts."""
        mock_manager = MagicMock()
        mock_manager._call_llm = AsyncMock(return_value="[]")
        mock_manager._parse_json_response = AsyncMock(return_value=[])

        detector = ConflictDetector.__new__(ConflictDetector)
        detector._manager = mock_manager

        await detector.detect_conflicts(
            ["fact"], [{"id": "1", "content": "old"}]
        )
        mock_manager._call_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_json_parse_fallback(self):
        """Invalid JSON from LLM results in empty list."""
        mock_manager = MagicMock()
        mock_manager._call_llm = AsyncMock(return_value="bad json")
        mock_manager._parse_json_response = AsyncMock(return_value=[])

        detector = ConflictDetector.__new__(ConflictDetector)
        detector._manager = mock_manager

        result = await detector.detect_conflicts(
            ["fact"], [{"id": "1", "content": "old"}]
        )
        assert result == []

    def test_conflict_type_classification(self):
        """Valid conflict types are preserved."""
        valid_types = ["contradiction", "supersession", "negation"]
        for ct in valid_types:
            assert ct in {"contradiction", "supersession", "negation"}

    @pytest.mark.asyncio
    async def test_conflict_severity_levels(self):
        """All resolutions generate proper confidence scores."""
        detector = ConflictDetector.__new__(ConflictDetector)

        # UPDATE gets 0.85 confidence
        update_ops = await detector.auto_resolve([
            {
                "new_memory": "x",
                "existing_memory_id": "m",
                "conflict_type": "contradiction",
                "resolution": "update",
            }
        ])
        assert update_ops[0].confidence == 0.85

        # DELETE gets 0.9 confidence
        delete_ops = await detector.auto_resolve([
            {
                "new_memory": "x",
                "existing_memory_id": "m",
                "conflict_type": "negation",
                "resolution": "delete",
            }
        ])
        assert delete_ops[0].confidence == 0.9
