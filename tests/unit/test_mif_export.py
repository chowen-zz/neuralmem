"""Tests for the MIF (Memory Interchange Format) export module."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from neuralmem.core.types import Memory, MemoryType
from neuralmem.export.mif import MIFExporter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def exporter() -> MIFExporter:
    return MIFExporter(version="0.2")


@pytest.fixture
def sample_memory() -> Memory:
    return Memory(
        id="TESTID001",
        content="The user prefers dark mode in their IDE.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.8,
        user_id="user-42",
        tags=("preference", "ui"),
        source="conversation",
        created_at=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2025, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_memories() -> list[Memory]:
    return [
        Memory(
            id="MEM001",
            content="Fact A: the sky is blue.",
            memory_type=MemoryType.SEMANTIC,
            importance=0.9,
            tags=("science",),
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ),
        Memory(
            id="MEM002",
            content="Event B: the user attended a meeting.",
            memory_type=MemoryType.EPISODIC,
            importance=0.6,
            tags=("meeting",),
            created_at=datetime(2025, 2, 1, tzinfo=timezone.utc),
            updated_at=datetime(2025, 2, 1, tzinfo=timezone.utc),
        ),
        Memory(
            id="MEM003",
            content="Procedure C: deploy steps.",
            memory_type=MemoryType.PROCEDURAL,
            importance=0.7,
            created_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
            updated_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExportMemory:
    """Tests for export_memory (single memory -> MIF dict)."""

    def test_export_memory_fields(self, exporter: MIFExporter, sample_memory: Memory):
        """Single memory export has all required MIF fields."""
        result = exporter.export_memory(sample_memory)

        required_fields = [
            "id", "content", "memory_type", "importance", "confidence",
            "source_refs", "provenance", "tags", "user_id",
            "supersedes", "contradicts", "validity", "metadata",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_export_preserves_importance(self, exporter: MIFExporter, sample_memory: Memory):
        """Importance score round-trips correctly."""
        result = exporter.export_memory(sample_memory)
        assert result["importance"] == 0.8

    def test_export_preserves_tags(self, exporter: MIFExporter, sample_memory: Memory):
        """Tags round-trip correctly."""
        result = exporter.export_memory(sample_memory)
        assert result["tags"] == ["preference", "ui"]

    def test_export_preserves_provenance(self, exporter: MIFExporter, sample_memory: Memory):
        """Provenance dict is complete with expected keys."""
        result = exporter.export_memory(sample_memory)
        prov = result["provenance"]
        assert "created_at" in prov
        assert "updated_at" in prov
        assert "extractor" in prov
        assert "embedder" in prov
        assert prov["created_at"] == "2025-01-01T00:00:00+00:00"
        assert prov["updated_at"] == "2025-01-02T00:00:00+00:00"

    def test_export_memory_type_string(self, exporter: MIFExporter, sample_memory: Memory):
        """memory_type is exported as a plain string, not enum."""
        result = exporter.export_memory(sample_memory)
        assert result["memory_type"] == "semantic"
        assert isinstance(result["memory_type"], str)


class TestExportJson:
    """Tests for export_json (list -> MIF JSON)."""

    def test_export_json_format(self, exporter: MIFExporter, sample_memories: list[Memory]):
        """JSON output is valid and has version/count/memories."""
        json_str = exporter.export_json(sample_memories)
        data = json.loads(json_str)

        assert data["version"] == "0.2"
        assert data["count"] == 3
        assert isinstance(data["memories"], list)
        assert len(data["memories"]) == 3

    def test_export_json_to_file(
        self, exporter: MIFExporter, sample_memories: list[Memory], tmp_path: Path
    ):
        """Writes to file correctly."""
        out = tmp_path / "export.json"
        json_str = exporter.export_json(sample_memories, output_path=str(out))

        assert out.exists()
        file_content = out.read_text(encoding="utf-8")
        assert file_content == json_str

    def test_export_multiple_memories(self, exporter: MIFExporter, sample_memories: list[Memory]):
        """Count matches actual number of memories."""
        json_str = exporter.export_json(sample_memories)
        data = json.loads(json_str)
        assert data["count"] == len(data["memories"])

    def test_export_empty_list(self, exporter: MIFExporter):
        """Empty memories list works."""
        json_str = exporter.export_json([])
        data = json.loads(json_str)
        assert data["count"] == 0
        assert data["memories"] == []


class TestVersionField:
    """Tests for the version field."""

    def test_version_field(self):
        """Exported version matches constructor param."""
        for v in ("0.1", "0.2", "1.0"):
            exp = MIFExporter(version=v)
            json_str = exp.export_json([])
            data = json.loads(json_str)
            assert data["version"] == v


class TestExportMarkdown:
    """Tests for export_markdown."""

    def test_export_markdown_format(self, exporter: MIFExporter, sample_memories: list[Memory]):
        """Markdown has headers and memory content."""
        md = exporter.export_markdown(sample_memories)

        assert "# NeuralMem Memory Export" in md
        assert "## Memory 1" in md
        assert "## Memory 2" in md
        assert "## Memory 3" in md
        assert "Fact A: the sky is blue." in md
        assert "**Version:** 0.2" in md
        assert "**Count:** 3" in md

    def test_export_markdown_to_file(
        self, exporter: MIFExporter, sample_memories: list[Memory], tmp_path: Path
    ):
        """Writes markdown to file correctly."""
        out = tmp_path / "export.md"
        md = exporter.export_markdown(sample_memories, output_path=str(out))

        assert out.exists()
        file_content = out.read_text(encoding="utf-8")
        assert file_content == md


class TestImportJson:
    """Tests for import_json (round-trip, error handling)."""

    def test_import_json_roundtrip(self, exporter: MIFExporter, sample_memories: list[Memory]):
        """Export then import returns same data."""
        json_str = exporter.export_json(sample_memories)
        imported = exporter.import_json(json_str)

        assert len(imported) == len(sample_memories)
        for orig, imp in zip(sample_memories, imported):
            assert imp["id"] == orig.id
            assert imp["content"] == orig.content

    def test_import_invalid_json(self, exporter: MIFExporter):
        """Invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            exporter.import_json("{bad json!!!")


class TestValidate:
    """Tests for validate."""

    def test_validate_valid(self, exporter: MIFExporter):
        """Valid entry passes validation."""
        entry = {
            "id": "X1",
            "content": "hello",
            "importance": 0.5,
            "confidence": 0.9,
            "source_refs": [],
            "provenance": {},
            "tags": [],
            "supersedes": [],
            "contradicts": [],
            "validity": {},
            "metadata": {},
        }
        is_valid, errors = exporter.validate(entry)
        assert is_valid is True
        assert errors == []

    def test_validate_missing_id(self, exporter: MIFExporter):
        """Missing id fails validation."""
        entry = {"content": "hello"}
        is_valid, errors = exporter.validate(entry)
        assert is_valid is False
        assert any("id" in e for e in errors)

    def test_validate_missing_content(self, exporter: MIFExporter):
        """Missing content fails validation."""
        entry = {"id": "X1"}
        is_valid, errors = exporter.validate(entry)
        assert is_valid is False
        assert any("content" in e for e in errors)
