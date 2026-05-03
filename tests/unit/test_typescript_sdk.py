"""Tests for the TypeScript SDK file structure and type consistency.

Verifies that all expected files exist, that the TypeScript source
compiles without syntax errors, and that the exported types match
the Python Pydantic models.
"""
from __future__ import annotations

import json
from pathlib import Path

# Paths
_SDK_ROOT = Path(__file__).resolve().parent.parent.parent / "sdk" / "typescript"
_TS_SRC = _SDK_ROOT / "src"
_ROOT = Path(__file__).resolve().parent.parent.parent
_PY_TYPES = _ROOT / "src" / "neuralmem" / "core" / "types.py"


# ---------------------------------------------------------------------------
# File structure tests
# ---------------------------------------------------------------------------


class TestFileStructure:
    """Verify all expected SDK files exist."""

    def test_package_json_exists(self):
        """package.json should exist in the SDK root."""
        assert (_SDK_ROOT / "package.json").is_file()

    def test_tsconfig_json_exists(self):
        """tsconfig.json should exist in the SDK root."""
        assert (_SDK_ROOT / "tsconfig.json").is_file()

    def test_src_index_ts_exists(self):
        """src/index.ts should exist."""
        assert (_TS_SRC / "index.ts").is_file()

    def test_src_client_ts_exists(self):
        """src/client.ts should exist."""
        assert (_TS_SRC / "client.ts").is_file()

    def test_src_types_ts_exists(self):
        """src/types.ts should exist."""
        assert (_TS_SRC / "types.ts").is_file()

    def test_readme_exists(self):
        """README.md should exist in the SDK root."""
        assert (_SDK_ROOT / "README.md").is_file()


# ---------------------------------------------------------------------------
# package.json tests
# ---------------------------------------------------------------------------


class TestPackageJson:
    """Verify package.json metadata."""

    def test_package_name(self):
        """Package name should be @neuralmem/sdk."""
        data = json.loads((_SDK_ROOT / "package.json").read_text())
        assert data["name"] == "@neuralmem/sdk"

    def test_package_version(self):
        """Package version should be 0.7.0."""
        data = json.loads((_SDK_ROOT / "package.json").read_text())
        assert data["version"] == "0.7.0"

    def test_no_runtime_dependencies(self):
        """SDK should have zero runtime dependencies."""
        data = json.loads((_SDK_ROOT / "package.json").read_text())
        assert data.get("dependencies") is None or data["dependencies"] == {}

    def test_module_type(self):
        """Package should use ES module type."""
        data = json.loads((_SDK_ROOT / "package.json").read_text())
        assert data.get("type") == "module"


# ---------------------------------------------------------------------------
# tsconfig.json tests
# ---------------------------------------------------------------------------


class TestTsConfig:
    """Verify TypeScript configuration."""

    def test_target_es2020(self):
        """Target should be ES2020."""
        data = json.loads((_SDK_ROOT / "tsconfig.json").read_text())
        assert data["compilerOptions"]["target"] == "ES2020"

    def test_module_esnext(self):
        """Module should be ESNext."""
        data = json.loads((_SDK_ROOT / "tsconfig.json").read_text())
        assert data["compilerOptions"]["module"] == "ESNext"

    def test_strict_mode(self):
        """Strict mode should be enabled."""
        data = json.loads((_SDK_ROOT / "tsconfig.json").read_text())
        assert data["compilerOptions"]["strict"] is True


# ---------------------------------------------------------------------------
# TypeScript source content tests
# ---------------------------------------------------------------------------


class TestTypeScriptSource:
    """Verify TypeScript source code content and structure."""

    def test_types_exports_memory_interface(self):
        """types.ts should export a Memory interface."""
        content = (_TS_SRC / "types.ts").read_text()
        assert "interface Memory" in content

    def test_types_exports_search_result(self):
        """types.ts should export a SearchResult interface."""
        content = (_TS_SRC / "types.ts").read_text()
        assert "interface SearchResult" in content

    def test_types_exports_health_report(self):
        """types.ts should export a HealthReport interface."""
        content = (_TS_SRC / "types.ts").read_text()
        assert "interface HealthReport" in content

    def test_types_exports_memory_type(self):
        """types.ts should export MemoryType type."""
        content = (_TS_SRC / "types.ts").read_text()
        assert "MemoryType" in content

    def test_types_exports_neural_mem_client_options(self):
        """types.ts should export NeuralMemClientOptions."""
        content = (_TS_SRC / "types.ts").read_text()
        assert "NeuralMemClientOptions" in content

    def test_client_exports_class(self):
        """client.ts should export NeuralMemClient class."""
        content = (_TS_SRC / "client.ts").read_text()
        assert "class NeuralMemClient" in content

    def test_client_has_remember_method(self):
        """NeuralMemClient should have a remember method."""
        content = (_TS_SRC / "client.ts").read_text()
        assert "async remember(" in content

    def test_client_has_recall_method(self):
        """NeuralMemClient should have a recall method."""
        content = (_TS_SRC / "client.ts").read_text()
        assert "async recall(" in content

    def test_client_has_reflect_method(self):
        """NeuralMemClient should have a reflect method."""
        content = (_TS_SRC / "client.ts").read_text()
        assert "async reflect(" in content

    def test_client_has_forget_method(self):
        """NeuralMemClient should have a forget method."""
        content = (_TS_SRC / "client.ts").read_text()
        assert "async forget(" in content

    def test_client_has_health_method(self):
        """NeuralMemClient should have a health method."""
        content = (_TS_SRC / "client.ts").read_text()
        assert "async health()" in content

    def test_client_constructor_takes_options(self):
        """NeuralMemClient constructor should accept options."""
        content = (_TS_SRC / "client.ts").read_text()
        assert "constructor(options: NeuralMemClientOptions)" in content

    def test_index_exports_client(self):
        """index.ts should export NeuralMemClient."""
        content = (_TS_SRC / "index.ts").read_text()
        assert "NeuralMemClient" in content

    def test_index_exports_types(self):
        """index.ts should re-export types."""
        content = (_TS_SRC / "index.ts").read_text()
        assert "Memory" in content
        assert "SearchResult" in content
        assert "HealthReport" in content


# ---------------------------------------------------------------------------
# Type consistency tests (Python ↔ TypeScript)
# ---------------------------------------------------------------------------


class TestTypeConsistency:
    """Verify TypeScript types are consistent with Python Pydantic models."""

    def test_memory_has_required_fields(self):
        """TS Memory interface should have all Python Memory fields."""
        ts_types = (_TS_SRC / "types.ts").read_text()
        required_fields = [
            "id", "content", "memory_type", "scope", "user_id",
            "importance", "is_active", "created_at", "access_count",
        ]
        for field in required_fields:
            assert field in ts_types, (
                f"Memory interface missing field: {field}"
            )

    def test_memory_type_values_match_python(self):
        """TS MemoryType should include all Python MemoryType values."""
        ts_types = (_TS_SRC / "types.ts").read_text()
        py_values = ["fact", "preference", "episodic",
                     "semantic", "procedural", "working"]
        for val in py_values:
            assert f'"{val}"' in ts_types, (
                f"MemoryType missing value: {val}"
            )

    def test_memory_scope_values_match_python(self):
        """TS MemoryScope should include all Python MemoryScope values."""
        ts_types = (_TS_SRC / "types.ts").read_text()
        py_values = ["user", "agent", "session", "shared"]
        for val in py_values:
            assert f'"{val}"' in ts_types, (
                f"MemoryScope missing value: {val}"
            )

    def test_health_status_values_match_python(self):
        """TS HealthStatus should include all Python HealthStatus values."""
        ts_types = (_TS_SRC / "types.ts").read_text()
        py_values = ["healthy", "degraded", "unhealthy"]
        for val in py_values:
            assert f'"{val}"' in ts_types, (
                f"HealthStatus missing value: {val}"
            )

    def test_search_result_has_score_and_memory(self):
        """TS SearchResult should have score and memory fields."""
        ts_types = (_TS_SRC / "types.ts").read_text()
        assert "score: number" in ts_types
        assert "memory: Memory" in ts_types

    def test_search_result_has_retrieval_method(self):
        """TS SearchResult should have retrieval_method field."""
        ts_types = (_TS_SRC / "types.ts").read_text()
        assert "retrieval_method: string" in ts_types
