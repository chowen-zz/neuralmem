"""Unit tests for NeuralMem V1.7 ExampleProjectGenerator — all mock-based.

Covers:
  • ProjectTemplate / FileNode dataclasses
  • generate() for all supported languages
  • generate_from_memory()
  • scaffold_to_disk()
  • build_file_tree()
  • export_to_json()
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neuralmem.community.example_generator import (
    ExampleProjectGenerator,
    FileNode,
    ProjectTemplate,
)


# =============================================================================
# ProjectTemplate / FileNode
# =============================================================================

class TestProjectTemplate:
    def test_to_dict_serializes(self):
        tmpl = ProjectTemplate(
            name="demo",
            description="A demo project",
            language="python",
            files={"main.py": "print(1)"},
            dependencies=["neuralmem"],
        )
        d = tmpl.to_dict()
        assert d["name"] == "demo"
        assert d["language"] == "python"
        assert "generated_at" in d


class TestFileNode:
    def test_file_node_tree(self):
        root = FileNode(path="project", is_directory=True)
        root.children.append(FileNode(path="main.py", content="print(1)"))
        assert len(root.children) == 1
        assert root.children[0].path == "main.py"


# =============================================================================
# ExampleProjectGenerator — generate()
# =============================================================================

class TestExampleProjectGeneratorGenerate:
    def test_generate_python(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("Build an agent memory system", language="python")
        assert tmpl.language == "python"
        assert "main.py" in tmpl.files
        assert "pyproject.toml" in tmpl.files
        assert "README.md" in tmpl.files
        assert "test_main.py" in tmpl.files
        assert ".gitignore" in tmpl.files
        assert "neuralmem" in tmpl.dependencies[0]
        assert tmpl.scripts["start"] == "python main.py"
        assert tmpl.scripts["test"] == "pytest"

    def test_generate_javascript(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("JS agent memory", language="javascript")
        assert tmpl.language == "javascript"
        assert "main.js" in tmpl.files
        assert "package.json" in tmpl.files
        assert tmpl.scripts["start"] == "node main.js"

    def test_generate_typescript(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("TS agent memory", language="typescript")
        assert tmpl.language == "typescript"
        assert "main.ts" in tmpl.files
        assert "package.json" in tmpl.files
        assert "@types/node" in tmpl.dev_dependencies

    def test_generate_go(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("Go agent memory", language="go")
        assert tmpl.language == "go"
        assert "main.go" in tmpl.files
        assert "go.mod" in tmpl.files
        assert tmpl.scripts["start"] == "go run main.go"

    def test_generate_rust(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("Rust agent memory", language="rust")
        assert tmpl.language == "rust"
        assert "main.rs" in tmpl.files
        assert "Cargo.toml" in tmpl.files
        assert tmpl.scripts["start"] == "cargo run"

    def test_generate_unsupported_language_raises(self):
        gen = ExampleProjectGenerator()
        with pytest.raises(ValueError, match="Unsupported language"):
            gen.generate("X", language="java")

    def test_generate_with_custom_name(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("Need", language="python", project_name="my-project")
        assert tmpl.name == "my-project"

    def test_generate_with_extra_deps(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("Need", language="python", extra_deps=["requests", "httpx"])
        assert "requests" in tmpl.dependencies
        assert "httpx" in tmpl.dependencies

    def test_generate_infers_tags(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("Build an agent with vector search API", language="python")
        tags = [t.lower() for t in tmpl.tags]
        assert "python" in tags
        assert "agent" in tags
        assert "vector-search" in tags
        assert "api" in tags

    def test_generate_infers_features(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("Agent with search and cache API", language="python")
        features = tmpl.readme
        assert "Semantic search" in features or "search" in features.lower()


# =============================================================================
# generate_from_memory()
# =============================================================================

class TestExampleProjectGeneratorFromMemory:
    def test_generate_from_memory_extracts_need(self):
        gen = ExampleProjectGenerator()
        memory = "The user wants to build a chatbot with persistent memory using NeuralMem."
        tmpl = gen.generate_from_memory(memory, language="python")
        assert tmpl.description == memory[:200]
        assert "main.py" in tmpl.files

    def test_generate_from_memory_falls_back_to_truncated(self):
        gen = ExampleProjectGenerator()
        memory = "Short."
        tmpl = gen.generate_from_memory(memory, language="python")
        assert tmpl.description == "Short."


# =============================================================================
# scaffold_to_disk()
# =============================================================================

class TestExampleProjectGeneratorScaffold:
    def test_scaffold_to_disk_creates_files(self, tmp_path: Path):
        gen = ExampleProjectGenerator(output_dir=tmp_path)
        tmpl = gen.generate("Agent memory", language="python")
        root = gen.scaffold_to_disk(tmpl)
        assert root.exists()
        assert (root / "main.py").exists()
        assert (root / "pyproject.toml").exists()
        assert (root / "README.md").exists()
        assert (root / "test_main.py").exists()
        assert (root / ".gitignore").exists()

    def test_scaffold_uses_custom_base_dir(self, tmp_path: Path):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("Agent memory", language="python")
        root = gen.scaffold_to_disk(tmpl, base_dir=tmp_path / "custom")
        assert root.name == "custom"
        assert (root / "main.py").exists()


# =============================================================================
# build_file_tree()
# =============================================================================

class TestExampleProjectGeneratorFileTree:
    def test_build_file_tree_flat(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("X", language="python")
        tree = gen.build_file_tree(tmpl)
        assert tree.path == tmpl.name
        assert tree.is_directory is True
        names = {c.path for c in tree.children}
        assert "main.py" in names
        assert "pyproject.toml" in names

    def test_build_file_tree_nested(self):
        gen = ExampleProjectGenerator()
        tmpl = ProjectTemplate(
            name="nested",
            description="D",
            language="python",
            files={"src/main.py": "x", "src/utils.py": "y", "README.md": "z"},
        )
        tree = gen.build_file_tree(tmpl)
        assert tree.is_directory
        src = next(c for c in tree.children if c.path == "src")
        assert src.is_directory
        assert {c.path for c in src.children} == {"main.py", "utils.py"}


# =============================================================================
# export_to_json()
# =============================================================================

class TestExampleProjectGeneratorExport:
    def test_export_to_json(self, tmp_path: Path):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("X", language="python")
        path = gen.export_to_json(tmpl, tmp_path / "proj.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == tmpl.name
        assert data["language"] == "python"


# =============================================================================
# get_history()
# =============================================================================

class TestExampleProjectGeneratorHistory:
    def test_get_history_returns_copy(self):
        gen = ExampleProjectGenerator()
        gen.generate("A", language="python")
        history = gen.get_history()
        assert len(history) == 1
        history.clear()
        assert len(gen.get_history()) == 1


# =============================================================================
# Language-specific content checks
# =============================================================================

class TestLanguageSpecificContent:
    def test_python_main_imports_neuralmem(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("X", language="python")
        assert "from neuralmem import NeuralMem" in tmpl.files["main.py"]

    def test_javascript_main_requires_sdk(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("X", language="javascript")
        assert "@neuralmem/sdk" in tmpl.files["main.js"]

    def test_typescript_main_has_types(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("X", language="typescript")
        assert "Promise<void>" in tmpl.files["main.ts"]

    def test_go_main_imports_sdk(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("X", language="go")
        assert "github.com/neuralmem/go-sdk" in tmpl.files["main.go"]

    def test_rust_main_uses_neuralmem(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("X", language="rust")
        assert "neuralmem::NeuralMem" in tmpl.files["main.rs"]

    def test_python_package_file(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("X", language="python")
        assert "[project]" in tmpl.files["pyproject.toml"]

    def test_js_package_file(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("X", language="javascript")
        pkg = json.loads(tmpl.files["package.json"])
        assert pkg["name"] == tmpl.name

    def test_go_mod_file(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("X", language="go")
        assert f"module {tmpl.name}" in tmpl.files["go.mod"]

    def test_rust_cargo_file(self):
        gen = ExampleProjectGenerator()
        tmpl = gen.generate("X", language="rust")
        assert "[package]" in tmpl.files["Cargo.toml"]

    def test_gitignore_per_language(self):
        gen = ExampleProjectGenerator()
        py_tmpl = gen.generate("X", language="python")
        assert "venv/" in py_tmpl.files[".gitignore"]
        js_tmpl = gen.generate("Y", language="javascript")
        assert "node_modules/" in js_tmpl.files[".gitignore"]

    def test_test_files_present(self):
        gen = ExampleProjectGenerator()
        for lang in gen.SUPPORTED_LANGUAGES:
            tmpl = gen.generate("X", language=lang)
            test_file = f"test_main{gen._language_packs[lang]['ext']}"
            assert test_file in tmpl.files
