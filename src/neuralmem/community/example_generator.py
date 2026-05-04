"""NeuralMem V1.7 community growth engine — example project generator.

ExampleProjectGenerator: generate complete example projects from user needs
with project scaffolding, dependency management, and README generation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ProjectTemplate:
    """Represents a generated project template."""

    name: str
    description: str
    language: str
    files: dict[str, str] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    dev_dependencies: list[str] = field(default_factory=list)
    scripts: dict[str, str] = field(default_factory=dict)
    readme: str = ""
    tags: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "language": self.language,
            "files": self.files,
            "dependencies": self.dependencies,
            "dev_dependencies": self.dev_dependencies,
            "scripts": self.scripts,
            "readme": self.readme,
            "tags": self.tags,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class FileNode:
    """A file or directory in the project tree."""

    path: str
    content: str = ""
    is_directory: bool = False
    children: list[FileNode] = field(default_factory=list)


class ExampleProjectGenerator:
    """Generate complete example projects from user needs."""

    SUPPORTED_LANGUAGES: set[str] = {"python", "javascript", "typescript", "go", "rust"}

    TEMPLATE_README = """# {name}

{description}

## Features

{features}

## Installation

```bash
{install_cmd}
```

## Usage

```bash
{usage_cmd}
```

## Project Structure

```
{structure}
```

## License

MIT
"""

    DEFAULT_PY_DEPS: list[str] = ["neuralmem>=0.9", "pydantic>=2.0"]
    DEFAULT_JS_DEPS: list[str] = ["@neuralmem/sdk"]
    DEFAULT_TS_DEPS: list[str] = ["@neuralmem/sdk", "typescript"]
    DEFAULT_GO_DEPS: list[str] = []
    DEFAULT_RUST_DEPS: list[str] = []

    def __init__(self, output_dir: Path | str | None = None) -> None:
        self._output_dir = Path(output_dir) if output_dir else Path.cwd()
        self._history: list[ProjectTemplate] = []
        self._language_packs: dict[str, dict[str, Any]] = {
            "python": {
                "deps": self.DEFAULT_PY_DEPS,
                "dev_deps": ["pytest>=8.0", "ruff>=0.4"],
                "ext": ".py",
                "package_file": "pyproject.toml",
                "run_cmd": "python main.py",
            },
            "javascript": {
                "deps": self.DEFAULT_JS_DEPS,
                "dev_deps": ["jest"],
                "ext": ".js",
                "package_file": "package.json",
                "run_cmd": "node main.js",
            },
            "typescript": {
                "deps": self.DEFAULT_TS_DEPS,
                "dev_deps": ["jest", "@types/node"],
                "ext": ".ts",
                "package_file": "package.json",
                "run_cmd": "npx ts-node main.ts",
            },
            "go": {
                "deps": self.DEFAULT_GO_DEPS,
                "dev_deps": [],
                "ext": ".go",
                "package_file": "go.mod",
                "run_cmd": "go run main.go",
            },
            "rust": {
                "deps": self.DEFAULT_RUST_DEPS,
                "dev_deps": [],
                "ext": ".rs",
                "package_file": "Cargo.toml",
                "run_cmd": "cargo run",
            },
        }

    def generate(
        self,
        user_need: str,
        language: str = "python",
        project_name: str | None = None,
        extra_deps: list[str] | None = None,
    ) -> ProjectTemplate:
        """Generate a complete example project from a user need description."""
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")

        name = project_name or self._slugify(user_need)
        pack = self._language_packs[language]
        deps = pack["deps"] + (extra_deps or [])
        dev_deps = pack["dev_deps"]

        files: dict[str, str] = {}
        files["main" + pack["ext"]] = self._generate_main_file(user_need, language)
        files[pack["package_file"]] = self._generate_package_file(name, language, deps, dev_deps)
        files["README.md"] = self._generate_readme(name, user_need, language, files)

        # Add test file
        files["test_main" + pack["ext"]] = self._generate_test_file(user_need, language)

        # Add config file
        files[".gitignore"] = self._generate_gitignore(language)

        template = ProjectTemplate(
            name=name,
            description=user_need,
            language=language,
            files=files,
            dependencies=deps,
            dev_dependencies=dev_deps,
            scripts={"start": pack["run_cmd"], "test": self._test_cmd(language)},
            readme=files["README.md"],
            tags=self._infer_tags(user_need, language),
        )
        self._history.append(template)
        return template

    def generate_from_memory(
        self,
        memory_content: str,
        language: str = "python",
        project_name: str | None = None,
    ) -> ProjectTemplate:
        """Generate a project based on stored memory content."""
        need = self._extract_need_from_memory(memory_content)
        return self.generate(need, language=language, project_name=project_name)

    def scaffold_to_disk(
        self, template: ProjectTemplate, base_dir: Path | str | None = None
    ) -> Path:
        """Write a project template to disk."""
        root = Path(base_dir) if base_dir else self._output_dir / template.name
        root.mkdir(parents=True, exist_ok=True)
        for filename, content in template.files.items():
            file_path = root / filename
            file_path.write_text(content, encoding="utf-8")
        return root

    def get_history(self) -> list[ProjectTemplate]:
        """Return all generated project templates."""
        return self._history.copy()

    def export_to_json(self, template: ProjectTemplate, output_path: Path | str) -> Path:
        """Export a project template to JSON."""
        path = Path(output_path)
        path.write_text(
            json.dumps(template.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def build_file_tree(self, template: ProjectTemplate) -> FileNode:
        """Build a hierarchical file tree from a flat file dict."""
        root = FileNode(path=template.name, is_directory=True)
        for filepath in sorted(template.files.keys()):
            parts = Path(filepath).parts
            current = root
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current.children.append(
                        FileNode(path=part, content=template.files[filepath])
                    )
                else:
                    existing = next(
                        (c for c in current.children if c.path == part and c.is_directory),
                        None,
                    )
                    if existing is None:
                        existing = FileNode(path=part, is_directory=True)
                        current.children.append(existing)
                    current = existing
        return root

    # --------------------------------------------------------------------- #
    # Internal generators
    # --------------------------------------------------------------------- #

    def _generate_main_file(self, user_need: str, language: str) -> str:
        """Generate the main source file."""
        if language == "python":
            return self._python_main(user_need)
        if language == "javascript":
            return self._javascript_main(user_need)
        if language == "typescript":
            return self._typescript_main(user_need)
        if language == "go":
            return self._go_main(user_need)
        if language == "rust":
            return self._rust_main(user_need)
        return ""

    def _python_main(self, need: str) -> str:
        return f'''"""
{need}
"""
from neuralmem import NeuralMem


def main():
    mem = NeuralMem()
    # TODO: implement your workflow
    mem.store("Initial context")
    result = mem.retrieve("What was stored?")
    print("Result:", result)


if __name__ == "__main__":
    main()
'''

    def _javascript_main(self, need: str) -> str:
        return f'''// {need}
const {{ NeuralMem }} = require('@neuralmem/sdk');

async function main() {{
    const mem = new NeuralMem();
    await mem.store('Initial context');
    const result = await mem.retrieve('What was stored?');
    console.log('Result:', result);
}}

main().catch(console.error);
'''

    def _typescript_main(self, need: str) -> str:
        return f'''// {need}
import {{ NeuralMem }} from '@neuralmem/sdk';

async function main(): Promise<void> {{
    const mem = new NeuralMem();
    await mem.store('Initial context');
    const result = await mem.retrieve('What was stored?');
    console.log('Result:', result);
}}

main().catch(console.error);
'''

    def _go_main(self, need: str) -> str:
        return f'''package main

// {need}

import (
	"fmt"
	"github.com/neuralmem/go-sdk/neuralmem"
)

func main() {{
	mem := neuralmem.New()
	mem.Store("Initial context")
	result, _ := mem.Retrieve("What was stored?")
	fmt.Println("Result:", result)
}}
'''

    def _rust_main(self, need: str) -> str:
        return f'''// {need}
use neuralmem::NeuralMem;

fn main() {{
    let mem = NeuralMem::new();
    mem.store("Initial context".to_string());
    let result = mem.retrieve("What was stored?");
    println!("Result: {{:?}}", result);
}}
'''

    def _generate_test_file(self, user_need: str, language: str) -> str:
        """Generate a basic test file."""
        if language == "python":
            return f'''"""Tests for: {user_need}"""
import pytest
from main import main


def test_main_runs():
    assert main() is None
'''
        if language in ("javascript", "typescript"):
            ext = "ts" if language == "typescript" else "js"
            return f'''// Tests for: {user_need}
const {{ main }} = require('./main.{ext}');

test('main runs', async () => {{
    await expect(main()).resolves.toBeUndefined();
}});
'''
        if language == "go":
            return f'''package main

import "testing"

func TestMain(t *testing.T) {{
	// TODO: add tests for: {user_need}
}}
'''
        if language == "rust":
            return f'''#[cfg(test)]
mod tests {{
    #[test]
    fn it_works() {{
        // TODO: add tests for: {user_need}
        assert_eq!(2 + 2, 4);
    }}
}}
'''
        return ""

    def _generate_package_file(
        self, name: str, language: str, deps: list[str], dev_deps: list[str]
    ) -> str:
        """Generate language-specific package/dependency file."""
        if language == "python":
            deps_str = "\n".join(f'    "{d}",' for d in deps)
            dev_str = "\n".join(f'    "{d}",' for d in dev_deps)
            return f'''[project]
name = "{name}"
version = "0.1.0"
dependencies = [
{deps_str}
]

[project.optional-dependencies]
dev = [
{dev_str}
]
'''
        if language in ("javascript", "typescript"):
            deps_dict: dict[str, str] = {}
            for d in deps:
                if ">=" in d:
                    deps_dict[d.split(">=")[0]] = f"^{d.split('>=')[1]}"
                else:
                    deps_dict[d] = "^1.0.0"
            dev_dict: dict[str, str] = {}
            for d in dev_deps:
                if ">=" in d:
                    dev_dict[d.split(">=")[0]] = f"^{d.split('>=')[1]}"
                else:
                    dev_dict[d] = "^1.0.0"
            return json.dumps(
                {
                    "name": name,
                    "version": "0.1.0",
                    "dependencies": deps_dict,
                    "devDependencies": dev_dict,
                },
                indent=2,
            )
        if language == "go":
            dep_lines = "\n".join(f"require {d}" for d in deps if d)
            return f'''module {name}

go 1.22

{dep_lines}
'''
        if language == "rust":
            dep_lines = "\n".join(f'{d.split(" ")[0]} = "{d.split(" ")[1] if " " in d else "1.0"}"' for d in deps)
            return f'''[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

[dependencies]
{dep_lines}
'''
        return ""

    def _generate_readme(
        self, name: str, description: str, language: str, files: dict[str, str]
    ) -> str:
        """Generate a README.md for the project."""
        pack = self._language_packs[language]
        features = "\n".join(f"- {f}" for f in self._infer_features(description))
        structure = "\n".join(sorted(files.keys()))
        return self.TEMPLATE_README.format(
            name=name,
            description=description,
            features=features,
            install_cmd=f"# Install dependencies for {language}",
            usage_cmd=pack["run_cmd"],
            structure=structure,
        )

    def _generate_gitignore(self, language: str) -> str:
        """Generate a .gitignore file."""
        common = "__pycache__/\n*.pyc\n.env\nnode_modules/\ntarget/\n*.log\n"
        if language == "python":
            return common + "venv/\n.venv/\ndist/\n*.egg-info/\n"
        if language in ("javascript", "typescript"):
            return common + "build/\ndist/\n*.tsbuildinfo\n"
        if language == "go":
            return common + "bin/\n"
        if language == "rust":
            return common + "Cargo.lock\n"
        return common

    def _test_cmd(self, language: str) -> str:
        """Return the test command for a language."""
        cmds = {
            "python": "pytest",
            "javascript": "jest",
            "typescript": "jest",
            "go": "go test ./...",
            "rust": "cargo test",
        }
        return cmds.get(language, "echo 'No tests configured'")

    def _infer_tags(self, need: str, language: str) -> list[str]:
        """Infer project tags from need description."""
        tags = [language, "neuralmem", "example"]
        lower = need.lower()
        keyword_map = {
            "agent": "agent",
            "memory": "memory",
            "cache": "cache",
            "vector": "vector-search",
            "embedding": "embedding",
            "api": "api",
            "cli": "cli",
            "web": "web",
            "fastapi": "fastapi",
            "flask": "flask",
        }
        for kw, tag in keyword_map.items():
            if kw in lower and tag not in tags:
                tags.append(tag)
        return tags

    def _infer_features(self, need: str) -> list[str]:
        """Infer feature list from need description."""
        features = ["NeuralMem integration", "Persistent memory storage"]
        lower = need.lower()
        if "search" in lower:
            features.append("Semantic search")
        if "api" in lower or "rest" in lower:
            features.append("REST API endpoints")
        if "cache" in lower:
            features.append("Intelligent caching")
        if "agent" in lower:
            features.append("Agent context management")
        return features

    def _extract_need_from_memory(self, content: str) -> str:
        """Extract a project need description from memory content."""
        lines = content.strip().splitlines()
        for line in lines[:10]:
            stripped = line.strip()
            if stripped and len(stripped) > 20:
                return stripped[:200]
        return content[:200]

    def _slugify(self, text: str) -> str:
        """Convert text to a project-friendly slug."""
        slug = re.sub(r"[^\w\s-]", "", text.lower())
        slug = re.sub(r"[-\s]+", "-", slug).strip("-")
        return slug[:50] or "neuralmem-project"
