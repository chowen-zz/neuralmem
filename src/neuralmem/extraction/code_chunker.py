"""AST-aware code chunker for NeuralMem.

Supports Python, JavaScript, TypeScript via AST parsing (where available)
and regex/line-based fallback for all languages.

Chunking strategies:
  - chunk_by_functions:  split by top-level / class-level functions
  - chunk_by_classes:    split by class definitions
  - chunk_by_blocks:     split by logical blocks (functions + classes +
                         top-level statements grouped by blank-line boundaries)

Each chunk preserves context (imports, class context for methods, module docstring)
and carries metadata: language, chunk_type, name, start_line, end_line.
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_logger = logging.getLogger(__name__)


class ChunkType(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    BLOCK = "block"
    IMPORTS = "imports"
    UNKNOWN = "unknown"


class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    UNKNOWN = "unknown"


@dataclass
class CodeChunk:
    content: str
    language: Language
    chunk_type: ChunkType
    name: str = ""
    start_line: int = 0
    end_line: int = 0
    # Context preserved for downstream use
    imports: str = ""
    class_context: str = ""
    module_docstring: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class CodeChunker:
    """Parse source code and emit structured chunks."""

    def __init__(
        self,
        max_chunk_size: int = 4000,
        min_chunk_size: int = 10,
        preserve_imports: bool = True,
        preserve_class_context: bool = True,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.preserve_imports = preserve_imports
        self.preserve_class_context = preserve_class_context

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------
    @staticmethod
    def detect_language(source: str, filename: str | None = None) -> Language:
        if filename:
            lower = filename.lower()
            if lower.endswith(".py"):
                return Language.PYTHON
            if lower.endswith(".js") or lower.endswith(".jsx"):
                return Language.JAVASCRIPT
            if lower.endswith(".ts") or lower.endswith(".tsx"):
                return Language.TYPESCRIPT

        # Heuristic: look for Python-specific syntax
        if re.search(r"^\s*(def |class |import |from .* import )", source, re.M):
            return Language.PYTHON
        # JS/TS heuristic
        if re.search(r"^\s*(const |let |var |function |export |import )", source, re.M):
            if "type " in source or "interface " in source or ": " in source.split("{")[0]:
                return Language.TYPESCRIPT
            return Language.JAVASCRIPT
        return Language.UNKNOWN

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chunk(
        self,
        source: str,
        strategy: str = "blocks",
        filename: str | None = None,
        language: Language | None = None,
    ) -> list[CodeChunk]:
        """Chunk *source* using the requested *strategy*.

        Strategies: ``functions``, ``classes``, ``blocks`` (default).
        """
        lang = language or self.detect_language(source, filename)
        if lang == Language.PYTHON:
            return self._chunk_python(source, strategy)
        if lang in (Language.JAVASCRIPT, Language.TYPESCRIPT):
            return self._chunk_js_ts(source, lang, strategy)
        # Fallback: treat everything as plain text blocks
        return self._chunk_fallback(source, lang)

    def chunk_by_functions(
        self, source: str, filename: str | None = None, language: Language | None = None
    ) -> list[CodeChunk]:
        return self.chunk(source, strategy="functions", filename=filename, language=language)

    def chunk_by_classes(
        self, source: str, filename: str | None = None, language: Language | None = None
    ) -> list[CodeChunk]:
        return self.chunk(source, strategy="classes", filename=filename, language=language)

    def chunk_by_blocks(
        self, source: str, filename: str | None = None, language: Language | None = None
    ) -> list[CodeChunk]:
        return self.chunk(source, strategy="blocks", filename=filename, language=language)

    # ------------------------------------------------------------------
    # Python chunking (AST-based)
    # ------------------------------------------------------------------
    def _chunk_python(self, source: str, strategy: str) -> list[CodeChunk]:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            _logger.warning("Python parse failed (%s); falling back to line-based chunking.", exc)
            return self._chunk_fallback(source, Language.PYTHON)

        lines = source.splitlines(keepends=True)
        imports = self._extract_python_imports(lines)
        module_doc = ast.get_docstring(tree) or ""

        chunks: list[CodeChunk] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if strategy == "classes":
                    continue
                chunk = self._python_function_chunk(
                    node, lines, imports, module_doc, is_method=False
                )
                chunks.append(chunk)
            elif isinstance(node, ast.ClassDef):
                if strategy == "functions":
                    # Still emit methods inside the class as separate chunks
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            chunk = self._python_function_chunk(
                                item, lines, imports, module_doc, is_method=True, class_node=node
                            )
                            chunks.append(chunk)
                elif strategy == "classes":
                    chunk = self._python_class_chunk(node, lines, imports, module_doc)
                    chunks.append(chunk)
                else:  # blocks
                    chunk = self._python_class_chunk(node, lines, imports, module_doc)
                    chunks.append(chunk)
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            chunk = self._python_function_chunk(
                                item, lines, imports, module_doc, is_method=True, class_node=node
                            )
                            chunks.append(chunk)
            else:
                if strategy == "blocks":
                    chunk = self._python_stmt_chunk(node, lines, imports, module_doc)
                    if chunk:
                        chunks.append(chunk)

        # Merge tiny orphan top-level statements into contiguous blocks.
        # Also ensure imports are attached to every chunk (they were already
        # captured during AST traversal for functions/classes, but orphaned
        # statement chunks may have dropped them when min_chunk_size filtered
        # them out).
        if strategy == "blocks":
            chunks = self._merge_small_chunks(chunks)
            # If _merge_small_chunks dropped everything because every piece was
            # below min_chunk_size, fall back to blank-line boundaries so we
            # never return an empty list for non-empty source.
            if not chunks:
                chunks = self._chunk_fallback(source, Language.PYTHON)

        return chunks or self._chunk_fallback(source, Language.PYTHON)

    def _extract_python_imports(self, lines: list[str]) -> str:
        import_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_lines.append(line)
            elif stripped.startswith("#") or not stripped:
                continue
            else:
                # Stop at first non-import code, but keep going if we hit a
                # continuation line (e.g. multi-line import in parentheses).
                if import_lines and (stripped.startswith("(") or stripped.startswith(",")):
                    import_lines.append(line)
                    continue
                break
        return "".join(import_lines)

    def _node_source(self, node: ast.AST, lines: list[str]) -> str:
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", start + 1)
        return "".join(lines[start:end])

    def _python_function_chunk(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: list[str],
        imports: str,
        module_doc: str,
        is_method: bool = False,
        class_node: ast.ClassDef | None = None,
    ) -> CodeChunk:
        src = self._node_source(node, lines)
        name = node.name
        chunk_type = ChunkType.METHOD if is_method else ChunkType.FUNCTION
        class_ctx = ""
        if is_method and class_node and self.preserve_class_context:
            class_ctx = f"class {class_node.name}:"
        return CodeChunk(
            content=src,
            language=Language.PYTHON,
            chunk_type=chunk_type,
            name=name,
            start_line=getattr(node, "lineno", 0),
            end_line=getattr(node, "end_lineno", 0),
            imports=imports if self.preserve_imports else "",
            class_context=class_ctx,
            module_docstring=module_doc,
        )

    def _python_class_chunk(
        self,
        node: ast.ClassDef,
        lines: list[str],
        imports: str,
        module_doc: str,
    ) -> CodeChunk:
        src = self._node_source(node, lines)
        return CodeChunk(
            content=src,
            language=Language.PYTHON,
            chunk_type=ChunkType.CLASS,
            name=node.name,
            start_line=getattr(node, "lineno", 0),
            end_line=getattr(node, "end_lineno", 0),
            imports=imports if self.preserve_imports else "",
            class_context="",
            module_docstring=module_doc,
        )

    def _python_stmt_chunk(
        self,
        node: ast.AST,
        lines: list[str],
        imports: str,
        module_doc: str,
    ) -> CodeChunk | None:
        # Skip imports (already captured) and docstrings
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return None
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            return None
        src = self._node_source(node, lines)
        if len(src.strip()) < self.min_chunk_size:
            return None
        return CodeChunk(
            content=src,
            language=Language.PYTHON,
            chunk_type=ChunkType.BLOCK,
            name="",
            start_line=getattr(node, "lineno", 0),
            end_line=getattr(node, "end_lineno", 0),
            imports=imports if self.preserve_imports else "",
            class_context="",
            module_docstring=module_doc,
        )

    # ------------------------------------------------------------------
    # JS / TS chunking (regex-based)
    # ------------------------------------------------------------------
    def _chunk_js_ts(self, source: str, lang: Language, strategy: str) -> list[CodeChunk]:
        lines = source.splitlines(keepends=True)
        imports = self._extract_js_imports(lines)

        chunks: list[CodeChunk] = []

        # Functions:  function name(...)  |  const name = (...) =>  |  async function ...
        func_pattern = re.compile(
            r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(|"
            r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(.*?\)\s*=>"
        )
        # Classes
        class_pattern = re.compile(r"^(?:export\s+)?class\s+(\w+)")

        # First pass: find boundaries by tracking brace depth
        boundaries = self._js_brace_boundaries(source)

        # Map line -> (name, type)
        line_info: dict[int, tuple[str, ChunkType]] = {}
        for i, line in enumerate(lines, start=1):
            m = func_pattern.search(line)
            if m:
                name = m.group(1) or m.group(2)
                line_info[i] = (name, ChunkType.FUNCTION)
            m = class_pattern.search(line)
            if m:
                line_info[i] = (m.group(1), ChunkType.CLASS)

        # Build chunks from boundaries
        for start_line, end_line in boundaries:
            name = ""
            chunk_type = ChunkType.BLOCK
            for ln in range(start_line, min(end_line + 1, start_line + 3)):
                if ln in line_info:
                    name, chunk_type = line_info[ln]
                    break

            if strategy == "functions" and chunk_type != ChunkType.FUNCTION:
                continue
            if strategy == "classes" and chunk_type != ChunkType.CLASS:
                continue

            src = "".join(lines[start_line - 1 : end_line])
            chunks.append(
                CodeChunk(
                    content=src,
                    language=lang,
                    chunk_type=chunk_type,
                    name=name,
                    start_line=start_line,
                    end_line=end_line,
                    imports=imports if self.preserve_imports else "",
                )
            )

        if strategy == "blocks":
            chunks = self._merge_small_chunks(chunks)

        return chunks or self._chunk_fallback(source, lang)

    def _extract_js_imports(self, lines: list[str]) -> str:
        import_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("export "):
                import_lines.append(line)
            elif stripped.startswith("//") or not stripped:
                continue
            else:
                break
        return "".join(import_lines)

    def _js_brace_boundaries(self, source: str) -> list[tuple[int, int]]:
        """Return (start_line, end_line) for each top-level brace-delimited block."""
        lines = source.splitlines(keepends=True)
        boundaries: list[tuple[int, int]] = []
        depth = 0
        start = 0
        in_block = False
        for i, line in enumerate(lines, start=1):
            # Ignore braces inside strings / comments (naive)
            code = re.sub(r'"(?:\\.|[^"])*"', '""', line)
            code = re.sub(r"'(?:\\.|[^'])*'", "''", code)
            code = re.sub(r"`(?:\\.|[^`])*`", "``", code)
            code = re.sub(r"//.*", "", code)
            for ch in code:
                if ch == "{":
                    if not in_block:
                        start = i
                        in_block = True
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and in_block:
                        boundaries.append((start, i))
                        in_block = False
        # If no braces found, fall back to blank-line boundaries
        if not boundaries:
            return self._blank_line_boundaries(lines)
        return boundaries

    # ------------------------------------------------------------------
    # Fallback chunking (language-agnostic)
    # ------------------------------------------------------------------
    def _chunk_fallback(self, source: str, lang: Language) -> list[CodeChunk]:
        lines = source.splitlines(keepends=True)
        boundaries = self._blank_line_boundaries(lines)
        chunks: list[CodeChunk] = []
        for start, end in boundaries:
            src = "".join(lines[start - 1 : end])
            if len(src.strip()) < self.min_chunk_size:
                continue
            chunks.append(
                CodeChunk(
                    content=src,
                    language=lang,
                    chunk_type=ChunkType.BLOCK,
                    start_line=start,
                    end_line=end,
                )
            )
        # If every boundary was below min_chunk_size, return the whole source
        # as a single block so we never return an empty list.
        if not chunks and source.strip():
            chunks.append(
                CodeChunk(
                    content=source,
                    language=lang,
                    chunk_type=ChunkType.BLOCK,
                    start_line=1,
                    end_line=len(lines),
                )
            )
        return chunks

    def _blank_line_boundaries(self, lines: list[str]) -> list[tuple[int, int]]:
        """Group consecutive non-blank lines separated by blank lines."""
        boundaries: list[tuple[int, int]] = []
        start: int | None = None
        for i, line in enumerate(lines, start=1):
            if line.strip():
                if start is None:
                    start = i
            else:
                if start is not None:
                    boundaries.append((start, i - 1))
                    start = None
        if start is not None:
            boundaries.append((start, len(lines)))
        return boundaries

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    def _merge_small_chunks(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Merge consecutive tiny chunks until they reach *min_chunk_size*."""
        if not chunks:
            return chunks
        merged: list[CodeChunk] = []
        buffer: list[CodeChunk] = []

        def flush() -> None:
            nonlocal buffer, merged
            if not buffer:
                return
            if len(buffer) == 1:
                # Even a single undersized chunk is kept so we never drop
                # the only remaining content.
                merged.append(buffer[0])
            else:
                content = "\n".join(c.content for c in buffer)
                merged.append(
                    CodeChunk(
                        content=content,
                        language=buffer[0].language,
                        chunk_type=ChunkType.BLOCK,
                        name="",
                        start_line=buffer[0].start_line,
                        end_line=buffer[-1].end_line,
                        imports=buffer[0].imports,
                        class_context="",
                        module_docstring=buffer[0].module_docstring,
                    )
                )
            buffer = []

        for chunk in chunks:
            if len(chunk.content) < self.min_chunk_size:
                buffer.append(chunk)
            else:
                flush()
                merged.append(chunk)
        flush()
        return merged
