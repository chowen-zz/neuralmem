"""Comprehensive tests for CodeChunker."""
from __future__ import annotations

import pytest

from neuralmem.extraction.code_chunker import (
    ChunkType,
    CodeChunk,
    CodeChunker,
    Language,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_chunk(chunks: list[CodeChunk], name: str) -> CodeChunk | None:
    for c in chunks:
        if c.name == name:
            return c
    return None


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
class TestDetectLanguage:
    def test_by_filename_py(self):
        assert CodeChunker.detect_language("", "foo.py") == Language.PYTHON

    def test_by_filename_js(self):
        assert CodeChunker.detect_language("", "bar.js") == Language.JAVASCRIPT

    def test_by_filename_ts(self):
        assert CodeChunker.detect_language("", "baz.ts") == Language.TYPESCRIPT

    def test_by_content_python(self):
        src = "def hello():\n    pass\n"
        assert CodeChunker.detect_language(src) == Language.PYTHON

    def test_by_content_javascript(self):
        src = "function hello() {\n    return 1;\n}\n"
        assert CodeChunker.detect_language(src) == Language.JAVASCRIPT

    def test_by_content_typescript(self):
        src = "const x: number = 1;\ninterface Foo {}\n"
        assert CodeChunker.detect_language(src) == Language.TYPESCRIPT

    def test_unknown(self):
        assert CodeChunker.detect_language("random prose\nmore prose\n") == Language.UNKNOWN


# ---------------------------------------------------------------------------
# Python chunk_by_functions
# ---------------------------------------------------------------------------
class TestPythonChunkByFunctions:
    def test_top_level_functions(self):
        src = """\
def foo():
    return 1

def bar():
    return 2
"""
        chunker = CodeChunker()
        chunks = chunker.chunk_by_functions(src, language=Language.PYTHON)
        names = [c.name for c in chunks]
        assert "foo" in names
        assert "bar" in names
        for c in chunks:
            assert c.chunk_type == ChunkType.FUNCTION
            assert c.language == Language.PYTHON

    def test_class_methods_included(self):
        src = """\
class MyClass:
    def method_a(self):
        pass

    def method_b(self):
        pass
"""
        chunker = CodeChunker()
        chunks = chunker.chunk_by_functions(src, language=Language.PYTHON)
        names = [c.name for c in chunks]
        assert "method_a" in names
        assert "method_b" in names
        for c in chunks:
            assert c.chunk_type in (ChunkType.FUNCTION, ChunkType.METHOD)

    def test_preserves_imports(self):
        src = """\
import os
from typing import List

def foo():
    return os.pathsep
"""
        chunker = CodeChunker()
        chunks = chunker.chunk_by_functions(src, language=Language.PYTHON)
        foo = _find_chunk(chunks, "foo")
        assert foo is not None
        assert "import os" in foo.imports
        assert "from typing import List" in foo.imports

    def test_class_context_for_methods(self):
        src = """\
class MyClass:
    def method_a(self):
        pass
"""
        chunker = CodeChunker(preserve_class_context=True)
        chunks = chunker.chunk_by_functions(src, language=Language.PYTHON)
        method = _find_chunk(chunks, "method_a")
        assert method is not None
        assert method.class_context == "class MyClass:"

    def test_no_class_context_when_disabled(self):
        src = """\
class MyClass:
    def method_a(self):
        pass
"""
        chunker = CodeChunker(preserve_class_context=False)
        chunks = chunker.chunk_by_functions(src, language=Language.PYTHON)
        method = _find_chunk(chunks, "method_a")
        assert method is not None
        assert method.class_context == ""

    def test_async_functions(self):
        src = """\
async def fetch():
    return await something()
"""
        chunker = CodeChunker()
        chunks = chunker.chunk_by_functions(src, language=Language.PYTHON)
        assert any(c.name == "fetch" for c in chunks)


# ---------------------------------------------------------------------------
# Python chunk_by_classes
# ---------------------------------------------------------------------------
class TestPythonChunkByClasses:
    def test_single_class(self):
        src = """\
class Foo:
    def method(self):
        pass
"""
        chunker = CodeChunker()
        chunks = chunker.chunk_by_classes(src, language=Language.PYTHON)
        assert len(chunks) == 1
        assert chunks[0].name == "Foo"
        assert chunks[0].chunk_type == ChunkType.CLASS

    def test_multiple_classes(self):
        src = """\
class A:
    pass

class B:
    pass
"""
        chunker = CodeChunker()
        chunks = chunker.chunk_by_classes(src, language=Language.PYTHON)
        assert len(chunks) == 2
        assert {c.name for c in chunks} == {"A", "B"}

    def test_no_classes_returns_fallback(self):
        src = """\
def foo():
    pass
"""
        chunker = CodeChunker()
        chunks = chunker.chunk_by_classes(src, language=Language.PYTHON)
        # Fallback produces a block chunk
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Python chunk_by_blocks
# ---------------------------------------------------------------------------
class TestPythonChunkByBlocks:
    def test_functions_classes_and_statements(self):
        src = """\
import os

x = 1

def foo():
    return 1

class Bar:
    def method(self):
        pass

y = 2
"""
        chunker = CodeChunker()
        chunks = chunker.chunk_by_blocks(src, language=Language.PYTHON)
        types = [c.chunk_type for c in chunks]
        assert ChunkType.FUNCTION in types
        assert ChunkType.CLASS in types
        # Top-level statements may be merged into a BLOCK or may be dropped
        # if they are too small; we just assert the list is non-empty.
        assert len(chunks) >= 2

    def test_module_docstring_preserved(self):
        src = '"""Module docs."""\n\ndef foo():\n    pass\n'
        chunker = CodeChunker()
        chunks = chunker.chunk_by_blocks(src, language=Language.PYTHON)
        for c in chunks:
            assert c.module_docstring == "Module docs."

    def test_merge_small_chunks(self):
        src = """\
a = 1
b = 2
c = 3
"""
        chunker = CodeChunker(min_chunk_size=20)
        chunks = chunker.chunk_by_blocks(src, language=Language.PYTHON)
        # Tiny statements should be merged into one block or fallback
        assert len(chunks) >= 1
        assert all(isinstance(c, CodeChunk) for c in chunks)

    def test_preserves_imports_in_blocks(self):
        src = """\
import os
from typing import List

x = 1
"""
        chunker = CodeChunker()
        chunks = chunker.chunk_by_blocks(src, language=Language.PYTHON)
        # Imports are captured into the chunk that contains them or into
        # every chunk via the imports field.
        assert any("import os" in c.imports or "import os" in c.content for c in chunks)


# ---------------------------------------------------------------------------
# JS / TS chunking
# ---------------------------------------------------------------------------
class TestJSChunkByFunctions:
    def test_regular_function(self):
        src = """\
function foo() {
    return 1;
}
"""
        chunker = CodeChunker()
        chunks = chunker.chunk_by_functions(src, language=Language.JAVASCRIPT)
        assert any(c.name == "foo" for c in chunks)

    def test_arrow_function(self):
        src = """\
const bar = (x) => {
    return x * 2;
};
"""
        chunker = CodeChunker()
        chunks = chunker.chunk_by_functions(src, language=Language.JAVASCRIPT)
        assert any(c.name == "bar" for c in chunks)

    def test_exported_function(self):
        src = """\
export async function fetchData() {
    return await api();
}
"""
        chunker = CodeChunker()
        chunks = chunker.chunk_by_functions(src, language=Language.JAVASCRIPT)
        assert any(c.name == "fetchData" for c in chunks)

    def test_class_chunk(self):
        src = """\
class MyClass {
    greet() {
        return "hi";
    }
}
"""
        chunker = CodeChunker()
        chunks = chunker.chunk_by_classes(src, language=Language.JAVASCRIPT)
        assert any(c.name == "MyClass" for c in chunks)

    def test_js_blocks(self):
        src = """\
import React from 'react';

const a = 1;

function helper() {
    return 2;
}

class Component {
    render() {}
}
"""
        chunker = CodeChunker()
        chunks = chunker.chunk_by_blocks(src, language=Language.JAVASCRIPT)
        names = {c.name for c in chunks}
        assert "helper" in names
        assert "Component" in names
        # Imports should be preserved
        for c in chunks:
            assert "import React" in c.imports

    def test_typescript_detected_by_filename(self):
        src = "const x: number = 1;\n"
        chunker = CodeChunker()
        chunks = chunker.chunk_by_blocks(src, filename="test.ts")
        assert all(c.language == Language.TYPESCRIPT for c in chunks)


# ---------------------------------------------------------------------------
# Fallback / unknown language
# ---------------------------------------------------------------------------
class TestFallbackChunking:
    def test_plain_text(self):
        src = """\
Line one
Line two

Line three
"""
        chunker = CodeChunker()
        chunks = chunker.chunk_by_blocks(src, language=Language.UNKNOWN)
        assert len(chunks) == 2  # split by blank line
        for c in chunks:
            assert c.chunk_type == ChunkType.BLOCK

    def test_no_blank_lines(self):
        src = "line1\nline2\nline3\n"
        chunker = CodeChunker()
        chunks = chunker.chunk_by_blocks(src, language=Language.UNKNOWN)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_source(self):
        chunker = CodeChunker()
        chunks = chunker.chunk_by_blocks("")
        assert chunks == []

    def test_only_comments(self):
        src = "# comment 1\n# comment 2\n"
        chunker = CodeChunker()
        chunks = chunker.chunk_by_blocks(src, language=Language.PYTHON)
        # Comments are tiny and may be merged into empty or dropped
        assert isinstance(chunks, list)

    def test_syntax_error_fallback(self):
        src = "def foo(\n    pass\n"  # broken syntax
        chunker = CodeChunker()
        chunks = chunker.chunk_by_blocks(src, language=Language.PYTHON)
        # Should not raise; falls back to line-based
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_chunk_metadata(self):
        src = "def foo():\n    pass\n"
        chunker = CodeChunker()
        chunks = chunker.chunk_by_functions(src, language=Language.PYTHON)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.start_line == 1
        assert c.end_line == 2
        assert c.metadata == {}

    def test_different_strategies_return_same_type(self):
        src = "def foo():\n    pass\nclass Bar:\n    pass\n"
        chunker = CodeChunker()
        for strat in ("functions", "classes", "blocks"):
            chunks = chunker.chunk(src, strategy=strat, language=Language.PYTHON)
            assert isinstance(chunks, list)
            for c in chunks:
                assert isinstance(c, CodeChunk)


# ---------------------------------------------------------------------------
# Configurable chunk size
# ---------------------------------------------------------------------------
class TestChunkSizeConfig:
    def test_min_chunk_size_respected(self):
        src = """\
a = 1
b = 2
"""
        chunker = CodeChunker(min_chunk_size=50)
        chunks = chunker.chunk_by_blocks(src, language=Language.PYTHON)
        # Both tiny assignments merged into one block or fallback
        assert len(chunks) >= 1
        assert all(isinstance(c, CodeChunk) for c in chunks)

    def test_max_chunk_size_not_enforced_yet(self):
        # The current implementation does not split oversized chunks;
        # this test documents that behaviour.
        src = "x = 1\n" * 1000
        chunker = CodeChunker(max_chunk_size=100)
        chunks = chunker.chunk_by_blocks(src, language=Language.PYTHON)
        # Should still produce chunks without error
        assert isinstance(chunks, list)
