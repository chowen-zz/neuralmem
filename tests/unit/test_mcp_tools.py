"""MCP tools/resources 单元测试"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from neuralmem.mcp.tools import format_search_results, parse_memory_type, parse_tags
from neuralmem.mcp.resources import get_stats_resource
from neuralmem.core.types import Memory, SearchResult, MemoryType


def test_parse_memory_type_valid():
    assert parse_memory_type("semantic") == MemoryType.SEMANTIC
    assert parse_memory_type("episodic") == MemoryType.EPISODIC
    assert parse_memory_type("procedural") == MemoryType.PROCEDURAL
    assert parse_memory_type("working") == MemoryType.WORKING


def test_parse_memory_type_invalid():
    assert parse_memory_type("unknown") is None
    assert parse_memory_type("") is None
    assert parse_memory_type("INVALID") is None


def test_parse_tags_basic():
    tags = parse_tags("ai, memory, agent")
    assert tags == ["ai", "memory", "agent"]


def test_parse_tags_empty():
    assert parse_tags("") is None
    # "   " 非空字符串，split 后过滤空串得到 []（空列表），不是 None
    assert parse_tags("   ") == []


def test_parse_tags_single():
    assert parse_tags("python") == ["python"]


def test_format_search_results_empty():
    result = format_search_results([])
    assert result == "No relevant memories found."


def test_format_search_results_with_items():
    m = Memory(content="User likes Python", memory_type=MemoryType.SEMANTIC)
    r = SearchResult(memory=m, score=0.85, retrieval_method="semantic")
    result = format_search_results([r])
    assert "0.85" in result
    assert "semantic" in result
    assert "Python" in result


def test_format_search_results_multiple():
    memories = [
        SearchResult(
            memory=Memory(content=f"Memory {i}"),
            score=0.9 - i * 0.1,
            retrieval_method="keyword",
        )
        for i in range(3)
    ]
    result = format_search_results(memories)
    lines = result.strip().split("\n")
    assert len(lines) == 3
    assert "1." in lines[0]
    assert "3." in lines[2]


def test_get_stats_resource():
    engine = MagicMock()
    engine.get_stats.return_value = {
        "total": 42,
        "by_type": {"semantic": 30, "episodic": 12},
        "entity_count": 15,
    }
    result = get_stats_resource(engine, "user-1")
    assert "42" in result
    assert "user-1" in result
