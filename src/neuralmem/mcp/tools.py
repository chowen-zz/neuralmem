"""MCP 工具输入/输出辅助函数"""
from __future__ import annotations
from neuralmem.core.types import MemoryType, SearchResult


def format_search_results(results: list[SearchResult]) -> str:
    """将 SearchResult 列表格式化为 MCP 工具输出字符串"""
    if not results:
        return "No relevant memories found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. [{r.score:.2f}] ({r.retrieval_method}) "
            f"[{r.memory.memory_type.value}] {r.memory.content}"
        )
    return "\n".join(lines)


def parse_memory_type(value: str) -> MemoryType | None:
    """安全解析 memory_type 字符串"""
    if not value:
        return None
    try:
        return MemoryType(value.lower())
    except ValueError:
        return None


def parse_tags(tags_str: str) -> list[str] | None:
    """解析逗号分隔的标签字符串"""
    if not tags_str:
        return None
    return [t.strip() for t in tags_str.split(",") if t.strip()]
