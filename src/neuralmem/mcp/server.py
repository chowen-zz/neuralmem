"""NeuralMem MCP Server — 连接任意 MCP 客户端（Claude Desktop / Cursor 等）"""
from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from neuralmem.core.memory import NeuralMem
from neuralmem.mcp.resources import get_stats_resource
from neuralmem.mcp.tools import (
    format_search_results,
    parse_memory_type,
    parse_tags,
)

_logger = logging.getLogger(__name__)

# FastMCP 实例 — 导出别名 `mcp` 兼容 CLI entry_point
server = FastMCP("neuralmem")
mcp = server  # alias for CLI: `from neuralmem.mcp.server import mcp`

_mem: NeuralMem | None = None


def _get_mem() -> NeuralMem:
    global _mem
    if _mem is None:
        _mem = NeuralMem()
    return _mem


# ==================== Core CRUD ====================


@server.tool()
def remember(content: str, user_id: str = "", tags: str = "", memory_type: str = "") -> str:
    """Store a new memory."""
    mem = _get_mem()
    mt = parse_memory_type(memory_type)
    tag_list = parse_tags(tags)
    kwargs = {}
    if user_id:
        kwargs["user_id"] = user_id
    if tag_list:
        kwargs["tags"] = tag_list
    if mt:
        kwargs["memory_type"] = mt
    memories = mem.remember(content, **kwargs)
    return f"Remembered {len(memories)} memory item(s). IDs: {[m.id[:8] for m in memories]}"


@server.tool()
def recall(
    query: str,
    user_id: str = "",
    limit: int = 10,
    explain: bool = False,
) -> str:
    """Search memories by semantic similarity + keyword + graph + temporal."""
    mem = _get_mem()
    kwargs = {}
    if user_id:
        kwargs["user_id"] = user_id
    results = mem.recall(query, limit=limit, **kwargs)
    return format_search_results(results, show_explanations=explain)


@server.tool()
def reflect(topic: str, limit: int = 10) -> str:
    """Reflect on a topic — retrieve related memories and synthesize insights."""
    mem = _get_mem()
    return mem.reflect(topic=topic, limit=limit)


@server.tool()
def forget(memory_id: str) -> str:
    """Delete a memory by ID."""
    mem = _get_mem()
    mem.forget(memory_id=memory_id)
    return f"Deleted memory {memory_id[:8]}"


# ==================== Update / Conflict ====================


@server.tool()
def update_memory(memory_id: str, new_content: str = "", importance: float = -1) -> str:
    """Update a memory's content or importance score."""
    mem = _get_mem()
    updates = {}
    if new_content:
        updates["content"] = new_content
    if importance >= 0:
        updates["importance"] = importance
    if not updates:
        return "No updates provided."
    mem.storage.update_memory(memory_id, **updates)
    return f"Updated memory {memory_id[:8]}"


@server.tool()
def resolve_conflict(
    memory_id: str,
    action: str = "reactivate",
) -> str:
    """Resolve a conflict by reactivating or deleting a superseded memory.

    Args:
        memory_id: The memory to resolve.
        action: 'reactivate' or 'delete'.
    """
    mem = _get_mem()
    success = mem.resolve_conflict(memory_id, action=action)
    if success:
        return f"Conflict resolved: {action} on {memory_id[:8]}"
    return f"No action taken for {memory_id[:8]} (not found or not superseded)."


# ==================== Batch Operations ====================


@server.tool()
def remember_batch(
    contents: list[str],
    user_id: str = "",
    tags: str = "",
    memory_type: str = "",
) -> str:
    """Store multiple memories in batch."""
    mem = _get_mem()
    kwargs = {}
    if user_id:
        kwargs["user_id"] = user_id
    tag_list = parse_tags(tags)
    if tag_list:
        kwargs["tags"] = tag_list
    mt = parse_memory_type(memory_type)
    if mt:
        kwargs["memory_type"] = mt
    memories = mem.remember_batch(contents, **kwargs)
    return (
        f"Batch stored {len(memories)} memory item(s). "
        f"IDs: {[m.id[:8] for m in memories]}"
    )


@server.tool()
def forget_batch(
    ids: str = "",
    tags: str = "",
    user_id: str = "",
    dry_run: bool = False,
) -> str:
    """Batch delete memories by IDs or tags."""
    mem = _get_mem()
    memory_ids = [i.strip() for i in ids.split(",") if i.strip()] if ids else None
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    kwargs: dict = {}
    if memory_ids:
        kwargs["memory_ids"] = memory_ids
    if tag_list:
        kwargs["tags"] = tag_list
    if user_id:
        kwargs["user_id"] = user_id
    if dry_run:
        kwargs["dry_run"] = dry_run
    result = mem.forget_batch(**kwargs)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ==================== Export / Import ====================


@server.tool()
def export_memories(
    user_id: str = "",
    format: str = "json",
    include_embeddings: bool = False,
) -> str:
    """Export memories in various formats (json, markdown, csv)."""
    mem = _get_mem()
    kwargs = {}
    if user_id:
        kwargs["user_id"] = user_id
    return mem.export_memories(
        format=format,
        include_embeddings=include_embeddings,
        **kwargs,
    )


@server.tool()
def import_memories(data: str, format: str = "json") -> str:
    """Import memories from JSON, Markdown, or CSV data."""
    mem = _get_mem()
    imported = mem.import_memories(data, format=format)
    return f"Imported {len(imported)} memories."


# ==================== Lifecycle ====================


@server.tool()
def consolidate(user_id: str = "") -> str:
    """Run memory consolidation: decay old memories, merge similar, remove forgotten."""
    mem = _get_mem()
    kwargs = {}
    if user_id:
        kwargs["user_id"] = user_id
    result = mem.consolidate(**kwargs)
    return f"Consolidation complete: {result}"


@server.tool()
def cleanup_expired() -> str:
    """Remove all memories whose expiration time has passed."""
    mem = _get_mem()
    count = mem.cleanup_expired()
    return f"Cleaned up {count} expired memories."


# ==================== Resources ====================


@server.resource("neuralmem://stats")
def get_stats() -> str:
    """Memory statistics."""
    return get_stats_resource(_get_mem(), user_id="default")


def main():
    """入口点"""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
