"""NeuralMem MCP Server — 连接任意 MCP 客户端（Claude Desktop / Cursor 等）"""
from __future__ import annotations
import logging
import asyncio
from mcp.server.fastmcp import FastMCP
from neuralmem.core.memory import NeuralMem
from neuralmem.mcp.tools import format_search_results, parse_memory_type, parse_tags
from neuralmem.mcp.resources import get_stats_resource

_logger = logging.getLogger(__name__)

MAX_RECALL_LIMIT = 100  # 防止无限扫描 DoS

mcp = FastMCP(
    "NeuralMem",
    instructions="Persistent memory for AI agents. Remember, recall, reflect.",
)

_engine: NeuralMem | None = None


def get_engine() -> NeuralMem:
    global _engine
    if _engine is None:
        _engine = NeuralMem()
    return _engine


@mcp.tool()
async def remember(
    content: str,
    memory_type: str = "semantic",
    tags: str = "",
    user_id: str = "default",
) -> str:
    """
    Store a memory. The content will be automatically analyzed to extract
    entities, relationships, and key facts. Memories persist across sessions.

    Args:
        content: What to remember (natural language text, facts, preferences)
        memory_type: One of: semantic, episodic, procedural, working
        tags: Comma-separated tags for organization
        user_id: User identifier for memory scoping
    """
    mt = parse_memory_type(memory_type)
    tag_list = parse_tags(tags)
    memories = await asyncio.to_thread(
        get_engine().remember,
        content,
        user_id=user_id,
        memory_type=mt,
        tags=tag_list,
    )
    return f"Stored {len(memories)} memories. IDs: {[m.id[:8] for m in memories]}"


@mcp.tool()
async def recall(
    query: str,
    user_id: str = "default",
    limit: int = 5,
    memory_type: str = "",
) -> str:
    """
    Retrieve relevant memories based on a query. Uses 4-strategy hybrid retrieval:
    semantic search, keyword matching, graph traversal, and temporal filtering with RRF fusion.

    Args:
        query: What to search for (natural language)
        user_id: User identifier
        limit: Maximum number of results (default: 5)
        memory_type: Filter by type (empty = all types)
    """
    limit = max(1, min(limit, MAX_RECALL_LIMIT))
    types = [parse_memory_type(memory_type)] if memory_type else None
    mt_list = [t for t in (types or []) if t is not None]
    results = await asyncio.to_thread(
        get_engine().recall,
        query,
        user_id=user_id,
        limit=limit,
        memory_types=mt_list or None,
    )
    return format_search_results(results)


@mcp.tool()
async def reflect(
    topic: str,
    user_id: str = "default",
) -> str:
    """
    Reason over stored memories about a topic. Performs multi-hop retrieval
    and graph traversal to build a comprehensive understanding.

    Args:
        topic: Subject to reflect on
        user_id: User identifier
    """
    return await asyncio.to_thread(get_engine().reflect, topic, user_id=user_id)


@mcp.tool()
async def forget(
    memory_id: str = "",
    user_id: str = "default",
    tags: str = "",
) -> str:
    """
    Delete specific memories. Supports GDPR-compliant complete deletion.

    Args:
        memory_id: Specific memory ID to delete (empty = use other filters)
        user_id: Delete all memories for this user (use with caution)
        tags: Delete memories with these tags (comma-separated)
    """
    tag_list = parse_tags(tags)
    count = await asyncio.to_thread(
        get_engine().forget,
        memory_id=memory_id or None,
        user_id=user_id if not memory_id else None,
        tags=tag_list,
    )
    return f"Deleted {count} memories."


@mcp.tool()
async def consolidate(user_id: str = "default") -> str:
    """
    Run background memory maintenance: merge duplicates, apply decay (stub in v0.1).
    Call periodically (e.g., daily) to keep memory clean.
    """
    stats = await asyncio.to_thread(get_engine().consolidate, user_id=user_id)
    return (
        f"Consolidation complete: "
        f"{stats['decayed']} decayed, "
        f"{stats['merged']} merged, "
        f"{stats['forgotten']} forgotten."
    )


@mcp.resource("neuralmem://stats/{user_id}")
async def memory_stats(user_id: str) -> str:
    """Get memory statistics for a user"""
    return await asyncio.to_thread(get_stats_resource, get_engine(), user_id)


def main() -> None:
    """CLI 入口点（由 neuralmem mcp 命令调用）"""
    import sys
    transport = "stdio"
    if "--http" in sys.argv:
        transport = "streamable-http"
    elif "--sse" in sys.argv:
        transport = "sse"
    _logger.info("Starting NeuralMem MCP Server (transport=%s)", transport)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
