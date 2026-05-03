"""NeuralMem CLI — neuralmem mcp / serve / add / search / stats / import / consolidate"""
from __future__ import annotations

import argparse
import json
import logging
import sys

_logger = logging.getLogger(__name__)


def cmd_mcp(args: argparse.Namespace) -> None:
    """启动 MCP Server（stdio 传输，供 Claude Desktop / Cursor 接入）"""
    from neuralmem.mcp.server import server as mcp_server

    transport = "streamable-http" if getattr(args, "http", False) else "stdio"
    print(
        "NeuralMem v0.1.0 | Apache-2.0",
        file=sys.stderr,
    )
    mcp_server.run(transport=transport)


def cmd_add(args: argparse.Namespace) -> None:
    """添加一条记忆"""
    from neuralmem.core.memory import NeuralMem

    mem = NeuralMem()
    content = " ".join(args.content)
    memories = mem.remember(content, user_id=args.user_id)
    for m in memories:
        print(f"Stored [{m.memory_type.value}] {m.id[:8]}: {m.content[:60]}")
    if not memories:
        print("No new memories stored (possibly duplicate).")


def cmd_search(args: argparse.Namespace) -> None:
    """搜索记忆"""
    from neuralmem.core.memory import NeuralMem

    mem = NeuralMem()
    query = " ".join(args.query)
    results = mem.recall(query, user_id=args.user_id, limit=args.limit)
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.2f}] ({r.retrieval_method}) {r.memory.content}")
        if getattr(args, "explain", False) and r.explanation:
            print(f"   → {r.explanation}")


def cmd_stats(args: argparse.Namespace) -> None:
    """显示记忆库统计"""
    from neuralmem.core.memory import NeuralMem

    mem = NeuralMem()
    stats = mem.get_stats()
    print(json.dumps(stats, indent=2, default=str))


def cmd_batch_add(args: argparse.Namespace) -> None:
    """批量添加记忆 — 从文件或 stdin 读取，每行一条"""
    from neuralmem.core.memory import NeuralMem

    mem = NeuralMem()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    else:
        print("Reading from stdin (one memory per line, Ctrl+D to finish):", file=sys.stderr)
        lines = [line.strip() for line in sys.stdin if line.strip()]

    if not lines:
        print("No content provided.")
        return

    mt = None
    if args.memory_type:
        from neuralmem.core.types import MemoryType

        try:
            mt = MemoryType(args.memory_type.lower())
        except ValueError:
            print(f"Invalid memory type: {args.memory_type}")
            return

    tags_list = None
    if args.tags:
        tags_list = [t.strip() for t in args.tags.split(",") if t.strip()]

    def progress(current: int, total: int, preview: str) -> None:
        if preview == "done":
            print(f"\nDone: {total} items processed.", file=sys.stderr)
        else:
            print(f"[{current + 1}/{total}] {preview}", file=sys.stderr)

    memories = mem.remember_batch(
        lines,
        user_id=args.user_id,
        memory_type=mt,
        tags=tags_list,
        progress_callback=progress,
    )
    print(f"Stored {len(memories)} memories from {len(lines)} items.")
    for m in memories:
        print(f"  [{m.memory_type.value}] {m.id[:8]}: {m.content[:60]}")


def cmd_export(args: argparse.Namespace) -> None:
    """导出记忆"""
    from neuralmem.core.memory import NeuralMem

    mem = NeuralMem()

    user_id = args.user_id if args.all_users is False else None
    result = mem.export_memories(
        user_id=user_id,
        format=args.format,
        include_embeddings=args.include_embeddings,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Exported to {args.output}")
    else:
        print(result)


def cmd_import(args: argparse.Namespace) -> None:
    """导入记忆（JSON/Markdown/CSV）"""
    from neuralmem.core.memory import NeuralMem

    mem = NeuralMem()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            data = f.read()
    else:
        print("Reading from stdin (Ctrl+D to finish):", file=sys.stderr)
        data = sys.stdin.read()

    if not data.strip():
        print("No data provided.")
        return

    memories = mem.import_memories(data, format=args.format)
    print(f"Imported {len(memories)} memories.")
    for m in memories:
        print(f"  [{m.get('memory_type', 'episodic')}] {str(m.get('content', ''))[:60]}")


def cmd_forget_batch(args: argparse.Namespace) -> None:
    """批量删除记忆"""
    from neuralmem.core.memory import NeuralMem

    mem = NeuralMem()

    memory_ids = None
    if args.ids:
        memory_ids = [mid.strip() for mid in args.ids.split(",") if mid.strip()]

    tags_list = None
    if args.tags:
        tags_list = [t.strip() for t in args.tags.split(",") if t.strip()]

    result = mem.forget_batch(
        memory_ids=memory_ids,
        user_id=args.user_id if not memory_ids else None,
        tags=tags_list,
        dry_run=args.dry_run,
    )

    prefix = "[DRY RUN] " if result["dry_run"] else ""
    print(f"{prefix}Would delete {result['count']} memories." if result["dry_run"]
          else f"Deleted {result['count']} memories.")
    if result["memory_ids"]:
        for mid in result["memory_ids"]:
            print(f"  {mid[:12]}")


def cmd_consolidate(args: argparse.Namespace) -> None:
    """运行记忆整理：衰减、合并、清理"""
    from neuralmem.core.memory import NeuralMem

    mem = NeuralMem()
    result = mem.consolidate(user_id=args.user_id)
    print("Consolidation complete:")
    print(f"  Decayed:   {result['decayed']} memories")
    print(f"  Forgotten: {result['forgotten']} memories")
    print(f"  Merged:    {result['merged']} memories")


def cmd_reflect(args: argparse.Namespace) -> None:
    """对某个主题进行反思 — 检索相关记忆并综合洞察"""
    from neuralmem.core.memory import NeuralMem

    mem = NeuralMem()
    topic = " ".join(args.topic)
    result = mem.reflect(topic=topic, limit=args.limit)
    print(result)


def cmd_get(args: argparse.Namespace) -> None:
    """Get a memory by ID."""
    from neuralmem.core.memory import NeuralMem

    mem = NeuralMem()
    memory = mem.get(args.memory_id)
    if memory is None:
        print(f"Memory {args.memory_id[:8]} not found.")
        return
    print(f"ID:         {memory.id}")
    print(f"Content:    {memory.content}")
    print(f"Type:       {memory.memory_type.value}")
    print(f"Importance: {memory.importance}")
    print(f"Active:     {memory.is_active}")
    print(f"Created:    {memory.created_at}")
    print(f"Updated:    {memory.updated_at}")
    print(f"Tags:       {list(memory.tags)}")


def cmd_update(args: argparse.Namespace) -> None:
    """Update a memory's content."""
    from neuralmem.core.memory import NeuralMem

    mem = NeuralMem()
    new_content = " ".join(args.content)
    result = mem.update(args.memory_id, new_content)
    if result is None:
        print(f"Memory {args.memory_id[:8]} not found.")
        return
    print(f"Updated memory {result.id[:8]}")
    print(f"New content: {result.content}")


def cmd_history(args: argparse.Namespace) -> None:
    """Show version history for a memory."""
    from neuralmem.core.memory import NeuralMem

    mem = NeuralMem()
    entries = mem.history(args.memory_id)
    if not entries:
        print(f"No history found for {args.memory_id[:8]}.")
        return
    print(f"History for {args.memory_id[:8]} ({len(entries)} entries):")
    for e in entries:
        ts = e.changed_at.strftime("%Y-%m-%d %H:%M") if hasattr(
            e.changed_at, "strftime"
        ) else str(e.changed_at)
        old = (e.old_content[:50] + "..."
               if e.old_content and len(e.old_content) > 50
               else e.old_content or "-")
        new = (e.new_content[:50] + "..."
               if len(e.new_content) > 50
               else e.new_content)
        print(f"  [{ts}] {e.event}: {old} -> {new}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="neuralmem",
        description="NeuralMem — Local-first agent memory system",
    )
    parser.add_argument("--user-id", default="default", help="User identifier")
    parser.add_argument("--db", default=None, help="Database path override")
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    # mcp subcommand
    p_mcp = sub.add_parser("mcp", help="Start MCP server (stdio)")
    p_mcp.add_argument("--http", action="store_true", help="Use HTTP transport")
    p_mcp.set_defaults(func=cmd_mcp)

    # serve = alias for mcp
    p_serve = sub.add_parser("serve", help="Alias for 'mcp'")
    p_serve.add_argument("--http", action="store_true")
    p_serve.set_defaults(func=cmd_mcp)

    # add subcommand
    p_add = sub.add_parser("add", help="Add a memory")
    p_add.add_argument("content", nargs="+", help="Memory content")
    p_add.set_defaults(func=cmd_add)

    # search subcommand
    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("-k", "--limit", type=int, default=5)
    p_search.add_argument("--explain", action="store_true", help="Show explanation for each result")
    p_search.set_defaults(func=cmd_search)

    # stats subcommand
    p_stats = sub.add_parser("stats", help="Show memory statistics")
    p_stats.set_defaults(func=cmd_stats)

    # batch-add subcommand
    p_batch_add = sub.add_parser("batch-add", help="Batch add memories from file or stdin")
    p_batch_add.add_argument("-f", "--file", default=None, help="Input file (one memory per line)")
    p_batch_add.add_argument("-t", "--memory-type", default=None, help="Memory type for all items")
    p_batch_add.add_argument("--tags", default=None, help="Comma-separated tags for all items")
    p_batch_add.set_defaults(func=cmd_batch_add)

    # export subcommand
    p_export = sub.add_parser("export", help="Export memories to file or stdout")
    p_export.add_argument("-f", "--format", default="json", choices=["json", "markdown", "csv"],
                          help="Export format (default: json)")
    p_export.add_argument("-o", "--output", default=None, help="Output file path")
    p_export.add_argument("--all-users", action="store_true", default=False,
                          help="Export all users (default: current user only)")
    p_export.add_argument("--include-embeddings", action="store_true", default=False,
                          help="Include embedding vectors in JSON export")
    p_export.set_defaults(func=cmd_export)

    # import subcommand
    p_import = sub.add_parser("import", help="Import memories from file or stdin")
    p_import.add_argument("-f", "--file", default=None, help="Input file path")
    p_import.add_argument("--format", default="json", choices=["json", "markdown", "csv"],
                          help="Import format (default: json)")
    p_import.set_defaults(func=cmd_import)

    # forget-batch subcommand
    p_forget_batch = sub.add_parser("forget-batch", help="Batch delete memories")
    p_forget_batch.add_argument("--ids", default=None, help="Comma-separated memory IDs to delete")
    p_forget_batch.add_argument(
        "--tags", default=None,
        help="Delete memories with these tags (comma-separated)",
    )
    p_forget_batch.add_argument("--dry-run", action="store_true", default=False,
                                help="Preview what would be deleted without actually deleting")
    p_forget_batch.set_defaults(func=cmd_forget_batch)

    # consolidate subcommand
    p_consolidate = sub.add_parser(
        "consolidate", help="Run memory consolidation (decay, merge, cleanup)"
    )
    p_consolidate.set_defaults(func=cmd_consolidate)

    # reflect subcommand
    p_reflect = sub.add_parser(
        "reflect", help="Reflect on a topic — synthesize insights from memories"
    )
    p_reflect.add_argument("topic", nargs="+", help="Topic to reflect on")
    p_reflect.add_argument("-k", "--limit", type=int, default=10, help="Max memories to consider")
    p_reflect.set_defaults(func=cmd_reflect)

    # get subcommand
    p_get = sub.add_parser("get", help="Get a memory by ID")
    p_get.add_argument("memory_id", help="Memory ID")
    p_get.set_defaults(func=cmd_get)

    # update subcommand
    p_update = sub.add_parser("update", help="Update a memory's content")
    p_update.add_argument("memory_id", help="Memory ID")
    p_update.add_argument("content", nargs="+", help="New content text")
    p_update.set_defaults(func=cmd_update)

    # history subcommand
    p_history = sub.add_parser("history", help="Show version history for a memory")
    p_history.add_argument("memory_id", help="Memory ID")
    p_history.set_defaults(func=cmd_history)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
