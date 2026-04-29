"""NeuralMem CLI — neuralmem mcp / serve / add / search / stats"""
from __future__ import annotations
import argparse
import json
import sys
import logging

_logger = logging.getLogger(__name__)


def cmd_mcp(args: argparse.Namespace) -> None:
    """启动 MCP Server（stdio 传输，供 Claude Desktop / Cursor 接入）"""
    from neuralmem.mcp.server import mcp
    transport = "streamable-http" if getattr(args, "http", False) else "stdio"
    print(
        "NeuralMem v0.1.0 | AGPL-3.0 | Commercial license: neuralmem.dev/pricing",
        file=sys.stderr,
    )
    mcp.run(transport=transport)


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


def cmd_stats(args: argparse.Namespace) -> None:
    """显示记忆库统计"""
    from neuralmem.core.memory import NeuralMem
    mem = NeuralMem()
    stats = mem.get_stats()
    print(json.dumps(stats, indent=2, default=str))


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
    p_search.set_defaults(func=cmd_search)

    # stats subcommand
    p_stats = sub.add_parser("stats", help="Show memory statistics")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
