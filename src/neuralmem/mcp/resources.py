"""MCP Resource 定义"""
from __future__ import annotations

from neuralmem.core.memory import NeuralMem


def get_stats_resource(engine: NeuralMem, user_id: str = "default") -> str:
    """生成 neuralmem://stats Resource 内容"""
    stats = engine.get_stats()
    lines = [
        f"User: {user_id}",
        f"Total memories: {stats.get('total', 0)}",
        f"By type: {stats.get('by_type', {})}",
        f"Graph nodes: {stats.get('node_count', 0)}",
        f"Graph edges: {stats.get('edge_count', 0)}",
    ]
    return "\n".join(lines)
