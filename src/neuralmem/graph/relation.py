"""关系边辅助函数"""
from __future__ import annotations
from neuralmem.core.types import Relation


def relation_to_edge_attrs(relation: Relation) -> dict:
    """将 Relation 转换为 NetworkX 边属性字典"""
    return {
        "relation_type": relation.relation_type,
        "weight": relation.weight,
        "timestamp": relation.timestamp.isoformat(),
        "metadata": dict(relation.metadata),
    }
