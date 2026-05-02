"""实体节点辅助函数"""
from __future__ import annotations

from neuralmem.core.types import Entity


def entity_to_node_attrs(entity: Entity) -> dict:
    """将 Entity 转换为 NetworkX 节点属性字典"""
    return {
        "name": entity.name,
        "entity_type": entity.entity_type,
        "aliases": list(entity.aliases),
        "attributes": dict(entity.attributes),
        "first_seen": entity.first_seen.isoformat(),
        "last_seen": entity.last_seen.isoformat(),
    }


def node_attrs_to_entity(entity_id: str, attrs: dict) -> Entity:
    """从 NetworkX 节点属性重建 Entity"""
    from datetime import datetime
    return Entity(
        id=entity_id,
        name=attrs["name"],
        entity_type=attrs.get("entity_type", "unknown"),
        aliases=tuple(attrs.get("aliases", [])),
        attributes=attrs.get("attributes", {}),
        first_seen=datetime.fromisoformat(attrs["first_seen"]),
        last_seen=datetime.fromisoformat(attrs["last_seen"]),
    )
