"""NetworkX 知识图谱 — 内存级图谱 + 增量 SQLite 持久化"""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from typing import TYPE_CHECKING

import networkx as nx

from neuralmem.core.exceptions import GraphError, StorageError
from neuralmem.core.types import Entity, Relation
from neuralmem.graph.entity import entity_to_node_attrs, node_attrs_to_entity
from neuralmem.graph.relation import relation_to_edge_attrs

if TYPE_CHECKING:
    from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """基于 NetworkX DiGraph 的知识图谱，支持线程安全写入与增量 SQLite 持久化"""

    _SNAPSHOT_KEY = "__graph_snapshot__"

    def __init__(self, storage: StorageBackend) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._lock = threading.Lock()
        self._storage = storage
        # Dirty tracking for incremental persistence
        self._dirty_nodes: set[str] = set()
        self._dirty_edges: set[tuple[str, str]] = set()  # (source_id, target_id)
        # Batch mode: when True, mutations skip per-call persistence
        self._defer_persistence: bool = False
        self._load_snapshot()

    # ------------------------------------------------------------------
    # 实体操作
    # ------------------------------------------------------------------

    def upsert_entity(self, entity: Entity) -> None:
        """插入或更新实体节点"""
        with self._lock:
            if self._graph.has_node(entity.id):
                existing = self._graph.nodes[entity.id]
                attrs = entity_to_node_attrs(entity)
                # 保留原 first_seen，更新其余字段
                attrs["first_seen"] = existing.get("first_seen", attrs["first_seen"])
                # 保留已有的 memory_ids
                attrs["memory_ids"] = existing.get("memory_ids", [])
                self._graph.nodes[entity.id].update(attrs)
            else:
                attrs = entity_to_node_attrs(entity)
                attrs["memory_ids"] = []
                self._graph.add_node(entity.id, **attrs)
            self._dirty_nodes.add(entity.id)
        if not self._defer_persistence:
            self._save_snapshot_async()

    def get_entity(self, entity_id: str) -> Entity | None:
        """按 ID 查询实体，不存在返回 None"""
        with self._lock:
            if not self._graph.has_node(entity_id):
                return None
            return node_attrs_to_entity(entity_id, dict(self._graph.nodes[entity_id]))

    def get_entities(self, user_id: str | None = None) -> list[Entity]:
        """获取实体列表，可按 user_id 过滤。

        过滤语义：仅排除 attributes["user_id"] 明确属于其他用户的实体；
        没有 user_id 属性的实体（共享/未标注）始终包含。
        过滤依赖实体 attributes 中的 user_id 字段，需在 upsert_entity 时通过
        attributes 传入才能生效隔离。
        """
        with self._lock:
            entities = [
                node_attrs_to_entity(nid, dict(attrs))
                for nid, attrs in self._graph.nodes(data=True)
            ]
        if user_id is None:
            return entities
        return [
            e for e in entities
            if e.attributes.get("user_id") in (None, user_id)
        ]

    def find_entities(self, query: str) -> list[Entity]:
        """按名字模糊匹配（不区分大小写）"""
        q = query.lower()
        with self._lock:
            return [
                node_attrs_to_entity(nid, dict(attrs))
                for nid, attrs in self._graph.nodes(data=True)
                if q in attrs.get("name", "").lower()
            ]

    # ------------------------------------------------------------------
    # 关系操作
    # ------------------------------------------------------------------

    def add_relation(self, relation: Relation) -> None:
        """添加有向边；若边已存在则取权重平均值"""
        with self._lock:
            src, tgt = relation.source_id, relation.target_id
            attrs = relation_to_edge_attrs(relation)
            if self._graph.has_edge(src, tgt):
                old_weight = self._graph[src][tgt].get("weight", relation.weight)
                attrs["weight"] = (old_weight + relation.weight) / 2.0
                self._graph[src][tgt].update(attrs)
            else:
                self._graph.add_edge(src, tgt, **attrs)
            self._dirty_edges.add((src, tgt))
        if not self._defer_persistence:
            self._save_snapshot_async()

    # ------------------------------------------------------------------
    # 图遍历
    # ------------------------------------------------------------------

    def get_neighbors(self, entity_ids: list[str], depth: int = 1) -> list[Entity]:
        """BFS 获取邻居，排除输入节点本身"""
        seed_set = set(entity_ids)
        visited: set[str] = set(entity_ids)
        queue: deque[tuple[str, int]] = deque((eid, 0) for eid in entity_ids)
        result: list[Entity] = []

        with self._lock:
            while queue:
                nid, d = queue.popleft()
                if d >= depth:
                    continue
                for neighbor in self._graph.successors(nid):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        if neighbor not in seed_set and self._graph.has_node(neighbor):
                            result.append(
                                node_attrs_to_entity(
                                    neighbor, dict(self._graph.nodes[neighbor])
                                )
                            )
                        queue.append((neighbor, d + 1))
        return result

    def traverse_for_memories(
        self,
        entity_ids: list[str],
        depth: int = 2,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """遍历图谱收集关联记忆 ID，距离越近分数越高

        Returns:
            list of (memory_id, score)
        """
        depth_scores = {1: 0.9, 2: 0.6}
        results: dict[str, float] = {}

        visited: set[str] = set(entity_ids)
        queue: deque[tuple[str, int]] = deque((eid, 0) for eid in entity_ids)

        with self._lock:
            # 先收集种子节点自身的记忆
            for eid in entity_ids:
                if self._graph.has_node(eid):
                    for mid in self._graph.nodes[eid].get("memory_ids", []):
                        results[mid] = max(results.get(mid, 0.0), 1.0)

            while queue:
                nid, d = queue.popleft()
                if d >= depth:
                    continue
                for neighbor in self._graph.successors(nid):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        score = depth_scores.get(d + 1, 0.3)
                        if self._graph.has_node(neighbor):
                            for mid in self._graph.nodes[neighbor].get("memory_ids", []):
                                results[mid] = max(results.get(mid, 0.0), score)
                        queue.append((neighbor, d + 1))

        return list(results.items())

    # ------------------------------------------------------------------
    # 记忆关联
    # ------------------------------------------------------------------

    def link_memory_to_entity(self, memory_id: str, entity_id: str) -> None:
        """在节点上追加关联的记忆 ID"""
        with self._lock:
            if not self._graph.has_node(entity_id):
                raise GraphError(f"Entity not found: {entity_id}")
            node = self._graph.nodes[entity_id]
            node["memory_ids"] = list(set(node.get("memory_ids", []) + [memory_id]))
            self._dirty_nodes.add(entity_id)
        if not self._defer_persistence:
            self._save_snapshot_async()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _load_snapshot(self) -> None:
        """从 storage 加载图谱：优先尝试增量表，回退到 JSON 快照"""
        # 1. Try incremental tables first
        if self._load_from_tables():
            _logger.debug(
                "Graph loaded from incremental tables: %d nodes",
                self._graph.number_of_nodes(),
            )
            return
        # 2. Fall back to JSON snapshot (migration path)
        self._load_from_json()

    def _load_from_tables(self) -> bool:
        """Load graph from graph_nodes / graph_edges tables.

        Returns True if data was found and loaded.
        """
        try:
            nodes = self._storage.load_graph_nodes()
            if not nodes:
                return False

            self._graph = nx.DiGraph()
            for n in nodes:
                self._graph.add_node(
                    n["id"],
                    name=n["name"],
                    entity_type=n["entity_type"],
                    aliases=n.get("aliases", []),
                    attributes=n.get("attributes", {}),
                    first_seen=n["first_seen"],
                    last_seen=n["last_seen"],
                    memory_ids=n.get("memory_ids", []),
                )

            edges = self._storage.load_graph_edges()
            if edges:
                for e in edges:
                    self._graph.add_edge(
                        e["source_id"],
                        e["target_id"],
                        relation_type=e["relation_type"],
                        weight=e["weight"],
                        timestamp=e["timestamp"],
                        metadata=e.get("metadata", {}),
                    )
            return True
        except Exception as exc:
            _logger.debug("Failed to load graph from tables (%s)", exc)
            return False

    def _load_from_json(self) -> None:
        """Load graph from legacy JSON snapshot (migration path)."""
        try:
            raw = self._storage.load_graph_snapshot()
            if raw is None:
                return
            data = raw if isinstance(raw, dict) else json.loads(raw)
            self._graph = nx.node_link_graph(data)
            _logger.debug("Graph snapshot loaded: %d nodes", self._graph.number_of_nodes())
            # Mark all nodes/edges as dirty so next persist populates the tables
            for nid in self._graph.nodes:
                self._dirty_nodes.add(nid)
            for src, tgt in self._graph.edges:
                self._dirty_edges.add((src, tgt))
        except (json.JSONDecodeError, TypeError, KeyError, nx.NetworkXError) as exc:
            _logger.debug("No graph snapshot loaded (%s)", exc)

    def _save_snapshot_async(self) -> None:
        """后台持久化图谱（增量表 + JSON 快照向后兼容）。

        在调用线程持锁序列化脏数据，后台线程仅负责 I/O，消除锁竞争。
        """
        # Snapshot dirty items and clear the sets under lock
        with self._lock:
            if not self._dirty_nodes and not self._dirty_edges:
                # Nothing changed — still save full JSON for backward compat
                try:
                    data = nx.node_link_data(self._graph)
                except (TypeError, ValueError):
                    return

                def _worker_json() -> None:
                    try:
                        self._storage.save_graph_snapshot(data)
                        _logger.debug(
                            "Graph snapshot saved (no incremental changes): %d nodes",
                            len(data.get("nodes", [])),
                        )
                    except (StorageError, OSError) as exc:
                        _logger.debug("Graph snapshot save failed: %s", exc)

                threading.Thread(target=_worker_json, daemon=True).start()
                return

            # Collect dirty node data
            dirty_node_ids = list(self._dirty_nodes)
            dirty_edge_keys = list(self._dirty_edges)
            dirty_nodes_data: list[dict] = []
            for nid in dirty_node_ids:
                if self._graph.has_node(nid):
                    attrs = dict(self._graph.nodes[nid])
                    dirty_nodes_data.append({
                        "id": nid,
                        "name": attrs.get("name", ""),
                        "entity_type": attrs.get("entity_type", "unknown"),
                        "aliases": attrs.get("aliases", []),
                        "attributes": attrs.get("attributes", {}),
                        "first_seen": attrs.get("first_seen", ""),
                        "last_seen": attrs.get("last_seen", ""),
                        "memory_ids": attrs.get("memory_ids", []),
                    })

            # Collect dirty edge data
            dirty_edges_data: list[dict] = []
            for src, tgt in dirty_edge_keys:
                if self._graph.has_edge(src, tgt):
                    attrs = dict(self._graph[src][tgt])
                    dirty_edges_data.append({
                        "source_id": src,
                        "target_id": tgt,
                        "relation_type": attrs.get("relation_type", ""),
                        "weight": attrs.get("weight", 1.0),
                        "timestamp": attrs.get("timestamp", ""),
                        "metadata": attrs.get("metadata", {}),
                    })

            # Also snapshot full JSON for backward compatibility
            try:
                full_data = nx.node_link_data(self._graph)
            except (TypeError, ValueError):
                full_data = None

            # Clear dirty sets
            self._dirty_nodes.clear()
            self._dirty_edges.clear()

        def _worker() -> None:
            try:
                # Save incremental changes
                if dirty_nodes_data:
                    self._storage.save_graph_nodes_incremental(dirty_nodes_data)
                if dirty_edges_data:
                    self._storage.save_graph_edges_incremental(dirty_edges_data)
                _logger.debug(
                    "Graph incremental save: %d nodes, %d edges",
                    len(dirty_nodes_data),
                    len(dirty_edges_data),
                )
            except (StorageError, OSError) as exc:
                _logger.debug("Graph incremental save failed: %s", exc)

            # Also save full JSON snapshot for backward compatibility
            if full_data is not None:
                try:
                    self._storage.save_graph_snapshot(full_data)
                    _logger.debug(
                        "Graph snapshot saved: %d nodes",
                        len(full_data.get("nodes", [])),
                    )
                except (StorageError, OSError) as exc:
                    _logger.debug("Graph snapshot save failed: %s", exc)

        threading.Thread(target=_worker, daemon=True).start()

    def flush(self) -> None:
        """Synchronously persist all dirty graph data to storage.

        Call this after a batch of mutations (especially inside a batch()
        context) to persist all accumulated dirty state in one write.
        If nothing is dirty, this is a no-op.
        """
        with self._lock:
            if not self._dirty_nodes and not self._dirty_edges:
                return
            # Collect dirty data under lock
            dirty_nodes_data: list[dict] = []
            for nid in list(self._dirty_nodes):
                if self._graph.has_node(nid):
                    attrs = dict(self._graph.nodes[nid])
                    dirty_nodes_data.append({
                        "id": nid,
                        "name": attrs.get("name", ""),
                        "entity_type": attrs.get("entity_type", "unknown"),
                        "aliases": attrs.get("aliases", []),
                        "attributes": attrs.get("attributes", {}),
                        "first_seen": attrs.get("first_seen", ""),
                        "last_seen": attrs.get("last_seen", ""),
                        "memory_ids": attrs.get("memory_ids", []),
                    })

            dirty_edges_data: list[dict] = []
            for src, tgt in list(self._dirty_edges):
                if self._graph.has_edge(src, tgt):
                    attrs = dict(self._graph[src][tgt])
                    dirty_edges_data.append({
                        "source_id": src,
                        "target_id": tgt,
                        "relation_type": attrs.get("relation_type", ""),
                        "weight": attrs.get("weight", 1.0),
                        "timestamp": attrs.get("timestamp", ""),
                        "metadata": attrs.get("metadata", {}),
                    })

            try:
                full_data = nx.node_link_data(self._graph)
            except (TypeError, ValueError):
                full_data = None

            self._dirty_nodes.clear()
            self._dirty_edges.clear()

        # Perform I/O outside the lock
        if dirty_nodes_data:
            self._storage.save_graph_nodes_incremental(dirty_nodes_data)
        if dirty_edges_data:
            self._storage.save_graph_edges_incremental(dirty_edges_data)
        if full_data is not None:
            self._storage.save_graph_snapshot(full_data)
        _logger.debug(
            "Graph flush: %d nodes, %d edges",
            len(dirty_nodes_data),
            len(dirty_edges_data),
        )

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """返回节点数与边数统计"""
        with self._lock:
            return {
                "node_count": self._graph.number_of_nodes(),
                "edge_count": self._graph.number_of_edges(),
            }

    def batch(self):
        """Context manager for batch operations — defers persistence until exit.

        Usage:
            with graph.batch():
                for entity in entities:
                    graph.upsert_entity(entity)
                # flush() is called automatically on exit

        This avoids N redundant _save_snapshot_async() calls and replaces
        them with a single flush() at the end.
        """
        return _BatchContext(self)


class _BatchContext:
    """Helper for KnowledgeGraph.batch() context manager."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

    def __enter__(self) -> _BatchContext:
        self._graph._defer_persistence = True
        return self

    def __exit__(self, *exc_info) -> None:
        self._graph._defer_persistence = False
        self._graph.flush()
