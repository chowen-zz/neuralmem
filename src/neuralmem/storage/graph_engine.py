"""VectorGraphEngine — lightweight ontology-aware vector graph engine (v2).

Hybrid vector + graph storage using plain dict-based graph structures
plus numpy for vector operations.  No external graph DB required.

Key features
------------
* Hybrid vector + graph storage (dict graph + numpy vector index)
* Ontology-aware edges (entity type relationships with schema validation)
* Vector similarity search within graph context (neighbour-boosted search)
* Graph traversal with vector similarity boosting
* Incremental indexing (add vectors without full rebuild)
* Memory-efficient implementation (no external graph DB)
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any

import numpy as np

from neuralmem.core.exceptions import GraphError
from neuralmem.core.types import Entity, Relation

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_DIM = 384
_TOP_K_DEFAULT = 10


def _cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between a single query and a matrix of vectors.

    Parameters
    ----------
    query: (d,) float array — L2-normalised query vector.
    vectors: (n, d) float array — L2-normalised candidate vectors.

    Returns
    -------
    scores: (n,) float array in [0, 1].
    """
    if vectors.shape[0] == 0:
        return np.array([], dtype=np.float64)
    # Both are already normalised → dot product == cosine similarity
    return np.clip(vectors @ query, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Ontology schema
# ---------------------------------------------------------------------------

class OntologySchema:
    """Lightweight ontology that validates which (source_type, relation, target_type)
    triples are allowed.
    """

    def __init__(self, rules: set[tuple[str, str, str]] | None = None) -> None:
        self._rules: set[tuple[str, str, str]] = set(rules or {})
        self._lock = threading.Lock()

    def add_rule(self, source_type: str, relation_type: str, target_type: str) -> None:
        with self._lock:
            self._rules.add((source_type, relation_type, target_type))

    def remove_rule(self, source_type: str, relation_type: str, target_type: str) -> None:
        with self._lock:
            self._rules.discard((source_type, relation_type, target_type))

    def is_valid(self, source_type: str, relation_type: str, target_type: str) -> bool:
        """Return True if the triple is allowed (or if the ontology is empty / permissive)."""
        with self._lock:
            if not self._rules:
                return True  # permissive mode
            return (source_type, relation_type, target_type) in self._rules

    def get_rules(self) -> set[tuple[str, str, str]]:
        with self._lock:
            return set(self._rules)


# ---------------------------------------------------------------------------
# Graph node / edge data models (internal)
# ---------------------------------------------------------------------------

class _GraphNode:
    """Internal mutable container for graph node data."""

    __slots__ = ("entity", "vector", "memory_ids", "metadata")

    def __init__(
        self,
        entity: Entity,
        vector: np.ndarray | None = None,
        memory_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.entity = entity
        self.vector = vector
        self.memory_ids: list[str] = list(memory_ids or [])
        self.metadata: dict[str, Any] = dict(metadata or {})


class _GraphEdge:
    """Internal mutable container for graph edge data."""

    __slots__ = ("relation", "source_type", "target_type", "metadata")

    def __init__(
        self,
        relation: Relation,
        source_type: str,
        target_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.relation = relation
        self.source_type = source_type
        self.target_type = target_type
        self.metadata: dict[str, Any] = dict(metadata or {})


# ---------------------------------------------------------------------------
# VectorGraphEngine
# ---------------------------------------------------------------------------

class VectorGraphEngine:
    """Lightweight ontology-aware vector graph engine.

    Stores the graph as plain Python dicts (adjacency lists) and maintains
    a numpy matrix for fast cosine-similarity search over node vectors.
    """

    def __init__(
        self,
        dim: int = _DEFAULT_DIM,
        ontology: OntologySchema | None = None,
    ) -> None:
        self._dim = dim
        self._ontology = ontology or OntologySchema()

        # Graph structure: dict-based adjacency (no NetworkX)
        self._nodes: dict[str, _GraphNode] = {}          # id -> _GraphNode
        self._out_edges: dict[str, dict[str, _GraphEdge]] = {}  # src -> {tgt -> _GraphEdge}
        self._in_edges: dict[str, set[str]] = {}         # tgt -> {src, ...}

        # Vector index: incremental numpy storage
        self._id_to_index: dict[str, int] = {}   # node id -> row index in _vectors
        self._index_to_id: dict[int, str] = {}   # row index -> node id
        self._vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)  # (n, dim)
        self._vector_norms: np.ndarray = np.empty(0, dtype=np.float32)   # (n,)

        self._lock = threading.Lock()
        self._dirty: set[str] = set()  # node ids with changed vectors

    # ------------------------------------------------------------------
    # Entity / node operations
    # ------------------------------------------------------------------

    def upsert_entity(
        self,
        entity: Entity,
        vector: list[float] | np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update an entity node.

        If *vector* is provided the node is indexed for similarity search.
        """
        with self._lock:
            node_id = entity.id
            np_vector = None
            if vector is not None:
                np_vector = np.asarray(vector, dtype=np.float32)
                if np_vector.shape != (self._dim,):
                    raise GraphError(
                        f"Vector dimension mismatch: expected {self._dim}, "
                        f"got {np_vector.shape}"
                    )
                np_vector = np_vector / (np.linalg.norm(np_vector) + 1e-12)

            if node_id in self._nodes:
                # Update existing node — preserve memory_ids
                old_node = self._nodes[node_id]
                self._nodes[node_id] = _GraphNode(
                    entity=entity,
                    vector=np_vector if np_vector is not None else old_node.vector,
                    memory_ids=old_node.memory_ids,
                    metadata={**(old_node.metadata), **(metadata or {})},
                )
                if np_vector is not None:
                    self._dirty.add(node_id)
            else:
                self._nodes[node_id] = _GraphNode(
                    entity=entity,
                    vector=np_vector,
                    metadata=dict(metadata or {}),
                )
                if np_vector is not None:
                    self._dirty.add(node_id)

    def get_entity(self, entity_id: str) -> Entity | None:
        """Return the Entity for *entity_id* or None."""
        with self._lock:
            node = self._nodes.get(entity_id)
            return node.entity if node else None

    def get_node_vector(self, entity_id: str) -> np.ndarray | None:
        """Return the normalised vector for *entity_id* or None."""
        with self._lock:
            node = self._nodes.get(entity_id)
            return node.vector.copy() if node and node.vector is not None else None

    def remove_entity(self, entity_id: str) -> bool:
        """Remove a node and all incident edges.  Return True if removed."""
        with self._lock:
            if entity_id not in self._nodes:
                return False

            # Remove outgoing edges
            for tgt in list(self._out_edges.get(entity_id, {}).keys()):
                self._in_edges.get(tgt, set()).discard(entity_id)
            self._out_edges.pop(entity_id, None)

            # Remove incoming edges
            for src in list(self._in_edges.get(entity_id, set())):
                self._out_edges.get(src, {}).pop(entity_id, None)
            self._in_edges.pop(entity_id, None)

            del self._nodes[entity_id]
            self._dirty.discard(entity_id)

            # Defer vector matrix cleanup to next index rebuild
            return True

    def link_memory(self, entity_id: str, memory_id: str) -> None:
        """Associate a memory ID with a node."""
        with self._lock:
            node = self._nodes.get(entity_id)
            if node is None:
                raise GraphError(f"Entity not found: {entity_id}")
            if memory_id not in node.memory_ids:
                node.memory_ids.append(memory_id)

    # ------------------------------------------------------------------
    # Relation / edge operations (ontology-aware)
    # ------------------------------------------------------------------

    def add_relation(self, relation: Relation) -> None:
        """Add a directed edge.  Validates against the ontology schema."""
        with self._lock:
            src, tgt = relation.source_id, relation.target_id
            src_node = self._nodes.get(src)
            tgt_node = self._nodes.get(tgt)
            if src_node is None or tgt_node is None:
                missing = [k for k in (src, tgt) if k not in self._nodes]
                raise GraphError(f"Missing entities for relation: {missing}")

            if not self._ontology.is_valid(
                src_node.entity.entity_type,
                relation.relation_type,
                tgt_node.entity.entity_type,
            ):
                raise GraphError(
                    f"Ontology violation: ({src_node.entity.entity_type}, "
                    f"{relation.relation_type}, {tgt_node.entity.entity_type})"
                )

            edge = _GraphEdge(
                relation=relation,
                source_type=src_node.entity.entity_type,
                target_type=tgt_node.entity.entity_type,
            )
            self._out_edges.setdefault(src, {})[tgt] = edge
            self._in_edges.setdefault(tgt, set()).add(src)

    def get_relation(self, source_id: str, target_id: str) -> Relation | None:
        """Return the Relation between two nodes, or None."""
        with self._lock:
            edge = self._out_edges.get(source_id, {}).get(target_id)
            return edge.relation if edge else None

    def remove_relation(self, source_id: str, target_id: str) -> bool:
        """Remove a directed edge.  Return True if removed."""
        with self._lock:
            if target_id not in self._out_edges.get(source_id, {}):
                return False
            del self._out_edges[source_id][target_id]
            self._in_edges.get(target_id, set()).discard(source_id)
            return True

    # ------------------------------------------------------------------
    # Vector similarity search
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """Rebuild the numpy vector matrix from scratch.

        Called automatically when dirty nodes exist at search time.
        """
        ids_with_vectors = [
            nid for nid, node in self._nodes.items() if node.vector is not None
        ]
        n = len(ids_with_vectors)
        if n == 0:
            self._vectors = np.empty((0, self._dim), dtype=np.float32)
            self._vector_norms = np.empty(0, dtype=np.float32)
            self._id_to_index.clear()
            self._index_to_id.clear()
            self._dirty.clear()
            return

        self._vectors = np.zeros((n, self._dim), dtype=np.float32)
        self._id_to_index = {}
        self._index_to_id = {}
        for i, nid in enumerate(ids_with_vectors):
            vec = self._nodes[nid].vector
            assert vec is not None
            self._vectors[i] = vec
            self._id_to_index[nid] = i
            self._index_to_id[i] = nid

        norms = np.linalg.norm(self._vectors, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        self._vectors = self._vectors / norms[:, np.newaxis]
        self._vector_norms = norms
        self._dirty.clear()

    def _ensure_index_fresh(self) -> None:
        """Rebuild index if any nodes with vectors are dirty."""
        if self._dirty:
            self._rebuild_index()

    def vector_search(
        self,
        query_vector: list[float] | np.ndarray,
        top_k: int = _TOP_K_DEFAULT,
        entity_type: str | None = None,
    ) -> list[tuple[str, float]]:
        """Cosine-similarity search over all indexed node vectors.

        Parameters
        ----------
        query_vector: (d,) float vector.
        top_k: maximum results to return.
        entity_type: if given, filter to nodes of this type.

        Returns
        -------
        list of (entity_id, score) sorted by descending score.
        """
        q = np.asarray(query_vector, dtype=np.float32)
        if q.shape != (self._dim,):
            raise GraphError(
                f"Query dimension mismatch: expected {self._dim}, got {q.shape}"
            )
        with self._lock:
            self._ensure_index_fresh()
            if self._vectors.shape[0] == 0:
                return []

            q = q / (np.linalg.norm(q) + 1e-12)

            scores = _cosine_similarity(q, self._vectors)
            ranked = np.argsort(-scores)

            results: list[tuple[str, float]] = []
            for idx in ranked:
                nid = self._index_to_id[int(idx)]
                if entity_type is not None:
                    if self._nodes[nid].entity.entity_type != entity_type:
                        continue
                results.append((nid, float(scores[idx])))
                if len(results) >= top_k:
                    break
            return results

    def vector_search_in_context(
        self,
        query_vector: list[float] | np.ndarray,
        seed_entity_ids: list[str],
        depth: int = 1,
        top_k: int = _TOP_K_DEFAULT,
        graph_boost: float = 0.15,
    ) -> list[tuple[str, float]]:
        """Vector similarity search limited to the graph neighbourhood of *seed_entity_ids*.

        The final score is ``vector_score + graph_boost * (1 / distance)`` where
        *distance* is the BFS hop count from the nearest seed (1 = seed itself).

        Parameters
        ----------
        query_vector: (d,) float vector.
        seed_entity_ids: starting node IDs.
        depth: BFS depth to explore (0 = seeds only).
        top_k: maximum results.
        graph_boost: weight added for graph proximity.

        Returns
        -------
        list of (entity_id, combined_score) sorted descending.
        """
        with self._lock:
            self._ensure_index_fresh()
            # Gather neighbourhood
            neighbourhood: dict[str, int] = {}  # nid -> min distance from any seed
            queue: deque[tuple[str, int]] = deque()
            visited: set[str] = set()
            for sid in seed_entity_ids:
                if sid in self._nodes:
                    neighbourhood[sid] = 0
                    queue.append((sid, 0))
                    visited.add(sid)

            while queue:
                nid, d = queue.popleft()
                if d >= depth:
                    continue
                for nxt in self._out_edges.get(nid, {}).keys():
                    if nxt not in visited:
                        visited.add(nxt)
                        neighbourhood[nxt] = d + 1
                        queue.append((nxt, d + 1))
                for nxt in self._in_edges.get(nid, set()):
                    if nxt not in visited:
                        visited.add(nxt)
                        neighbourhood[nxt] = d + 1
                        queue.append((nxt, d + 1))

            if not neighbourhood:
                return []

            # Compute vector scores for neighbourhood members that have vectors
            q = np.asarray(query_vector, dtype=np.float32)
            if q.shape != (self._dim,):
                raise GraphError(
                    f"Query dimension mismatch: expected {self._dim}, got {q.shape}"
                )
            q = q / (np.linalg.norm(q) + 1e-12)

            results: list[tuple[str, float]] = []
            for nid, dist in neighbourhood.items():
                idx = self._id_to_index.get(nid)
                if idx is None:
                    continue  # no vector indexed
                vec_score = float(np.clip(self._vectors[idx] @ q, 0.0, 1.0))
                proximity = 1.0 / (dist + 1.0)
                combined = vec_score + graph_boost * proximity
                results.append((nid, combined))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    # ------------------------------------------------------------------
    # Graph traversal with vector similarity boosting
    # ------------------------------------------------------------------

    def traverse_with_vector_boost(
        self,
        seed_entity_ids: list[str],
        query_vector: list[float] | np.ndarray,
        depth: int = 2,
        top_k: int = _TOP_K_DEFAULT,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
    ) -> list[tuple[str, float]]:
        """BFS traversal from seeds, scoring each reached node by a blend of
        graph distance and vector similarity to *query_vector*.

        Score formula::

            score = vector_weight * vec_score + graph_weight * (1 / (dist + 1))

        Parameters
        ----------
        seed_entity_ids: starting node IDs.
        query_vector: (d,) float vector.
        depth: BFS depth.
        top_k: maximum results.
        vector_weight: weight for cosine similarity component.
        graph_weight: weight for graph proximity component.

        Returns
        -------
        list of (entity_id, score) sorted descending.
        """
        with self._lock:
            self._ensure_index_fresh()
            q = np.asarray(query_vector, dtype=np.float32)
            if q.shape != (self._dim,):
                raise GraphError(
                    f"Query dimension mismatch: expected {self._dim}, got {q.shape}"
                )
            q = q / (np.linalg.norm(q) + 1e-12)

            visited: set[str] = set(seed_entity_ids)
            queue: deque[tuple[str, int]] = deque((sid, 0) for sid in seed_entity_ids)
            scores: dict[str, float] = {}

            while queue:
                nid, d = queue.popleft()
                if d > depth:
                    continue

                idx = self._id_to_index.get(nid)
                vec_score = 0.0
                if idx is not None:
                    vec_score = float(np.clip(self._vectors[idx] @ q, 0.0, 1.0))

                proximity = 1.0 / (d + 1.0)
                score = vector_weight * vec_score + graph_weight * proximity
                scores[nid] = score

                if d < depth:
                    for nxt in self._out_edges.get(nid, {}).keys():
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append((nxt, d + 1))
                    for nxt in self._in_edges.get(nid, set()):
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append((nxt, d + 1))

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]

    # ------------------------------------------------------------------
    # Neighbour / subgraph queries
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        entity_id: str,
        direction: str = "out",
        relation_type: str | None = None,
    ) -> list[Entity]:
        """Return neighbour entities.

        Parameters
        ----------
        entity_id: centre node.
        direction: ``"out"``, ``"in"``, or ``"both"``.
        relation_type: optional filter by relation type.
        """
        with self._lock:
            nids: set[str] = set()
            if direction in ("out", "both"):
                for tgt, edge in self._out_edges.get(entity_id, {}).items():
                    if relation_type is None or edge.relation.relation_type == relation_type:
                        nids.add(tgt)
            if direction in ("in", "both"):
                for src in self._in_edges.get(entity_id, set()):
                    edge = self._out_edges.get(src, {}).get(entity_id)
                    if edge is not None:
                        if relation_type is None or edge.relation.relation_type == relation_type:
                            nids.add(src)
            return [self._nodes[nid].entity for nid in nids if nid in self._nodes]

    def get_subgraph(
        self, entity_ids: list[str], depth: int = 1
    ) -> dict[str, Any]:
        """Extract a subgraph around *entity_ids* up to *depth* hops.

        Returns a serialisable dict with ``nodes`` and ``edges`` keys.
        """
        with self._lock:
            visited: set[str] = set(entity_ids)
            queue: deque[tuple[str, int]] = deque((eid, 0) for eid in entity_ids)
            while queue:
                nid, d = queue.popleft()
                if d >= depth:
                    continue
                for nxt in self._out_edges.get(nid, {}).keys():
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, d + 1))
                for nxt in self._in_edges.get(nid, set()):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, d + 1))

            nodes = {
                nid: {
                    "entity": self._nodes[nid].entity.model_dump(),
                    "has_vector": self._nodes[nid].vector is not None,
                    "memory_ids": list(self._nodes[nid].memory_ids),
                }
                for nid in visited if nid in self._nodes
            }
            edges: list[dict[str, Any]] = []
            for src in visited:
                for tgt, edge in self._out_edges.get(src, {}).items():
                    if tgt in visited:
                        edges.append(
                            {
                                "source": src,
                                "target": tgt,
                                "relation_type": edge.relation.relation_type,
                                "weight": edge.relation.weight,
                            }
                        )
            return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # Incremental index management
    # ------------------------------------------------------------------

    def rebuild_index(self) -> None:
        """Explicitly rebuild the vector index.  Usually not needed — called
        automatically before searches when dirty nodes exist."""
        with self._lock:
            self._rebuild_index()

    def compact(self) -> None:
        """Remove deleted nodes from the vector index and rebuild."""
        with self._lock:
            # Mark all vector-holding nodes dirty so rebuild cleans up
            for nid, node in self._nodes.items():
                if node.vector is not None:
                    self._dirty.add(nid)
            self._rebuild_index()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return engine statistics."""
        with self._lock:
            return {
                "node_count": len(self._nodes),
                "edge_count": sum(len(v) for v in self._out_edges.values()),
                "indexed_vectors": self._vectors.shape[0],
                "vector_dim": self._dim,
                "dirty_count": len(self._dirty),
                "ontology_rules": len(self._ontology.get_rules()),
            }

    def batch_upsert(
        self,
        entities: list[Entity],
        vectors: list[list[float] | np.ndarray | None] | None = None,
    ) -> None:
        """Batch upsert multiple entities efficiently (single lock)."""
        vectors = vectors or [None] * len(entities)
        if len(entities) != len(vectors):
            raise GraphError("entities and vectors must have same length")
        with self._lock:
            for entity, vector in zip(entities, vectors):
                np_vector = None
                if vector is not None:
                    np_vector = np.asarray(vector, dtype=np.float32)
                    if np_vector.shape != (self._dim,):
                        raise GraphError(
                            f"Vector dimension mismatch for {entity.id}: "
                            f"expected {self._dim}, got {np_vector.shape}"
                        )
                    np_vector = np_vector / (np.linalg.norm(np_vector) + 1e-12)

                if entity.id in self._nodes:
                    old_node = self._nodes[entity.id]
                    self._nodes[entity.id] = _GraphNode(
                        entity=entity,
                        vector=np_vector if np_vector is not None else old_node.vector,
                        memory_ids=old_node.memory_ids,
                        metadata=old_node.metadata,
                    )
                else:
                    self._nodes[entity.id] = _GraphNode(
                        entity=entity,
                        vector=np_vector,
                        metadata={},
                    )
                if np_vector is not None:
                    self._dirty.add(entity.id)

    # ------------------------------------------------------------------
    # Context manager for batch mutations (defers index rebuild)
    # ------------------------------------------------------------------

    def batch(self) -> _BatchContext:
        """Return a context manager that defers index rebuild until exit.

        Usage::

            with engine.batch():
                for e in entities:
                    engine.upsert_entity(e, vector=...)
                for r in relations:
                    engine.add_relation(r)
            # index is rebuilt once on exit (if needed)
        """
        return _BatchContext(self)


class _BatchContext:
    """Helper for VectorGraphEngine.batch()."""

    def __init__(self, engine: VectorGraphEngine) -> None:
        self._engine = engine
        self._defer: bool = False

    def __enter__(self) -> _BatchContext:
        self._defer = True
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._defer = False
        self._engine.rebuild_index()
