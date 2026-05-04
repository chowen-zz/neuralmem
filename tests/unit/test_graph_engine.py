"""Unit tests for VectorGraphEngine — mock-based, no external DB."""
from __future__ import annotations

import numpy as np
import pytest

from neuralmem.core.exceptions import GraphError
from neuralmem.core.types import Entity, Relation
from neuralmem.storage.graph_engine import (
    OntologySchema,
    VectorGraphEngine,
    _cosine_similarity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_vec(dim: int = 384, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.random(dim).astype(np.float32).tolist()


def _make_entity(name: str, entity_type: str = "person") -> Entity:
    return Entity(name=name, entity_type=entity_type)


# ---------------------------------------------------------------------------
# OntologySchema
# ---------------------------------------------------------------------------

class TestOntologySchema:
    def test_empty_is_permissive(self) -> None:
        o = OntologySchema()
        assert o.is_valid("person", "knows", "person")
        assert o.is_valid("any", "thing", "goes")

    def test_add_and_check(self) -> None:
        o = OntologySchema()
        o.add_rule("person", "works_at", "company")
        assert o.is_valid("person", "works_at", "company")
        assert not o.is_valid("company", "works_at", "person")

    def test_remove_rule(self) -> None:
        o = OntologySchema()
        o.add_rule("a", "b", "c")
        o.add_rule("x", "y", "z")
        assert o.is_valid("a", "b", "c")
        o.remove_rule("a", "b", "c")
        assert not o.is_valid("a", "b", "c")
        assert o.is_valid("x", "y", "z")

    def test_get_rules(self) -> None:
        o = OntologySchema({("x", "y", "z")})
        assert o.get_rules() == {("x", "y", "z")}


# ---------------------------------------------------------------------------
# Engine basics
# ---------------------------------------------------------------------------

class TestVectorGraphEngineBasics:
    def test_init_defaults(self) -> None:
        engine = VectorGraphEngine()
        stats = engine.get_stats()
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["vector_dim"] == 384
        assert stats["indexed_vectors"] == 0

    def test_upsert_and_get_entity(self) -> None:
        engine = VectorGraphEngine()
        e = _make_entity("Alice")
        engine.upsert_entity(e)
        fetched = engine.get_entity(e.id)
        assert fetched is not None
        assert fetched.name == "Alice"
        assert fetched.entity_type == "person"

    def test_upsert_with_vector(self) -> None:
        engine = VectorGraphEngine(dim=8)
        vec = _mock_vec(dim=8, seed=1)
        e = _make_entity("Bob")
        engine.upsert_entity(e, vector=vec)
        v = engine.get_node_vector(e.id)
        assert v is not None
        assert v.shape == (8,)
        # Normalised
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-5)

    def test_upsert_wrong_dim_raises(self) -> None:
        engine = VectorGraphEngine(dim=8)
        with pytest.raises(GraphError):
            engine.upsert_entity(_make_entity("X"), vector=_mock_vec(dim=4))

    def test_remove_entity(self) -> None:
        engine = VectorGraphEngine()
        e = _make_entity("Charlie")
        engine.upsert_entity(e)
        assert engine.remove_entity(e.id) is True
        assert engine.get_entity(e.id) is None
        assert engine.remove_entity(e.id) is False

    def test_link_memory(self) -> None:
        engine = VectorGraphEngine()
        e = _make_entity("Dana")
        engine.upsert_entity(e)
        engine.link_memory(e.id, "mem-001")
        engine.link_memory(e.id, "mem-002")
        # memory_ids are internal; verify via subgraph dump
        sg = engine.get_subgraph([e.id])
        assert set(sg["nodes"][e.id]["memory_ids"]) == {"mem-001", "mem-002"}

    def test_link_memory_missing_entity(self) -> None:
        engine = VectorGraphEngine()
        with pytest.raises(GraphError):
            engine.link_memory("no-such-id", "mem-001")


# ---------------------------------------------------------------------------
# Relations / edges
# ---------------------------------------------------------------------------

class TestRelations:
    def test_add_relation(self) -> None:
        engine = VectorGraphEngine()
        a = _make_entity("Alice", "person")
        b = _make_entity("Bob", "person")
        engine.upsert_entity(a)
        engine.upsert_entity(b)
        r = Relation(source_id=a.id, target_id=b.id, relation_type="knows")
        engine.add_relation(r)
        fetched = engine.get_relation(a.id, b.id)
        assert fetched is not None
        assert fetched.relation_type == "knows"

    def test_add_relation_missing_node(self) -> None:
        engine = VectorGraphEngine()
        a = _make_entity("Alice")
        engine.upsert_entity(a)
        r = Relation(source_id=a.id, target_id="ghost", relation_type="knows")
        with pytest.raises(GraphError):
            engine.add_relation(r)

    def test_ontology_violation(self) -> None:
        o = OntologySchema()
        o.add_rule("person", "works_at", "company")
        engine = VectorGraphEngine(ontology=o)
        a = _make_entity("Alice", "person")
        b = _make_entity("Bob", "person")
        engine.upsert_entity(a)
        engine.upsert_entity(b)
        r = Relation(source_id=a.id, target_id=b.id, relation_type="works_at")
        with pytest.raises(GraphError):
            engine.add_relation(r)

    def test_remove_relation(self) -> None:
        engine = VectorGraphEngine()
        a = _make_entity("A")
        b = _make_entity("B")
        engine.upsert_entity(a)
        engine.upsert_entity(b)
        engine.add_relation(Relation(source_id=a.id, target_id=b.id, relation_type="r"))
        assert engine.remove_relation(a.id, b.id) is True
        assert engine.get_relation(a.id, b.id) is None
        assert engine.remove_relation(a.id, b.id) is False

    def test_get_neighbors(self) -> None:
        engine = VectorGraphEngine()
        a = _make_entity("A")
        b = _make_entity("B")
        c = _make_entity("C")
        engine.upsert_entity(a)
        engine.upsert_entity(b)
        engine.upsert_entity(c)
        engine.add_relation(Relation(source_id=a.id, target_id=b.id, relation_type="r1"))
        engine.add_relation(Relation(source_id=a.id, target_id=c.id, relation_type="r2"))
        # outgoing
        out = engine.get_neighbors(a.id, direction="out")
        assert len(out) == 2
        # incoming
        inn = engine.get_neighbors(b.id, direction="in")
        assert len(inn) == 1 and inn[0].id == a.id
        # both
        both = engine.get_neighbors(a.id, direction="both")
        assert len(both) == 2
        # filter by relation type
        filt = engine.get_neighbors(a.id, relation_type="r1")
        assert len(filt) == 1 and filt[0].id == b.id


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

class TestVectorSearch:
    def test_empty_search(self) -> None:
        engine = VectorGraphEngine(dim=4)
        q = [1.0, 0.0, 0.0, 0.0]
        assert engine.vector_search(q) == []

    def test_basic_similarity(self) -> None:
        engine = VectorGraphEngine(dim=3)
        # Three orthogonal-ish vectors
        e1 = _make_entity("A")
        e2 = _make_entity("B")
        e3 = _make_entity("C")
        engine.upsert_entity(e1, vector=[1.0, 0.0, 0.0])
        engine.upsert_entity(e2, vector=[0.0, 1.0, 0.0])
        engine.upsert_entity(e3, vector=[0.0, 0.0, 1.0])

        results = engine.vector_search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][0] == e1.id
        np.testing.assert_allclose(results[0][1], 1.0, atol=1e-5)

    def test_entity_type_filter(self) -> None:
        engine = VectorGraphEngine(dim=3)
        e1 = _make_entity("Alice", "person")
        e2 = _make_entity("Acme", "company")
        engine.upsert_entity(e1, vector=[1.0, 0.0, 0.0])
        engine.upsert_entity(e2, vector=[1.0, 0.0, 0.0])
        results = engine.vector_search([1.0, 0.0, 0.0], entity_type="company")
        assert len(results) == 1
        assert results[0][0] == e2.id

    def test_dimension_mismatch_query(self) -> None:
        engine = VectorGraphEngine(dim=3)
        with pytest.raises(GraphError):
            engine.vector_search([1.0, 0.0])

    def test_incremental_indexing(self) -> None:
        engine = VectorGraphEngine(dim=3)
        e1 = _make_entity("A")
        engine.upsert_entity(e1, vector=[1.0, 0.0, 0.0])
        # Search triggers lazy rebuild
        r1 = engine.vector_search([1.0, 0.0, 0.0])
        assert len(r1) == 1
        # Add another without explicit rebuild
        e2 = _make_entity("B")
        engine.upsert_entity(e2, vector=[0.0, 1.0, 0.0])
        r2 = engine.vector_search([0.0, 1.0, 0.0])
        assert len(r2) == 2
        assert r2[0][0] == e2.id

    def test_rebuild_index_explicit(self) -> None:
        engine = VectorGraphEngine(dim=3)
        e = _make_entity("A")
        engine.upsert_entity(e, vector=[1.0, 0.0, 0.0])
        engine.rebuild_index()
        assert engine.get_stats()["dirty_count"] == 0


# ---------------------------------------------------------------------------
# Search in graph context
# ---------------------------------------------------------------------------

class TestVectorSearchInContext:
    def test_context_search_basic(self) -> None:
        engine = VectorGraphEngine(dim=3)
        a = _make_entity("A")
        b = _make_entity("B")
        c = _make_entity("C")
        engine.upsert_entity(a, vector=[1.0, 0.0, 0.0])
        engine.upsert_entity(b, vector=[0.9, 0.1, 0.0])
        engine.upsert_entity(c, vector=[0.0, 1.0, 0.0])
        engine.add_relation(Relation(source_id=a.id, target_id=b.id, relation_type="r"))
        engine.add_relation(Relation(source_id=b.id, target_id=c.id, relation_type="r"))

        # Search around A with depth=1 should include A and B
        results = engine.vector_search_in_context(
            query_vector=[1.0, 0.0, 0.0],
            seed_entity_ids=[a.id],
            depth=1,
            top_k=10,
        )
        ids = [r[0] for r in results]
        assert a.id in ids
        assert b.id in ids
        # C is at depth 2, should not appear
        assert c.id not in ids

    def test_context_search_depth_zero(self) -> None:
        engine = VectorGraphEngine(dim=3)
        a = _make_entity("A")
        b = _make_entity("B")
        engine.upsert_entity(a, vector=[1.0, 0.0, 0.0])
        engine.upsert_entity(b, vector=[0.9, 0.1, 0.0])
        engine.add_relation(Relation(source_id=a.id, target_id=b.id, relation_type="r"))

        results = engine.vector_search_in_context(
            query_vector=[1.0, 0.0, 0.0],
            seed_entity_ids=[a.id],
            depth=0,
        )
        assert len(results) == 1
        assert results[0][0] == a.id

    def test_context_search_boost_ordering(self) -> None:
        engine = VectorGraphEngine(dim=3)
        a = _make_entity("A")
        b = _make_entity("B")
        c = _make_entity("C")
        # A and C have identical vectors, B is closer in graph
        engine.upsert_entity(a, vector=[1.0, 0.0, 0.0])
        engine.upsert_entity(b, vector=[0.5, 0.5, 0.0])
        engine.upsert_entity(c, vector=[1.0, 0.0, 0.0])
        engine.add_relation(Relation(source_id=a.id, target_id=b.id, relation_type="r"))

        results = engine.vector_search_in_context(
            query_vector=[1.0, 0.0, 0.0],
            seed_entity_ids=[a.id],
            depth=1,
            graph_boost=0.2,
        )
        # A should be first (distance 0), then B (graph proximity), then C (distance 2 no edge)
        # Actually C is not in neighbourhood because no edge to C
        ids = [r[0] for r in results]
        assert ids[0] == a.id
        assert b.id in ids


# ---------------------------------------------------------------------------
# Traversal with vector boost
# ---------------------------------------------------------------------------

class TestTraverseWithVectorBoost:
    def test_traversal_scores(self) -> None:
        engine = VectorGraphEngine(dim=3)
        a = _make_entity("A")
        b = _make_entity("B")
        c = _make_entity("C")
        engine.upsert_entity(a, vector=[1.0, 0.0, 0.0])
        engine.upsert_entity(b, vector=[0.9, 0.1, 0.0])
        engine.upsert_entity(c, vector=[0.0, 1.0, 0.0])
        engine.add_relation(Relation(source_id=a.id, target_id=b.id, relation_type="r"))
        engine.add_relation(Relation(source_id=b.id, target_id=c.id, relation_type="r"))

        results = engine.traverse_with_vector_boost(
            seed_entity_ids=[a.id],
            query_vector=[1.0, 0.0, 0.0],
            depth=2,
            top_k=10,
        )
        ids = [r[0] for r in results]
        assert a.id in ids
        assert b.id in ids
        assert c.id in ids
        # A should have highest score (distance 0 + perfect vec match)
        assert ids[0] == a.id

    def test_top_k_limits(self) -> None:
        engine = VectorGraphEngine(dim=3)
        a = _make_entity("A")
        b = _make_entity("B")
        engine.upsert_entity(a, vector=[1.0, 0.0, 0.0])
        engine.upsert_entity(b, vector=[0.0, 1.0, 0.0])
        engine.add_relation(Relation(source_id=a.id, target_id=b.id, relation_type="r"))

        results = engine.traverse_with_vector_boost(
            seed_entity_ids=[a.id],
            query_vector=[1.0, 0.0, 0.0],
            depth=1,
            top_k=1,
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------

class TestSubgraph:
    def test_get_subgraph(self) -> None:
        engine = VectorGraphEngine()
        a = _make_entity("A")
        b = _make_entity("B")
        c = _make_entity("C")
        engine.upsert_entity(a)
        engine.upsert_entity(b)
        engine.upsert_entity(c)
        engine.add_relation(Relation(source_id=a.id, target_id=b.id, relation_type="r"))
        engine.add_relation(Relation(source_id=b.id, target_id=c.id, relation_type="r"))

        sg = engine.get_subgraph([a.id], depth=2)
        assert len(sg["nodes"]) == 3
        assert len(sg["edges"]) == 2
        edge_pairs = {(e["source"], e["target"]) for e in sg["edges"]}
        assert (a.id, b.id) in edge_pairs
        assert (b.id, c.id) in edge_pairs

    def test_get_subgraph_depth_one(self) -> None:
        engine = VectorGraphEngine()
        a = _make_entity("A")
        b = _make_entity("B")
        c = _make_entity("C")
        engine.upsert_entity(a)
        engine.upsert_entity(b)
        engine.upsert_entity(c)
        engine.add_relation(Relation(source_id=a.id, target_id=b.id, relation_type="r"))
        engine.add_relation(Relation(source_id=b.id, target_id=c.id, relation_type="r"))

        sg = engine.get_subgraph([a.id], depth=1)
        assert a.id in sg["nodes"]
        assert b.id in sg["nodes"]
        assert c.id not in sg["nodes"]


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------

class TestBatchOperations:
    def test_batch_upsert(self) -> None:
        engine = VectorGraphEngine(dim=3)
        entities = [_make_entity(str(i)) for i in range(5)]
        vectors = [[1.0, 0.0, 0.0] for _ in range(5)]
        engine.batch_upsert(entities, vectors)
        assert engine.get_stats()["node_count"] == 5
        for e in entities:
            assert engine.get_entity(e.id) is not None

    def test_batch_upsert_length_mismatch(self) -> None:
        engine = VectorGraphEngine(dim=3)
        with pytest.raises(GraphError):
            engine.batch_upsert([_make_entity("A")], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def test_batch_context_manager(self) -> None:
        engine = VectorGraphEngine(dim=3)
        a = _make_entity("A")
        b = _make_entity("B")
        with engine.batch():
            engine.upsert_entity(a, vector=[1.0, 0.0, 0.0])
            engine.upsert_entity(b, vector=[0.0, 1.0, 0.0])
            # Inside batch, index is not yet rebuilt
            assert engine.get_stats()["dirty_count"] == 2
        # After exit, index rebuilt
        assert engine.get_stats()["dirty_count"] == 0
        results = engine.vector_search([1.0, 0.0, 0.0])
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Stats & compact
# ---------------------------------------------------------------------------

class TestStatsAndCompact:
    def test_stats_after_mutations(self) -> None:
        engine = VectorGraphEngine(dim=3)
        a = _make_entity("A")
        b = _make_entity("B")
        engine.upsert_entity(a, vector=[1.0, 0.0, 0.0])
        engine.upsert_entity(b, vector=[0.0, 1.0, 0.0])
        engine.add_relation(Relation(source_id=a.id, target_id=b.id, relation_type="r"))
        # trigger index rebuild so stats reflect indexed state
        engine.rebuild_index()

        stats = engine.get_stats()
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1
        assert stats["indexed_vectors"] == 2
        assert stats["dirty_count"] == 0

    def test_compact_removes_deleted(self) -> None:
        engine = VectorGraphEngine(dim=3)
        a = _make_entity("A")
        b = _make_entity("B")
        engine.upsert_entity(a, vector=[1.0, 0.0, 0.0])
        engine.upsert_entity(b, vector=[0.0, 1.0, 0.0])
        engine.remove_entity(a.id)
        engine.compact()
        stats = engine.get_stats()
        assert stats["node_count"] == 1
        assert stats["indexed_vectors"] == 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestInternalHelpers:
    def test_cosine_similarity_empty(self) -> None:
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        empty = np.empty((0, 3), dtype=np.float32)
        scores = _cosine_similarity(q, empty)
        assert scores.shape == (0,)

    def test_cosine_similarity_orthogonal(self) -> None:
        q = np.array([1.0, 0.0], dtype=np.float32)
        vecs = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        scores = _cosine_similarity(q, vecs)
        np.testing.assert_allclose(scores, [0.0, 1.0], atol=1e-5)

    def test_cosine_similarity_parallel(self) -> None:
        q = np.array([1.0, 0.0], dtype=np.float32)
        # Pre-normalised vectors: [1,0] and [1,0] (same direction)
        vecs = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        scores = _cosine_similarity(q, vecs)
        np.testing.assert_allclose(scores, [1.0, 1.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Thread safety smoke test
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_upserts(self) -> None:
        import threading

        engine = VectorGraphEngine(dim=8)
        errors: list[Exception] = []
        created_ids: set[str] = set()
        id_lock = threading.Lock()

        def worker(seed: int) -> None:
            try:
                for i in range(20):
                    e = _make_entity(f"t{seed}_{i}")
                    with id_lock:
                        created_ids.add(e.id)
                    engine.upsert_entity(e, vector=_mock_vec(dim=8, seed=seed + i))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        stats = engine.get_stats()
        # Verify by counting unique IDs actually created (Entity.id is random ULID)
        assert stats["node_count"] == len(created_ids)
        # Rebuild then search to ensure no corruption
        engine.rebuild_index()
        results = engine.vector_search(_mock_vec(dim=8, seed=0), top_k=5)
        assert len(results) == 5

    def test_concurrent_reads_and_writes(self) -> None:
        import threading
        import time

        engine = VectorGraphEngine(dim=8)
        # Seed
        for i in range(10):
            engine.upsert_entity(_make_entity(f"seed{i}"), vector=_mock_vec(dim=8, seed=i))

        stop = threading.Event()
        read_errors: list[Exception] = []
        write_errors: list[Exception] = []

        def reader() -> None:
            while not stop.is_set():
                try:
                    engine.vector_search(_mock_vec(dim=8), top_k=3)
                    engine.get_neighbors(list(engine._nodes.keys())[0], direction="both")
                except Exception as exc:
                    read_errors.append(exc)

        def writer() -> None:
            for i in range(50):
                try:
                    e = _make_entity(f"w{i}")
                    engine.upsert_entity(e, vector=_mock_vec(dim=8, seed=100 + i))
                except Exception as exc:
                    write_errors.append(exc)
                time.sleep(0.001)

        r = threading.Thread(target=reader)
        w = threading.Thread(target=writer)
        r.start()
        w.start()
        w.join()
        stop.set()
        r.join()

        assert not read_errors
        assert not write_errors
