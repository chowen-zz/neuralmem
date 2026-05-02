"""Tests for MemoryConsolidator.merge_similar() — real implementation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from neuralmem.core.types import Memory
from neuralmem.lifecycle.consolidation import (
    MemoryConsolidator,
    _build_groups_from_sim_matrix,
    _cosine_similarity,
    _cosine_similarity_matrix,
    _merge_memories,
)

# ─────────────────────── helpers ───────────────────────

def _make_memory(
    content: str,
    importance: float = 0.5,
    created_minutes_ago: int = 0,
    tags: tuple[str, ...] = (),
    entity_ids: tuple[str, ...] = (),
    access_count: int = 0,
) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        content=content,
        importance=importance,
        created_at=now - timedelta(minutes=created_minutes_ago),
        last_accessed=now,
        tags=tags,
        entity_ids=entity_ids,
        access_count=access_count,
    )


def _mock_storage(memories: list[Memory] | None = None) -> MagicMock:
    s = MagicMock()
    s.list_memories.return_value = memories or []
    return s


def _mock_embedder(mapping: dict[str, list[float]]) -> MagicMock:
    """Create a mock embedder that maps content substrings to vectors."""
    e = MagicMock()

    def encode_one(text: str) -> list[float]:
        for key, vec in mapping.items():
            if key in text:
                return vec
        # fallback: random-ish deterministic vector
        return [0.1, 0.1, 0.1]

    e.encode_one.side_effect = encode_one
    return e


# ─────────────────── _cosine_similarity ────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_partial_similarity(self):
        v1 = [1.0, 0.0]
        v2 = [0.7071, 0.7071]
        sim = _cosine_similarity(v1, v2)
        assert 0.7 < sim < 0.72


# ─────────────────── numpy similarity matrix ───────────

class TestCosineSimilarityMatrix:
    def test_identity(self):
        vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        mat = _cosine_similarity_matrix(vecs)
        assert mat.shape == (3, 3)
        # Diagonal should be ~1.0
        np.testing.assert_allclose(np.diag(mat), [1.0, 1.0, 1.0], atol=1e-6)
        # Off-diagonal should be ~0.0
        assert mat[0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_identical_rows(self):
        vecs = np.array([[1, 0], [1, 0]], dtype=np.float32)
        mat = _cosine_similarity_matrix(vecs)
        assert mat[0, 1] == pytest.approx(1.0, abs=1e-6)

    def test_zero_row_no_crash(self):
        vecs = np.array([[0, 0], [1, 0]], dtype=np.float32)
        mat = _cosine_similarity_matrix(vecs)
        # Zero row norm → treated as zero vector → similarity 0
        assert mat[0, 1] == pytest.approx(0.0, abs=1e-6)


# ─────────────────── _build_groups_from_sim_matrix ─────

class TestBuildGroups:
    def test_no_similar_pairs(self):
        # Three orthogonal vectors → no groups
        sim = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        groups = _build_groups_from_sim_matrix(sim, threshold=0.85)
        assert groups == []

    def test_one_pair(self):
        sim = np.array([
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ])
        groups = _build_groups_from_sim_matrix(sim, threshold=0.85)
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1]

    def test_chain_group(self):
        # A~B and B~C but NOT A~C → greedy still groups all three
        # because B bridges A and C
        sim = np.array([
            [1.0, 0.9, 0.5],
            [0.9, 1.0, 0.9],
            [0.5, 0.9, 1.0],
        ])
        groups = _build_groups_from_sim_matrix(sim, threshold=0.85)
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1, 2]

    def test_two_separate_groups(self):
        sim = np.array([
            [1.0, 0.9, 0.0, 0.0],
            [0.9, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.95],
            [0.0, 0.0, 0.95, 1.0],
        ])
        groups = _build_groups_from_sim_matrix(sim, threshold=0.85)
        assert len(groups) == 2
        assert sorted(groups[0]) == [0, 1]
        assert sorted(groups[1]) == [2, 3]


# ─────────────────── _merge_memories ───────────────────

class TestMergeMemories:
    def test_keeps_shorter_content(self):
        m1 = _make_memory("short")
        m2 = _make_memory("a much longer content string here")
        result = _merge_memories(m1, m2)
        assert result["content"] == "short"

    def test_merges_tags(self):
        m1 = _make_memory("a", tags=("tag1", "tag2"))
        m2 = _make_memory("b", tags=("tag2", "tag3"))
        result = _merge_memories(m1, m2)
        assert set(result["tags"]) == {"tag1", "tag2", "tag3"}

    def test_merges_entity_ids(self):
        m1 = _make_memory("a", entity_ids=("e1",))
        m2 = _make_memory("b", entity_ids=("e1", "e2"))
        result = _merge_memories(m1, m2)
        assert set(result["entity_ids"]) == {"e1", "e2"}

    def test_takes_max_importance(self):
        m1 = _make_memory("a", importance=0.3)
        m2 = _make_memory("b", importance=0.8)
        result = _merge_memories(m1, m2)
        assert result["importance"] == pytest.approx(0.8)

    def test_sums_access_count(self):
        m1 = _make_memory("a", access_count=5)
        m2 = _make_memory("b", access_count=3)
        result = _merge_memories(m1, m2)
        assert result["access_count"] == 8


# ──────────────── MemoryConsolidator.merge_similar ─────

class TestMergeSimilar:
    """Tests for the full merge_similar workflow."""

    def test_stub_without_deps(self):
        """No storage/embedder → returns 0 (stub fallback)."""
        c = MemoryConsolidator()
        assert c.merge_similar() == 0

    def test_empty_storage(self):
        s = _mock_storage([])
        c = MemoryConsolidator(storage=s, embedder=MagicMock())
        assert c.merge_similar() == 0

    def test_single_memory(self):
        m = _make_memory("only one")
        s = _mock_storage([m])
        c = MemoryConsolidator(storage=s, embedder=MagicMock())
        assert c.merge_similar() == 0

    def test_two_similar_marked_superseded(self):
        """Two similar memories: one becomes canonical, other is superseded."""
        m1 = _make_memory("user likes Python", importance=0.6, created_minutes_ago=10)
        m2 = _make_memory("user prefers Python coding", importance=0.5, created_minutes_ago=5)

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.98, 0.0, 0.02]

        storage = _mock_storage([m1, m2])
        embedder = _mock_embedder({"Python": vec1, "prefers": vec2})

        c = MemoryConsolidator(
            storage=storage, embedder=embedder, merge_threshold=0.9
        )
        result = c.merge_similar()
        assert result == 1

        # Two update_memory calls: canonical update + supersede
        assert storage.update_memory.call_count == 2
        calls = storage.update_memory.call_args_list

        # First call: update canonical (m1, because it's earlier)
        assert calls[0][0][0] == m1.id
        assert "content" in calls[0][1]

        # Second call: mark m2 as superseded
        assert calls[1][0][0] == m2.id
        assert calls[1][1]["is_active"] is False
        assert calls[1][1]["superseded_by"] == m1.id

    def test_two_dissimilar_not_merged(self):
        """Two different memories → no merge."""
        m1 = _make_memory("weather is nice")
        m2 = _make_memory("user likes programming")

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]

        storage = _mock_storage([m1, m2])
        embedder = _mock_embedder({"weather": vec1, "programming": vec2})

        c = MemoryConsolidator(
            storage=storage, embedder=embedder, merge_threshold=0.85
        )
        assert c.merge_similar() == 0
        storage.update_memory.assert_not_called()

    def test_uses_existing_embeddings(self):
        """If memories already have embeddings, embedder.encode_one is NOT called."""
        vec = [1.0, 0.0, 0.0]
        m1 = _make_memory("test A").model_copy(update={"embedding": vec})
        m2 = _make_memory("test B").model_copy(update={"embedding": vec})

        storage = _mock_storage([m1, m2])
        embedder = MagicMock()

        c = MemoryConsolidator(
            storage=storage, embedder=embedder, merge_threshold=0.85
        )
        result = c.merge_similar()
        embedder.encode_one.assert_not_called()
        assert result == 1

    def test_three_memories_two_similar(self):
        """Three memories, two similar → merge the similar pair, leave the third."""
        m1 = _make_memory("likes cats", importance=0.5, created_minutes_ago=20)
        m2 = _make_memory("enjoys cats", importance=0.6, created_minutes_ago=10)
        m3 = _make_memory("likes dogs", importance=0.4, created_minutes_ago=5)

        vec_cat = [1.0, 0.0, 0.0]
        vec_dog = [0.0, 1.0, 0.0]

        storage = _mock_storage([m1, m2, m3])
        embedder = _mock_embedder({"cats": vec_cat, "enjoys": vec_cat, "dogs": vec_dog})

        c = MemoryConsolidator(
            storage=storage, embedder=embedder, merge_threshold=0.85
        )
        result = c.merge_similar()
        assert result == 1  # Only m2 should be superseded

        # m1 is canonical (earliest), m2 is superseded
        calls = storage.update_memory.call_args_list
        assert calls[1][0][0] == m2.id
        assert calls[1][1]["is_active"] is False
        assert calls[1][1]["superseded_by"] == m1.id

    def test_group_of_three(self):
        """Three similar memories all merge into the earliest."""
        m1 = _make_memory("A", importance=0.3, created_minutes_ago=30)
        m2 = _make_memory("B", importance=0.8, created_minutes_ago=20)
        m3 = _make_memory("C", importance=0.5, created_minutes_ago=10)

        vec = [1.0, 0.0, 0.0]
        # All three have identical embeddings → cosine sim = 1.0
        for m in [m1, m2, m3]:
            m = m.model_copy(update={"embedding": vec})

        storage = _mock_storage([
            m1.model_copy(update={"embedding": vec}),
            m2.model_copy(update={"embedding": vec}),
            m3.model_copy(update={"embedding": vec}),
        ])

        c = MemoryConsolidator(
            storage=storage,
            embedder=MagicMock(),
            merge_threshold=0.85,
        )
        result = c.merge_similar()
        assert result == 2  # m2 and m3 superseded

        calls = storage.update_memory.call_args_list
        # First call: canonical update (m1 — earliest)
        assert calls[0][0][0] == m1.id
        # Canonical importance should be max(0.3, 0.8, 0.5) = 0.8
        assert calls[0][1]["importance"] == pytest.approx(0.8)
        # Canonical access_count should be sum
        assert calls[0][1]["access_count"] == 0
        # m2 and m3 are superseded by m1
        superseded_ids = {calls[i][0][0] for i in [1, 2]}
        assert superseded_ids == {m2.id, m3.id}
        for i in [1, 2]:
            assert calls[i][1]["is_active"] is False
            assert calls[i][1]["superseded_by"] == m1.id

    def test_merged_content_includes_unique_parts(self):
        """Merged canonical content should include unique content from others."""
        m1 = _make_memory("user likes Python", created_minutes_ago=20)
        m2 = _make_memory("user likes Rust too", created_minutes_ago=10)

        vec = [1.0, 0.0, 0.0]

        storage = _mock_storage([
            m1.model_copy(update={"embedding": vec}),
            m2.model_copy(update={"embedding": vec}),
        ])

        c = MemoryConsolidator(
            storage=storage, embedder=MagicMock(), merge_threshold=0.85
        )
        c.merge_similar()

        calls = storage.update_memory.call_args_list
        merged_content = calls[0][1]["content"]
        assert "user likes Python" in merged_content
        assert "user likes Rust too" in merged_content

    def test_merged_content_deduplicates(self):
        """If canonical content already contains other's content, don't duplicate."""
        m1 = _make_memory("user likes Python programming", created_minutes_ago=20)
        m2 = _make_memory("Python", created_minutes_ago=10)

        vec = [1.0, 0.0, 0.0]
        storage = _mock_storage([
            m1.model_copy(update={"embedding": vec}),
            m2.model_copy(update={"embedding": vec}),
        ])

        c = MemoryConsolidator(
            storage=storage, embedder=MagicMock(), merge_threshold=0.85
        )
        c.merge_similar()

        calls = storage.update_memory.call_args_list
        merged_content = calls[0][1]["content"]
        # "Python" is a substring of canonical content → not duplicated
        assert merged_content == "user likes Python programming"

    def test_tags_merged_deduplicated(self):
        m1 = _make_memory("A", tags=("python", "coding"), created_minutes_ago=20)
        m2 = _make_memory("B", tags=("coding", "favourite"), created_minutes_ago=10)

        vec = [1.0, 0.0, 0.0]
        storage = _mock_storage([
            m1.model_copy(update={"embedding": vec}),
            m2.model_copy(update={"embedding": vec}),
        ])

        c = MemoryConsolidator(
            storage=storage, embedder=MagicMock(), merge_threshold=0.85
        )
        c.merge_similar()

        calls = storage.update_memory.call_args_list
        tags = calls[0][1]["tags"]
        assert set(tags) == {"python", "coding", "favourite"}
        # No duplicates
        assert len(tags) == len(set(tags))

    def test_custom_threshold_parameter(self):
        """similarity_threshold override works."""
        m1 = _make_memory("A")
        m2 = _make_memory("B")

        vec1 = [1.0, 0.0]
        vec2 = [0.85, 0.52]  # sim ~0.85

        storage = _mock_storage([
            m1.model_copy(update={"embedding": vec1}),
            m2.model_copy(update={"embedding": vec2}),
        ])

        c = MemoryConsolidator(storage=storage, embedder=MagicMock())

        # With default threshold 0.85 → might merge
        result_default = c.merge_similar(similarity_threshold=0.99)
        assert result_default == 0  # threshold too high

    def test_limit_parameter(self):
        """limit parameter is passed to list_memories."""
        storage = _mock_storage([])
        embedder = MagicMock()

        c = MemoryConsolidator(storage=storage, embedder=embedder)
        c.merge_similar(limit=50)

        storage.list_memories.assert_called_once_with(user_id=None, limit=50)

    def test_user_id_forwarded(self):
        """user_id is passed through to list_memories."""
        storage = _mock_storage([])
        embedder = MagicMock()

        c = MemoryConsolidator(storage=storage, embedder=embedder)
        c.merge_similar(user_id="alice")

        storage.list_memories.assert_called_once_with(user_id="alice", limit=100)

    def test_merge_preserves_importance_roundtrip(self):
        """After merge, canonical importance = max of group."""
        m1 = _make_memory("X", importance=0.2, created_minutes_ago=20)
        m2 = _make_memory("Y", importance=0.9, created_minutes_ago=10)

        vec = [1.0, 0.0]
        storage = _mock_storage([
            m1.model_copy(update={"embedding": vec}),
            m2.model_copy(update={"embedding": vec}),
        ])

        c = MemoryConsolidator(
            storage=storage, embedder=MagicMock(), merge_threshold=0.85
        )
        c.merge_similar()

        calls = storage.update_memory.call_args_list
        assert calls[0][1]["importance"] == pytest.approx(0.9)

    def test_multiple_independent_groups(self):
        """Two separate pairs of similar memories → two independent merges."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]

        m1 = _make_memory("group A1", created_minutes_ago=30).model_copy(
            update={"embedding": vec_a}
        )
        m2 = _make_memory("group A2", created_minutes_ago=20).model_copy(
            update={"embedding": vec_a}
        )
        m3 = _make_memory("group B1", created_minutes_ago=15).model_copy(
            update={"embedding": vec_b}
        )
        m4 = _make_memory("group B2", created_minutes_ago=10).model_copy(
            update={"embedding": vec_b}
        )

        storage = _mock_storage([m1, m2, m3, m4])
        c = MemoryConsolidator(
            storage=storage, embedder=MagicMock(), merge_threshold=0.85
        )
        result = c.merge_similar()
        assert result == 2  # m2 and m4 superseded

        calls = storage.update_memory.call_args_list
        superseded = [
            call for call in calls if call[1].get("is_active") is False
        ]
        assert len(superseded) == 2


# ─────────────────── Integration style ─────────────────

class TestConsolidateIntegration:
    """Verify the consolidator integrates with NeuralMem.consolidate()."""

    @pytest.mark.skip(
        reason="Requires local embedding model (fastembed) which may not be installed"
    )
    def test_consolidate_returns_merged_count(self, tmp_path):
        """NeuralMem.consolidate() should include merged count."""
        from neuralmem.core.memory import NeuralMem

        db = str(tmp_path / "test.db")
        nm = NeuralMem(db_path=db)

        # Store two nearly identical memories
        nm.remember("User prefers Python for scripting", user_id="test")
        nm.remember("User prefers Python scripting", user_id="test")

        result = nm.consolidate(user_id="test")
        assert "merged" in result
        assert result["merged"] >= 0
