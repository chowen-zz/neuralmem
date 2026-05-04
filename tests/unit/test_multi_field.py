"""Tests for multi-field embedding strategy."""
import pytest
from unittest.mock import MagicMock

from neuralmem.embedding.multi_field import (
    FieldEmbedding,
    MultiFieldEmbeddings,
    MultiFieldEmbedder,
    cosine_similarity,
)


class TestFieldEmbedding:
    def test_creation(self):
        fe = FieldEmbedding("content", [0.1, 0.2, 0.3], "test-model")
        assert fe.field_name == "content"
        assert fe.vector == [0.1, 0.2, 0.3]
        assert fe.model_name == "test-model"

    def test_defaults(self):
        fe = FieldEmbedding("summary", [0.1, 0.2])
        assert fe.model_name == "default"


class TestMultiFieldEmbeddings:
    def test_add_and_get(self):
        mfe = MultiFieldEmbeddings("mem-1")
        mfe.add_field("content", [0.1, 0.2], "model-a")
        mfe.add_field("summary", [0.3, 0.4], "model-b")

        fe = mfe.get_field("content")
        assert fe is not None
        assert fe.vector == [0.1, 0.2]
        assert mfe.get_field("missing") is None

    def test_to_dict(self):
        mfe = MultiFieldEmbeddings("mem-1")
        mfe.add_field("content", [0.1, 0.2])
        d = mfe.to_dict()
        assert d["memory_id"] == "mem-1"
        assert "content" in d["fields"]
        assert d["fields"]["content"]["vector"] == [0.1, 0.2]

    def test_from_dict(self):
        data = {
            "memory_id": "mem-2",
            "fields": {
                "content": {"field_name": "content", "vector": [0.5, 0.6], "model_name": "m1"}
            }
        }
        mfe = MultiFieldEmbeddings.from_dict(data)
        assert mfe.memory_id == "mem-2"
        assert mfe.get_field("content").vector == [0.5, 0.6]


class TestCosineSimilarity:
    def test_identical(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestMultiFieldEmbedder:
    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock()
        embedder.encode.return_value = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ]
        return embedder

    def test_embed_memory(self, mock_embedder):
        mfe = MultiFieldEmbedder(mock_embedder, fields=["content", "summary"])
        result = mfe.embed_memory("m1", "hello world", "greeting")

        assert result.memory_id == "m1"
        assert result.get_field("content") is not None
        assert result.get_field("summary") is not None
        mock_embedder.encode.assert_called_once()

    def test_embed_memory_without_summary(self, mock_embedder):
        mfe = MultiFieldEmbedder(mock_embedder, fields=["content", "summary"])
        result = mfe.embed_memory("m1", "hello world")
        assert result.get_field("content") is not None
        assert result.get_field("summary") is None

    def test_search_by_field(self, mock_embedder):
        mfe = MultiFieldEmbedder(mock_embedder, fields=["content"])
        candidates = [
            MultiFieldEmbeddings("m1"),
            MultiFieldEmbeddings("m2"),
        ]
        candidates[0].add_field("content", [1.0, 0.0, 0.0, 0.0])
        candidates[1].add_field("content", [0.0, 1.0, 0.0, 0.0])

        query = [1.0, 0.0, 0.0, 0.0]
        results = mfe.search_by_field(query, candidates, "content", top_k=2)
        assert len(results) == 2
        assert results[0][0] == "m1"
        assert results[0][1] > results[1][1]

    def test_hybrid_search(self, mock_embedder):
        mfe = MultiFieldEmbedder(mock_embedder, fields=["content", "summary"])
        candidates = [MultiFieldEmbeddings("m1"), MultiFieldEmbeddings("m2")]
        candidates[0].add_field("content", [1.0, 0.0, 0.0, 0.0])
        candidates[0].add_field("summary", [0.5, 0.5, 0.0, 0.0])
        candidates[1].add_field("content", [0.0, 1.0, 0.0, 0.0])
        candidates[1].add_field("summary", [0.0, 0.5, 0.5, 0.0])

        query = [1.0, 0.0, 0.0, 0.0]
        results = mfe.hybrid_search(query, candidates, {"content": 2.0, "summary": 1.0})
        assert results[0][0] == "m1"

    def test_field_templates(self, mock_embedder):
        mfe = MultiFieldEmbedder(
            mock_embedder,
            fields=["custom"],
            field_templates={"custom": "Topic: {content}"},
        )
        result = mfe.embed_memory("m1", "AI memory")
        assert result.get_field("custom") is not None
