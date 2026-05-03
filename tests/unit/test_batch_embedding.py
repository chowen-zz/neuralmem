"""Tests for BatchEmbeddingProcessor."""
from __future__ import annotations

from unittest.mock import MagicMock

from neuralmem.perf.batch_embedding import (
    BatchEmbeddingProcessor,
    BatchEmbeddingResult,
)


class TestBatchEmbeddingProcessor:
    def _mock_embedder(
        self,
        dim: int = 8,
        encode_side_effect=None,
    ):
        embedder = MagicMock()
        embedder.dimension = dim
        if encode_side_effect:
            embedder.encode.side_effect = encode_side_effect
        else:
            embedder.encode.return_value = [
                [0.1] * dim for _ in range(100)
            ]
        embedder.encode_one.return_value = [0.1] * dim
        return embedder

    def test_encode_batch_empty(self):
        embedder = self._mock_embedder()
        proc = BatchEmbeddingProcessor(embedder, batch_size=4)
        result = proc.encode_batch([])
        assert result == []

    def test_encode_batch_single(self):
        embedder = self._mock_embedder()
        embedder.encode.return_value = [[0.5] * 8]
        proc = BatchEmbeddingProcessor(embedder, batch_size=32)
        result = proc.encode_batch(["hello"])
        assert len(result) == 1
        assert result[0] == [0.5] * 8

    def test_encode_batch_chunking(self):
        embedder = self._mock_embedder()
        call_count = [0]

        def encode_side(texts):
            call_count[0] += 1
            return [[0.1] * 8 for _ in texts]

        embedder.encode.side_effect = encode_side
        proc = BatchEmbeddingProcessor(embedder, batch_size=3)
        texts = ["a", "b", "c", "d", "e", "e", "f", "g"]
        result = proc.encode_batch(texts)
        assert len(result) == 8
        # 8 texts / batch_size 3 = ceil(8/3) = 3 batches
        assert call_count[0] == 3

    def test_encode_batch_preserves_order(self):
        embedder = self._mock_embedder()
        texts = ["first", "second", "third"]

        def encode_side(t):
            return [[float(i)] for i in range(len(t))]

        embedder.encode.side_effect = encode_side
        proc = BatchEmbeddingProcessor(embedder, batch_size=2)
        result = proc.encode_batch(texts)
        assert len(result) == 3

    def test_encode_batch_with_retry_success(self):
        embedder = self._mock_embedder()
        embedder.encode.return_value = [[0.1] * 8]
        proc = BatchEmbeddingProcessor(embedder, batch_size=32)
        result = proc.encode_batch_with_retry(
            ["hello", "world"]
        )
        assert result.texts_processed == 2
        assert result.batches_processed == 1
        assert result.batches_failed == 0
        assert result.total_time > 0

    def test_encode_batch_with_retry_failure(self):
        embedder = self._mock_embedder()
        embedder.encode.side_effect = RuntimeError("API error")
        proc = BatchEmbeddingProcessor(embedder, batch_size=4)
        result = proc.encode_batch_with_retry(
            ["a", "b"], max_retries=2
        )
        assert result.batches_failed == 1
        assert result.batches_processed == 0
        # Should have zero vectors for failed items
        assert len(result.embeddings) == 2
        assert all(
            v == [0.0] * 8 for v in result.embeddings
        )

    def test_encode_batch_with_retry_empty(self):
        embedder = self._mock_embedder()
        proc = BatchEmbeddingProcessor(embedder, batch_size=4)
        result = proc.encode_batch_with_retry([])
        assert result.texts_processed == 0
        assert result.embeddings == []

    def test_encode_batch_with_retry_partial(self):
        embedder = self._mock_embedder()

        def encode_side(texts):
            if "a" in texts:
                raise RuntimeError("first batch fails")
            return [[0.1] * 8 for _ in texts]

        embedder.encode.side_effect = encode_side
        proc = BatchEmbeddingProcessor(embedder, batch_size=2)
        result = proc.encode_batch_with_retry(
            ["a", "b", "c", "d"], max_retries=2
        )
        # First batch fails after retries, second succeeds
        assert result.batches_failed == 1
        assert result.batches_processed == 1

    def test_estimate_cost(self):
        embedder = self._mock_embedder()
        proc = BatchEmbeddingProcessor(embedder, batch_size=10)
        texts = ["hello world"] * 20
        cost = proc.estimate_cost(texts)
        assert cost["total_texts"] == 20
        assert cost["estimated_batches"] == 2
        assert cost["batch_size"] == 10
        assert cost["estimated_tokens"] > 0

    def test_progress_callback(self):
        embedder = self._mock_embedder()
        embedder.encode.return_value = [[0.1] * 8]
        proc = BatchEmbeddingProcessor(embedder, batch_size=2)
        progress_calls: list[tuple[int, int]] = []
        proc.set_progress_callback(
            lambda c, t: progress_calls.append((c, t))
        )
        proc.encode_batch(["a", "b", "c"])
        assert len(progress_calls) == 2
        assert progress_calls[-1] == (3, 3)

    def test_batch_size_property(self):
        embedder = self._mock_embedder()
        proc = BatchEmbeddingProcessor(embedder, batch_size=16)
        assert proc.batch_size == 16

    def test_max_concurrent_property(self):
        embedder = self._mock_embedder()
        proc = BatchEmbeddingProcessor(
            embedder, batch_size=8, max_concurrent=2
        )
        assert proc.max_concurrent == 2

    def test_batch_embedding_result_dataclass(self):
        r = BatchEmbeddingResult()
        assert r.embeddings == []
        assert r.total_time == 0.0
        assert r.batches_processed == 0
        assert r.batches_failed == 0
        assert r.texts_processed == 0
