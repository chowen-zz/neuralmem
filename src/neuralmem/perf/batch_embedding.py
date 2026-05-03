"""Batch embedding processor with adaptive sizing and retry logic."""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from neuralmem.core.protocols import EmbedderProtocol

_logger = logging.getLogger(__name__)


@dataclass
class BatchEmbeddingResult:
    """Result of a batch embedding operation."""
    embeddings: list[list[float]] = field(default_factory=list)
    total_time: float = 0.0
    batches_processed: int = 0
    batches_failed: int = 0
    texts_processed: int = 0


class BatchEmbeddingProcessor:
    """Batch embedding processor with adaptive sizing and retry logic.

    Splits large text lists into configurable batches, processes them
    through an EmbedderProtocol implementation, and adapts batch size
    based on response times.
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        *,
        batch_size: int = 32,
        max_concurrent: int = 4,
    ) -> None:
        self._embedder = embedder
        self._batch_size = batch_size
        self._max_concurrent = max_concurrent
        self._lock = threading.Lock()
        self._response_times: list[float] = []
        self._on_progress: Callable[[int, int], None] | None = None

    @property
    def batch_size(self) -> int:
        """Current batch size (may be adjusted adaptively)."""
        with self._lock:
            return self._batch_size

    @property
    def max_concurrent(self) -> int:
        """Maximum concurrent batch operations."""
        return self._max_concurrent

    def set_progress_callback(
        self, callback: Callable[[int, int], None] | None
    ) -> None:
        """Set a progress callback: on_progress(completed, total)."""
        self._on_progress = callback

    def encode_batch(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Batch encode texts with automatic chunking.

        Args:
            texts: List of text strings to encode.

        Returns:
            List of embeddings in the same order as input texts.
        """
        if not texts:
            return []

        effective_size = self._get_adaptive_batch_size()
        all_embeddings: list[list[float]] = []
        completed = 0
        total = len(texts)

        for i in range(0, len(texts), effective_size):
            chunk = texts[i : i + effective_size]
            t0 = time.monotonic()
            embeddings = self._embedder.encode(chunk)
            elapsed = time.monotonic() - t0
            self._record_response_time(elapsed, len(chunk))
            all_embeddings.extend(embeddings)
            completed += len(chunk)
            if self._on_progress is not None:
                self._on_progress(completed, total)

        return all_embeddings

    def encode_batch_with_retry(
        self,
        texts: list[str],
        max_retries: int = 3,
    ) -> BatchEmbeddingResult:
        """Encode texts with retry logic for failed batches.

        Args:
            texts: List of text strings to encode.
            max_retries: Maximum retry attempts per batch.

        Returns:
            BatchEmbeddingResult with embeddings and stats.
        """
        if not texts:
            return BatchEmbeddingResult()

        effective_size = self._get_adaptive_batch_size()
        result = BatchEmbeddingResult()
        completed = 0
        total = len(texts)
        start = time.monotonic()

        for i in range(0, len(texts), effective_size):
            chunk = texts[i : i + effective_size]
            success = False
            for attempt in range(max_retries):
                try:
                    t0 = time.monotonic()
                    embeddings = self._embedder.encode(chunk)
                    elapsed = time.monotonic() - t0
                    self._record_response_time(elapsed, len(chunk))
                    result.embeddings.extend(embeddings)
                    result.batches_processed += 1
                    result.texts_processed += len(chunk)
                    success = True
                    break
                except Exception as exc:
                    _logger.warning(
                        "Batch %d attempt %d failed: %s",
                        i // effective_size,
                        attempt + 1,
                        exc,
                    )
            if not success:
                result.batches_failed += 1
                # Append zero vectors for failed batch
                dim = self._embedder.dimension
                result.embeddings.extend(
                    [[0.0] * dim] * len(chunk)
                )
                result.texts_processed += len(chunk)

            completed += len(chunk)
            if self._on_progress is not None:
                self._on_progress(completed, total)

        result.total_time = time.monotonic() - start
        return result

    def estimate_cost(self, texts: list[str]) -> dict[str, int]:
        """Estimate API calls and tokens for a text list.

        Args:
            texts: List of text strings.

        Returns:
            Dict with 'total_texts', 'estimated_batches',
            'estimated_tokens', 'batch_size'.
        """
        effective_size = self._get_adaptive_batch_size()
        n_batches = max(
            1, (len(texts) + effective_size - 1) // effective_size
        )
        # Rough token estimate: ~4 chars per token
        total_chars = sum(len(t) for t in texts)
        est_tokens = max(1, total_chars // 4)
        return {
            "total_texts": len(texts),
            "estimated_batches": n_batches,
            "estimated_tokens": est_tokens,
            "batch_size": effective_size,
        }

    def _get_adaptive_batch_size(self) -> int:
        """Compute adaptive batch size based on recent response times."""
        with self._lock:
            if len(self._response_times) < 3:
                return self._batch_size
            # Use recent average to adapt
            recent = self._response_times[-5:]
            avg_time = sum(recent) / len(recent)
            if avg_time > 2.0:
                return max(1, self._batch_size // 2)
            if avg_time < 0.5:
                return min(256, self._batch_size * 2)
            return self._batch_size

    def _record_response_time(
        self, elapsed: float, batch_len: int
    ) -> None:
        """Record normalized response time per item."""
        per_item = elapsed / max(1, batch_len)
        with self._lock:
            self._response_times.append(per_item)
            # Keep only last 20 samples
            if len(self._response_times) > 20:
                self._response_times = self._response_times[-20:]
