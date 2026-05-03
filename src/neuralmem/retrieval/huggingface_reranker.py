"""HuggingFaceReranker — uses HuggingFace Inference API for reranking."""
from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger(__name__)


class HuggingFaceReranker:
    """Reranker backed by HuggingFace's InferenceClient.

    Tries the ``feature_extraction`` endpoint to compute similarity scores
    between the query and each document.  Falls back gracefully.

    Requires the ``huggingface_hub`` package::

        pip install huggingface_hub

    Usage::

        reranker = HuggingFaceReranker(api_key="hf_...")
        scores = reranker.rerank("query", ["doc1", "doc2"], top_k=5)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        **kwargs: Any,
    ) -> None:
        try:
            from huggingface_hub import InferenceClient  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for HuggingFaceReranker. "
                "Install with: pip install huggingface_hub"
            ) from exc

        self._client = InferenceClient(token=api_key, **kwargs)
        self._model = model

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Rerank *documents* against *query*.

        Returns:
            List of ``(original_index, relevance_score)`` sorted by score
            descending, limited to *top_k* results.
        """
        if not documents:
            return []

        try:
            return self._rerank_via_feature_extraction(
                query, documents, top_k
            )
        except Exception as exc:
            _logger.warning(
                "feature_extraction failed (%s), "
                "falling back to pairwise similarity",
                exc,
            )
            return self._rerank_pairwise(query, documents, top_k)

    def _rerank_via_feature_extraction(
        self,
        query: str,
        documents: list[str],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Use feature_extraction to get embeddings, then cosine similarity."""
        import numpy as np

        query_emb = np.array(
            self._client.feature_extraction(
                query, model=self._model
            ),
            dtype=np.float32,
        ).flatten()

        scores: list[tuple[int, float]] = []
        for idx, doc in enumerate(documents):
            doc_emb = np.array(
                self._client.feature_extraction(
                    doc, model=self._model
                ),
                dtype=np.float32,
            ).flatten()

            norm_q = np.linalg.norm(query_emb)
            norm_d = np.linalg.norm(doc_emb)
            if norm_q == 0 or norm_d == 0:
                sim = 0.0
            else:
                sim = float(
                    np.dot(query_emb, doc_emb) / (norm_q * norm_d)
                )
            scores.append((idx, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _rerank_pairwise(
        self,
        query: str,
        documents: list[str],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Fallback: score each query-doc pair via the model.

        For cross-encoder models, the feature_extraction of a
        query-doc pair yields a scalar score.
        """
        scores: list[tuple[int, float]] = []
        for idx, doc in enumerate(documents):
            try:
                result = self._client.feature_extraction(
                    f"{query} [SEP] {doc}",
                    model=self._model,
                )
                score = float(result[0]) if result else 0.0
            except Exception:
                score = 0.0
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
