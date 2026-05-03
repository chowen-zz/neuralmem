"""CohereReranker — uses Cohere's rerank API."""
from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger(__name__)


class CohereReranker:
    """Reranker backed by the Cohere ``rerank`` endpoint.

    Requires the ``cohere`` Python package::

        pip install cohere

    Usage::

        reranker = CohereReranker(api_key="...")
        scores = reranker.rerank("query", ["doc1", "doc2"], top_k=5)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v2.0",
        **kwargs: Any,
    ) -> None:
        try:
            import cohere  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "cohere is required for CohereReranker. "
                "Install with: pip install cohere"
            ) from exc

        self._client = cohere.Client(api_key, **kwargs)
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

        response = self._client.rerank(
            query=query,
            documents=documents,
            model=self._model,
            top_n=min(top_k, len(documents)),
        )

        results: list[tuple[int, float]] = []
        for item in response.results:
            results.append((item.index, item.relevance_score))
        return results
