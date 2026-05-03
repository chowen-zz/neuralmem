"""RerankerFactory — registry pattern for pluggable reranking backends."""
from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger(__name__)


class RerankerFactory:
    """Factory for creating reranker instances by name.

    Usage::

        RerankerFactory.register('cohere', CohereReranker)
        reranker = RerankerFactory.create('cohere', api_key='...')
        scores = reranker.rerank("query", ["doc1", "doc2"], top_k=5)
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, reranker_cls: type) -> None:
        """Register a reranker class under *name*."""
        name_lower = name.lower()
        if name_lower in cls._registry:
            _logger.warning(
                "Overwriting registered reranker '%s'", name_lower
            )
        cls._registry[name_lower] = reranker_cls
        _logger.debug("Registered reranker: %s -> %s", name_lower, reranker_cls)

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> object:
        """Create a reranker instance by *name*, forwarding *kwargs*."""
        name_lower = name.lower()
        if name_lower not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "(none)"
            raise ValueError(
                f"Unknown reranker '{name}'. "
                f"Available: {available}"
            )
        reranker_cls = cls._registry[name_lower]
        return reranker_cls(**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """Return sorted list of registered reranker names."""
        return sorted(cls._registry)

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations (useful for testing)."""
        cls._registry.clear()
