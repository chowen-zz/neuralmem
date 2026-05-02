"""Cohere Embedding backend — requires neuralmem[cohere]."""
from __future__ import annotations

from collections.abc import Sequence

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError, NeuralMemError
from neuralmem.embedding.base import EmbeddingBackend

_KNOWN_DIMS: dict[str, int] = {
    "embed-multilingual-v3.0": 1024,
    "embed-english-v3.0": 1024,
    "embed-multilingual-light-v3.0": 384,
    "embed-english-light-v3.0": 384,
}


class CohereEmbedding(EmbeddingBackend):
    def __init__(self, config: NeuralMemConfig) -> None:
        try:
            import cohere as _cohere
        except ImportError as exc:
            raise NeuralMemError(
                "Install neuralmem[cohere] to use CohereEmbedding: pip install 'neuralmem[cohere]'"
            ) from exc
        if not config.cohere_api_key:
            raise ConfigError(
                "cohere_api_key is required for CohereEmbedding. Set NEURALMEM_COHERE_API_KEY."
            )
        self._client = _cohere.ClientV2(api_key=config.cohere_api_key)
        self._model = config.cohere_embedding_model

    @property
    def dimension(self) -> int:
        return _KNOWN_DIMS.get(self._model, 1024)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embed(
                texts=list(texts),
                model=self._model,
                input_type="search_document",
                embedding_types=["float"],
            )
            return [list(vec) for vec in response.embeddings.float_]
        except Exception as exc:
            raise EmbeddingError(f"Cohere embedding failed: {exc}") from exc
