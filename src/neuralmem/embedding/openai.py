"""OpenAI Embedding backend — requires neuralmem[openai]."""
from __future__ import annotations

from collections.abc import Sequence

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError, NeuralMemError
from neuralmem.embedding.base import EmbeddingBackend

_KNOWN_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedding(EmbeddingBackend):
    def __init__(self, config: NeuralMemConfig) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise NeuralMemError(
                "Install neuralmem[openai] to use OpenAIEmbedding: pip install 'neuralmem[openai]'"
            ) from exc
        if not config.openai_api_key:
            raise ConfigError(
                "openai_api_key is required for OpenAIEmbedding. "
                "Set NEURALMEM_OPENAI_API_KEY or OPENAI_API_KEY."
            )
        self._client = _openai.OpenAI(api_key=config.openai_api_key)
        self._model = config.openai_embedding_model

    @property
    def dimension(self) -> int:
        return _KNOWN_DIMS.get(self._model, 1536)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(model=self._model, input=list(texts))
            return [item.embedding for item in response.data]
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc
