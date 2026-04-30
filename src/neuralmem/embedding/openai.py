"""OpenAI Embedding 后端实现"""
from __future__ import annotations

from collections.abc import Sequence

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import EmbeddingError
from neuralmem.embedding.base import EmbeddingBackend

_OPENAI_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedding(EmbeddingBackend):
    """基于 OpenAI Embeddings API 的后端实现。"""

    def __init__(self, config: NeuralMemConfig) -> None:
        self._config = config
        self._model_name = config.openai_embedding_model
        import openai
        self._client = openai.OpenAI(api_key=config.openai_api_key)

    @property
    def dimension(self) -> int:
        return _OPENAI_DIMS.get(self._model_name, 1536)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(
                model=self._model_name,
                input=list(texts),
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}") from e
