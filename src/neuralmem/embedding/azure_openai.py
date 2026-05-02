"""Azure OpenAI Embedding backend — requires neuralmem[openai]."""
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


class AzureOpenAIEmbedding(EmbeddingBackend):
    def __init__(self, config: NeuralMemConfig) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise NeuralMemError(
                "Install neuralmem[openai] to use AzureOpenAIEmbedding: "
                "pip install 'neuralmem[openai]'"
            ) from exc
        if not config.azure_endpoint:
            raise ConfigError(
                "azure_endpoint is required for AzureOpenAIEmbedding. Set NEURALMEM_AZURE_ENDPOINT."
            )
        if not config.azure_api_key:
            raise ConfigError(
                "azure_api_key is required for AzureOpenAIEmbedding. Set NEURALMEM_AZURE_API_KEY."
            )
        self._client = _openai.AzureOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
        )
        self._deployment = config.azure_deployment

    @property
    def dimension(self) -> int:
        return _KNOWN_DIMS.get(self._deployment, 1536)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(model=self._deployment, input=list(texts))
            return [item.embedding for item in response.data]
        except Exception as exc:
            raise EmbeddingError(f"Azure OpenAI embedding failed: {exc}") from exc
