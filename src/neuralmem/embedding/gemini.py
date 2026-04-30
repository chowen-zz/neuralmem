"""Google Gemini Embedding backend — requires neuralmem[gemini]."""
from __future__ import annotations

from collections.abc import Sequence

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError, NeuralMemError
from neuralmem.embedding.base import EmbeddingBackend

_KNOWN_DIMS: dict[str, int] = {
    "text-embedding-004": 768,
    "embedding-001": 768,
}


class GeminiEmbedding(EmbeddingBackend):
    def __init__(self, config: NeuralMemConfig) -> None:
        try:
            import google.generativeai as _genai
        except ImportError as exc:
            raise NeuralMemError(
                "Install neuralmem[gemini] to use GeminiEmbedding: pip install 'neuralmem[gemini]'"
            ) from exc
        if not config.gemini_api_key:
            raise ConfigError(
                "gemini_api_key is required for GeminiEmbedding. Set NEURALMEM_GEMINI_API_KEY."
            )
        _genai.configure(api_key=config.gemini_api_key)
        self._genai = _genai
        self._model = config.gemini_embedding_model

    @property
    def dimension(self) -> int:
        return _KNOWN_DIMS.get(self._model, 768)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return [
                self._genai.embed_content(model=self._model, content=text)["embedding"]
                for text in texts
            ]
        except Exception as exc:
            raise EmbeddingError(f"Gemini embedding failed: {exc}") from exc
