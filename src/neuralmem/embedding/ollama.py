"""Ollama local embedding backend — calls /api/embed via httpx."""
from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import EmbeddingError
from neuralmem.embedding.base import EmbeddingBackend

_logger = logging.getLogger(__name__)

_KNOWN_DIMS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "bge-m3": 1024,
    "nomic-embed-text-v1.5": 768,
}


class OllamaEmbeddingBackend(EmbeddingBackend):
    """Ollama embedding backend using /api/embed endpoint.

    Requirements:
    - Ollama server running (default http://localhost:11434)
    - Model pulled (e.g. ``ollama pull nomic-embed-text``)

    Supports both single and batch encoding.
    """

    def __init__(self, config: NeuralMemConfig) -> None:
        base_url = config.ollama_url.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            from neuralmem.core.exceptions import ConfigError

            raise ConfigError(
                f"ollama_url must start with http:// or https://, got: {base_url!r}"
            )
        self._base_url = base_url
        self._model = config.ollama_embedding_model

    @property
    def dimension(self) -> int:
        """Return vector dimension; known models have exact values, default 768."""
        return _KNOWN_DIMS.get(self._model, 768)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Batch-encode texts into embedding vectors via Ollama /api/embed."""
        if not texts:
            return []
        try:
            response = httpx.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": list(texts)},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list):
                raise EmbeddingError(
                    f"Unexpected Ollama response: 'embeddings' is"
                    f" {type(embeddings).__name__}, expected list"
                )
            if len(embeddings) != len(texts):
                raise EmbeddingError(
                    f"Ollama returned {len(embeddings)} embeddings for {len(texts)} texts"
                )
            return [list(vec) for vec in embeddings]
        except httpx.HTTPStatusError as exc:
            body_preview = exc.response.text[:200]
            raise EmbeddingError(
                f"Ollama API error {exc.response.status_code}: {body_preview}"
            ) from exc
        except httpx.ConnectError as exc:
            raise EmbeddingError(
                f"Cannot connect to Ollama at {self._base_url}. Is the server running?"
            ) from exc
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Ollama embedding failed: {exc}") from exc

    def encode_one(self, text: str) -> list[float]:
        """Encode a single text into an embedding vector."""
        return self.encode([text])[0]
