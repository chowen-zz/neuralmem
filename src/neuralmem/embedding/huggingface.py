"""HuggingFace Inference API Embedding backend — uses httpx (no extra install needed)."""
from __future__ import annotations

from collections.abc import Sequence

import httpx

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError
from neuralmem.embedding.base import EmbeddingBackend

_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class HuggingFaceEmbedding(EmbeddingBackend):
    def __init__(self, config: NeuralMemConfig) -> None:
        if not config.hf_api_key:
            raise ConfigError(
                "hf_api_key is required for HuggingFaceEmbedding. Set NEURALMEM_HF_API_KEY."
            )
        self._api_key = config.hf_api_key
        self._model = config.hf_model
        base = config.hf_inference_url.rstrip("/")
        self._url = f"{base}/pipeline/feature-extraction/{self._model}"

    @property
    def dimension(self) -> int:
        return _KNOWN_DIMS.get(self._model, 1024)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = httpx.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"inputs": list(texts), "options": {"wait_for_model": True}},
                timeout=30.0,
            )
            response.raise_for_status()
            data: list = response.json()
            results: list[list[float]] = []
            for item in data:
                if item and isinstance(item[0], list):
                    # Mean pooling over token dimension
                    n = len(item)
                    dim = len(item[0])
                    pooled = [sum(item[t][d] for t in range(n)) / n for d in range(dim)]
                    results.append(pooled)
                else:
                    results.append(item)
            return results
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"HuggingFace API error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise EmbeddingError(f"HuggingFace embedding failed: {exc}") from exc
