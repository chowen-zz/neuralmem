"""Embedding backend factory — maps config.embedding_provider to a concrete EmbeddingBackend."""
from __future__ import annotations

from neuralmem.core.config import NeuralMemConfig
from neuralmem.embedding.base import EmbeddingBackend


def get_embedder(config: NeuralMemConfig) -> EmbeddingBackend:
    """Return the configured embedding backend. Unknown providers fall back to LocalEmbedding."""
    match config.embedding_provider:
        case "openai":
            from neuralmem.embedding.openai import OpenAIEmbedding
            return OpenAIEmbedding(config)
        case "cohere":
            from neuralmem.embedding.cohere import CohereEmbedding
            return CohereEmbedding(config)
        case "gemini":
            from neuralmem.embedding.gemini import GeminiEmbedding
            return GeminiEmbedding(config)
        case "huggingface":
            from neuralmem.embedding.huggingface import HuggingFaceEmbedding
            return HuggingFaceEmbedding(config)
        case "ollama":
            from neuralmem.embedding.ollama import OllamaEmbeddingBackend
            return OllamaEmbeddingBackend(config)
        case "azure_openai":
            from neuralmem.embedding.azure_openai import AzureOpenAIEmbedding
            return AzureOpenAIEmbedding(config)
        case _:
            from neuralmem.embedding.local import LocalEmbedding
            return LocalEmbedding(config)
