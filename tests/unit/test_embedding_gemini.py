import sys
from unittest.mock import MagicMock, patch

import pytest

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _mock_genai_sdk():
    mock = MagicMock()
    mock.embed_content.return_value = {"embedding": [0.1] * 768}
    return mock


def _patch_genai(mock_sdk):
    """Patch both google.generativeai and google namespace."""
    mock_google = MagicMock()
    mock_google.generativeai = mock_sdk
    return patch.dict(sys.modules, {
        "google": mock_google,
        "google.generativeai": mock_sdk,
    })


def test_gemini_missing_api_key_raises():
    mock_sdk = _mock_genai_sdk()
    with _patch_genai(mock_sdk):
        if "neuralmem.embedding.gemini" in sys.modules:
            del sys.modules["neuralmem.embedding.gemini"]
        from neuralmem.embedding.gemini import GeminiEmbedding
        with pytest.raises(ConfigError, match="gemini_api_key"):
            GeminiEmbedding(cfg())


def test_gemini_encode_returns_vectors():
    mock_sdk = _mock_genai_sdk()
    with _patch_genai(mock_sdk):
        if "neuralmem.embedding.gemini" in sys.modules:
            del sys.modules["neuralmem.embedding.gemini"]
        from neuralmem.embedding.gemini import GeminiEmbedding
        embedder = GeminiEmbedding(cfg(gemini_api_key="test-key"))
        result = embedder.encode(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 768


def test_gemini_encode_empty_returns_empty():
    mock_sdk = _mock_genai_sdk()
    with _patch_genai(mock_sdk):
        if "neuralmem.embedding.gemini" in sys.modules:
            del sys.modules["neuralmem.embedding.gemini"]
        from neuralmem.embedding.gemini import GeminiEmbedding
        embedder = GeminiEmbedding(cfg(gemini_api_key="test-key"))
        assert embedder.encode([]) == []


def test_gemini_api_failure_raises_embedding_error():
    mock_sdk = _mock_genai_sdk()
    mock_sdk.embed_content.side_effect = Exception("model not found")
    with _patch_genai(mock_sdk):
        if "neuralmem.embedding.gemini" in sys.modules:
            del sys.modules["neuralmem.embedding.gemini"]
        from neuralmem.embedding.gemini import GeminiEmbedding
        embedder = GeminiEmbedding(cfg(gemini_api_key="test-key"))
        with pytest.raises(EmbeddingError, match="Gemini embedding failed"):
            embedder.encode(["hello"])


def test_gemini_dimension():
    mock_sdk = _mock_genai_sdk()
    with _patch_genai(mock_sdk):
        if "neuralmem.embedding.gemini" in sys.modules:
            del sys.modules["neuralmem.embedding.gemini"]
        from neuralmem.embedding.gemini import GeminiEmbedding
        e = GeminiEmbedding(cfg(gemini_api_key="k"))
        assert e.dimension == 768
