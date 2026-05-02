import sys
from unittest.mock import MagicMock, patch

import pytest

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _mock_openai_sdk():
    mock = MagicMock()
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=[0.1] * 1536)]
    mock.OpenAI.return_value.embeddings.create.return_value = mock_resp
    return mock


def test_openai_missing_api_key_raises_config_error():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.openai" in sys.modules:
            del sys.modules["neuralmem.embedding.openai"]
        from neuralmem.embedding.openai import OpenAIEmbedding
        with pytest.raises(ConfigError, match="openai_api_key"):
            OpenAIEmbedding(cfg())


def test_openai_encode_returns_vectors():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.openai" in sys.modules:
            del sys.modules["neuralmem.embedding.openai"]
        from neuralmem.embedding.openai import OpenAIEmbedding
        embedder = OpenAIEmbedding(cfg(openai_api_key="sk-test"))
        result = embedder.encode(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == 1536


def test_openai_encode_empty_returns_empty():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.openai" in sys.modules:
            del sys.modules["neuralmem.embedding.openai"]
        from neuralmem.embedding.openai import OpenAIEmbedding
        embedder = OpenAIEmbedding(cfg(openai_api_key="sk-test"))
        assert embedder.encode([]) == []


def test_openai_api_failure_raises_embedding_error():
    mock_sdk = _mock_openai_sdk()
    mock_sdk.OpenAI.return_value.embeddings.create.side_effect = Exception("rate limit")
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.openai" in sys.modules:
            del sys.modules["neuralmem.embedding.openai"]
        from neuralmem.embedding.openai import OpenAIEmbedding
        embedder = OpenAIEmbedding(cfg(openai_api_key="sk-test"))
        with pytest.raises(EmbeddingError, match="OpenAI embedding failed"):
            embedder.encode(["hello"])


def test_openai_dimension_large_model():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.openai" in sys.modules:
            del sys.modules["neuralmem.embedding.openai"]
        from neuralmem.embedding.openai import OpenAIEmbedding
        e = OpenAIEmbedding(
            cfg(openai_api_key="sk-test", openai_embedding_model="text-embedding-3-large")
        )
        assert e.dimension == 3072
