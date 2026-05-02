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
    mock.AzureOpenAI.return_value.embeddings.create.return_value = mock_resp
    return mock


def test_azure_missing_endpoint_raises():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.azure_openai" in sys.modules:
            del sys.modules["neuralmem.embedding.azure_openai"]
        from neuralmem.embedding.azure_openai import AzureOpenAIEmbedding
        with pytest.raises(ConfigError, match="azure_endpoint"):
            AzureOpenAIEmbedding(cfg(azure_api_key="key"))


def test_azure_missing_api_key_raises():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.azure_openai" in sys.modules:
            del sys.modules["neuralmem.embedding.azure_openai"]
        from neuralmem.embedding.azure_openai import AzureOpenAIEmbedding
        with pytest.raises(ConfigError, match="azure_api_key"):
            AzureOpenAIEmbedding(cfg(azure_endpoint="https://my.openai.azure.com"))


def test_azure_encode_returns_vectors():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.azure_openai" in sys.modules:
            del sys.modules["neuralmem.embedding.azure_openai"]
        from neuralmem.embedding.azure_openai import AzureOpenAIEmbedding
        e = AzureOpenAIEmbedding(
            cfg(azure_endpoint="https://x.openai.azure.com", azure_api_key="k")
        )
        result = e.encode(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 1536


def test_azure_encode_empty_returns_empty():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.azure_openai" in sys.modules:
            del sys.modules["neuralmem.embedding.azure_openai"]
        from neuralmem.embedding.azure_openai import AzureOpenAIEmbedding
        e = AzureOpenAIEmbedding(
            cfg(azure_endpoint="https://x.openai.azure.com", azure_api_key="k")
        )
        assert e.encode([]) == []


def test_azure_api_failure_raises_embedding_error():
    mock_sdk = _mock_openai_sdk()
    mock_sdk.AzureOpenAI.return_value.embeddings.create.side_effect = Exception("auth failed")
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.azure_openai" in sys.modules:
            del sys.modules["neuralmem.embedding.azure_openai"]
        from neuralmem.embedding.azure_openai import AzureOpenAIEmbedding
        e = AzureOpenAIEmbedding(
            cfg(azure_endpoint="https://x.openai.azure.com", azure_api_key="k")
        )
        with pytest.raises(EmbeddingError, match="Azure OpenAI embedding failed"):
            e.encode(["hello"])
