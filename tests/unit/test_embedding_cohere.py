import sys
from unittest.mock import MagicMock, patch

import pytest

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _mock_cohere_sdk():
    mock = MagicMock()
    mock_resp = MagicMock()
    mock_resp.embeddings.float_ = [[0.1] * 1024]
    mock.ClientV2.return_value.embed.return_value = mock_resp
    return mock


def test_cohere_missing_api_key_raises():
    mock_sdk = _mock_cohere_sdk()
    with patch.dict(sys.modules, {"cohere": mock_sdk}):
        if "neuralmem.embedding.cohere" in sys.modules:
            del sys.modules["neuralmem.embedding.cohere"]
        from neuralmem.embedding.cohere import CohereEmbedding
        with pytest.raises(ConfigError, match="cohere_api_key"):
            CohereEmbedding(cfg())


def test_cohere_encode_returns_vectors():
    mock_sdk = _mock_cohere_sdk()
    with patch.dict(sys.modules, {"cohere": mock_sdk}):
        if "neuralmem.embedding.cohere" in sys.modules:
            del sys.modules["neuralmem.embedding.cohere"]
        from neuralmem.embedding.cohere import CohereEmbedding
        embedder = CohereEmbedding(cfg(cohere_api_key="test-key"))
        result = embedder.encode(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 1024


def test_cohere_encode_empty_returns_empty():
    mock_sdk = _mock_cohere_sdk()
    with patch.dict(sys.modules, {"cohere": mock_sdk}):
        if "neuralmem.embedding.cohere" in sys.modules:
            del sys.modules["neuralmem.embedding.cohere"]
        from neuralmem.embedding.cohere import CohereEmbedding
        embedder = CohereEmbedding(cfg(cohere_api_key="test-key"))
        assert embedder.encode([]) == []


def test_cohere_api_failure_raises_embedding_error():
    mock_sdk = _mock_cohere_sdk()
    mock_sdk.ClientV2.return_value.embed.side_effect = Exception("quota exceeded")
    with patch.dict(sys.modules, {"cohere": mock_sdk}):
        if "neuralmem.embedding.cohere" in sys.modules:
            del sys.modules["neuralmem.embedding.cohere"]
        from neuralmem.embedding.cohere import CohereEmbedding
        embedder = CohereEmbedding(cfg(cohere_api_key="test-key"))
        with pytest.raises(EmbeddingError, match="Cohere embedding failed"):
            embedder.encode(["hello"])


def test_cohere_dimension_light_model():
    mock_sdk = _mock_cohere_sdk()
    with patch.dict(sys.modules, {"cohere": mock_sdk}):
        if "neuralmem.embedding.cohere" in sys.modules:
            del sys.modules["neuralmem.embedding.cohere"]
        from neuralmem.embedding.cohere import CohereEmbedding
        e = CohereEmbedding(
            cfg(cohere_api_key="k", cohere_embedding_model="embed-multilingual-light-v3.0")
        )
        assert e.dimension == 384
