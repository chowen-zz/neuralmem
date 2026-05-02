from unittest.mock import MagicMock, patch

import pytest

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _mock_response(data):
    mock = MagicMock()
    mock.json.return_value = data
    mock.raise_for_status.return_value = None
    return mock


def test_hf_missing_api_key_raises():
    from neuralmem.embedding.huggingface import HuggingFaceEmbedding
    with pytest.raises(ConfigError, match="hf_api_key"):
        HuggingFaceEmbedding(cfg())


def test_hf_encode_flat_response():
    resp = _mock_response([[0.1] * 1024])
    with patch("neuralmem.embedding.huggingface.httpx.post", return_value=resp):
        from neuralmem.embedding.huggingface import HuggingFaceEmbedding
        embedder = HuggingFaceEmbedding(cfg(hf_api_key="hf-test"))
        result = embedder.encode(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 1024


def test_hf_encode_nested_response_mean_pools():
    """HF 有些模型返回 [n_texts, n_tokens, dim]，需要 mean pool。"""
    token_vecs = [[1.0, 0.0], [0.0, 1.0]]  # 2 tokens, dim=2
    resp = _mock_response([token_vecs])
    with patch("neuralmem.embedding.huggingface.httpx.post", return_value=resp):
        from neuralmem.embedding.huggingface import HuggingFaceEmbedding
        embedder = HuggingFaceEmbedding(cfg(hf_api_key="hf-test"))
        result = embedder.encode(["hello"])
        assert result[0] == pytest.approx([0.5, 0.5])


def test_hf_encode_empty_returns_empty():
    from neuralmem.embedding.huggingface import HuggingFaceEmbedding
    embedder = HuggingFaceEmbedding(cfg(hf_api_key="hf-test"))
    assert embedder.encode([]) == []


def test_hf_http_error_raises_embedding_error():
    import httpx
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "Service Unavailable"
    with patch(
        "neuralmem.embedding.huggingface.httpx.post",
        side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=mock_resp),
    ):
        from neuralmem.embedding.huggingface import HuggingFaceEmbedding
        embedder = HuggingFaceEmbedding(cfg(hf_api_key="hf-test"))
        with pytest.raises(EmbeddingError, match="HuggingFace API error 503"):
            embedder.encode(["hello"])
