"""LocalEmbedding 扩展测试（不实际加载模型）"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import EmbeddingError
from neuralmem.embedding.base import EmbeddingBackend
from neuralmem.embedding.local import _KNOWN_DIMS, LocalEmbedding


def test_local_embedding_inherits_backend():
    cfg = NeuralMemConfig()
    emb = LocalEmbedding(cfg)
    assert isinstance(emb, EmbeddingBackend)


def test_dimension_known_model():
    cfg = NeuralMemConfig(embedding_model="all-MiniLM-L6-v2")
    emb = LocalEmbedding(cfg)
    assert emb.dimension == 384


def test_dimension_unknown_model():
    cfg = NeuralMemConfig(embedding_model="unknown-model-xyz")
    emb = LocalEmbedding(cfg)
    assert emb.dimension == 384  # fallback


def test_known_dims_map():
    assert "all-MiniLM-L6-v2" in _KNOWN_DIMS
    assert _KNOWN_DIMS["all-MiniLM-L6-v2"] == 384


def test_model_not_loaded_on_init():
    cfg = NeuralMemConfig()
    emb = LocalEmbedding(cfg)
    assert emb._model is None


def test_encode_empty_list():
    cfg = NeuralMemConfig()
    emb = LocalEmbedding(cfg)
    result = emb.encode([])
    assert result == []


def test_encode_with_mock_model():
    cfg = NeuralMemConfig()
    emb = LocalEmbedding(cfg)
    mock_model = MagicMock()
    mock_model.embed.return_value = iter([np.array([0.1, 0.2, 0.3, 0.4])])
    emb._model = mock_model
    result = emb.encode(["test text"])
    assert len(result) == 1
    assert len(result[0]) == 4
    assert result[0][0] == pytest.approx(0.1)


def test_encode_one_with_mock_model():
    cfg = NeuralMemConfig()
    emb = LocalEmbedding(cfg)
    mock_model = MagicMock()
    mock_model.embed.return_value = iter([np.array([0.5, 0.5, 0.5, 0.5])])
    emb._model = mock_model
    result = emb.encode_one("single text")
    assert isinstance(result, list)
    assert len(result) == 4


def test_encode_multiple_texts_with_mock_model():
    cfg = NeuralMemConfig()
    emb = LocalEmbedding(cfg)
    mock_model = MagicMock()
    mock_model.embed.return_value = iter([
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0.5, 0.6, 0.7, 0.8]),
    ])
    emb._model = mock_model
    result = emb.encode(["text one", "text two"])
    assert len(result) == 2


def test_encode_raises_embedding_error_when_model_load_fails():
    """当 fastembed 不可用时，_get_model 应抛出 EmbeddingError"""
    cfg = NeuralMemConfig()
    emb = LocalEmbedding(cfg)
    # 模拟 _get_model 抛出 EmbeddingError
    with patch.object(emb, "_get_model", side_effect=EmbeddingError("no fastembed")):
        with pytest.raises(EmbeddingError):
            emb.encode(["test"])


def test_encode_wraps_unexpected_exception():
    """非 EmbeddingError 异常应被包装为 EmbeddingError"""
    cfg = NeuralMemConfig()
    emb = LocalEmbedding(cfg)
    mock_model = MagicMock()
    mock_model.embed.side_effect = RuntimeError("unexpected")
    emb._model = mock_model
    with pytest.raises(EmbeddingError):
        emb.encode(["test"])


def test_known_dims_bge_small():
    assert "BAAI/bge-small-en-v1.5" in _KNOWN_DIMS
    assert _KNOWN_DIMS["BAAI/bge-small-en-v1.5"] == 384
