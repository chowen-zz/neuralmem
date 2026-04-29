"""CrossEncoderReranker 测试（mock sentence-transformers）"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from neuralmem.retrieval.reranker import CrossEncoderReranker
from neuralmem.core.types import Memory


def _mem(content: str) -> Memory:
    return Memory(content=content)


def test_reranker_fallback_preserves_ids():
    reranker = CrossEncoderReranker()
    reranker._model = False  # 模拟不可用
    m1 = _mem("first memory content")
    m2 = _mem("second memory content")
    result = reranker.rerank("test", [(m1, 0.8), (m2, 0.6)])
    assert result[0][0] == m1.id
    assert result[1][0] == m2.id


def test_reranker_single_candidate_skips_model():
    reranker = CrossEncoderReranker()
    mock_model = MagicMock()
    reranker._model = mock_model
    result = reranker.rerank("query", [(_mem("only one"), 0.9)])
    mock_model.predict.assert_not_called()
    assert len(result) == 1


def test_reranker_empty_candidates():
    reranker = CrossEncoderReranker()
    result = reranker.rerank("query", [])
    assert result == []


def test_reranker_changes_order_when_model_available():
    reranker = CrossEncoderReranker()
    mock_model = MagicMock()
    import numpy as np
    mock_model.predict.return_value = np.array([0.3, 0.9])
    reranker._model = mock_model

    m1 = _mem("less relevant content")
    m2 = _mem("more relevant content")
    result = reranker.rerank("query", [(m1, 0.9), (m2, 0.5)])

    assert result[0][0] == m2.id
    assert result[1][0] == m1.id


def test_reranker_passes_correct_pairs_to_model():
    reranker = CrossEncoderReranker()
    mock_model = MagicMock()
    import numpy as np
    mock_model.predict.return_value = np.array([0.5, 0.8])
    reranker._model = mock_model

    m1 = _mem("content one")
    m2 = _mem("content two")
    reranker.rerank("my query", [(m1, 0.9), (m2, 0.7)])

    call_args = mock_model.predict.call_args[0][0]
    assert call_args == [("my query", "content one"), ("my query", "content two")]


def test_reranker_model_not_loaded_on_init():
    reranker = CrossEncoderReranker()
    assert reranker._model is None
