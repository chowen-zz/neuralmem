"""EmbedderProtocol 契约测试"""
from __future__ import annotations

from neuralmem.core.protocols import EmbedderProtocol


def test_mock_embedder_satisfies_protocol(mock_embedder):
    assert isinstance(mock_embedder, EmbedderProtocol)


def test_embedder_dimension(mock_embedder):
    assert mock_embedder.dimension == 4


def test_embedder_encode_returns_list(mock_embedder):
    result = mock_embedder.encode(["hello world"])
    assert isinstance(result, list)
    assert len(result) == 1
    assert len(result[0]) == 4


def test_embedder_encode_one(mock_embedder):
    result = mock_embedder.encode_one("hello")
    assert isinstance(result, list)
    assert len(result) == 4
