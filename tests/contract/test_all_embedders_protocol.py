"""Verify key embedding backends satisfy EmbedderProtocol at the interface level."""
import sys
from unittest.mock import MagicMock, patch
import pytest
from neuralmem.core.protocols import EmbedderProtocol
from neuralmem.core.config import NeuralMemConfig


def _cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _make_local_embedder():
    from neuralmem.embedding.local import LocalEmbedding
    return LocalEmbedding(_cfg())


def _make_hf_embedder():
    from neuralmem.embedding.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(_cfg(hf_api_key="hf-test"))


def _make_cohere_embedder():
    mock = MagicMock()
    mock.ClientV2.return_value.embed.return_value.embeddings.float_ = [[0.1] * 1024]
    with patch.dict(sys.modules, {"cohere": mock}):
        if "neuralmem.embedding.cohere" in sys.modules:
            del sys.modules["neuralmem.embedding.cohere"]
        from neuralmem.embedding.cohere import CohereEmbedding
        return CohereEmbedding(_cfg(cohere_api_key="ck-test"))


def _make_gemini_embedder():
    mock = MagicMock()
    mock.embed_content.return_value = {"embedding": [0.1] * 768}
    mock_google = MagicMock()
    mock_google.generativeai = mock
    with patch.dict(sys.modules, {"google": mock_google, "google.generativeai": mock}):
        if "neuralmem.embedding.gemini" in sys.modules:
            del sys.modules["neuralmem.embedding.gemini"]
        from neuralmem.embedding.gemini import GeminiEmbedding
        return GeminiEmbedding(_cfg(gemini_api_key="gk-test"))


@pytest.mark.parametrize("factory", [
    _make_local_embedder,
    _make_hf_embedder,
    _make_cohere_embedder,
    _make_gemini_embedder,
])
def test_embedder_satisfies_protocol(factory):
    """Each backend must satisfy EmbedderProtocol (runtime-checkable)."""
    embedder = factory()
    assert isinstance(embedder, EmbedderProtocol), (
        f"{type(embedder).__name__} does not satisfy EmbedderProtocol"
    )
    assert isinstance(embedder.dimension, int)
    assert embedder.dimension > 0
    assert hasattr(embedder, "encode")
    assert hasattr(embedder, "encode_one")
    assert embedder.encode([]) == []
