"""Tests for embedding and extractor registry routing."""
import sys
from unittest.mock import MagicMock, patch
import pytest
from neuralmem.core.config import NeuralMemConfig
from neuralmem.embedding.local import LocalEmbedding


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def test_registry_default_returns_local():
    from neuralmem.embedding.registry import get_embedder
    result = get_embedder(cfg())
    assert isinstance(result, LocalEmbedding)


def test_registry_unknown_provider_falls_back_to_local():
    from neuralmem.embedding.registry import get_embedder
    result = get_embedder(cfg(embedding_provider="nonexistent"))
    assert isinstance(result, LocalEmbedding)


def test_registry_openai_instantiates_openai_embedding():
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_openai}):
        if "neuralmem.embedding.openai" in sys.modules:
            del sys.modules["neuralmem.embedding.openai"]
        from neuralmem.embedding.registry import get_embedder
        # reimport registry to pick up fresh openai module
        if "neuralmem.embedding.registry" in sys.modules:
            del sys.modules["neuralmem.embedding.registry"]
        from neuralmem.embedding.registry import get_embedder as fresh_get
        from neuralmem.embedding.openai import OpenAIEmbedding
        result = fresh_get(cfg(embedding_provider="openai", openai_api_key="sk-test"))
        assert isinstance(result, OpenAIEmbedding)
