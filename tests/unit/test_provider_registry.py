"""Tests for embedding and extractor registry routing."""
import sys
from unittest.mock import MagicMock, patch

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
        # reimport registry to pick up fresh openai module
        if "neuralmem.embedding.registry" in sys.modules:
            del sys.modules["neuralmem.embedding.registry"]
        from neuralmem.embedding.openai import OpenAIEmbedding
        from neuralmem.embedding.registry import get_embedder as fresh_get
        result = fresh_get(cfg(embedding_provider="openai", openai_api_key="sk-test"))
        assert isinstance(result, OpenAIEmbedding)


# ==================== Extractor Registry ====================

def test_extractor_registry_default_returns_rule_extractor():
    from neuralmem.extraction.extractor import MemoryExtractor
    from neuralmem.extraction.extractor_registry import get_extractor
    result = get_extractor(cfg())
    assert isinstance(result, MemoryExtractor)


def test_extractor_registry_ollama_returns_llm_extractor():
    from neuralmem.extraction.extractor_registry import get_extractor
    from neuralmem.extraction.llm_extractor import LLMExtractor
    result = get_extractor(cfg(llm_extractor="ollama"))
    assert isinstance(result, LLMExtractor)


def test_extractor_registry_unknown_falls_back_to_rule():
    from neuralmem.extraction.extractor import MemoryExtractor
    from neuralmem.extraction.extractor_registry import get_extractor
    result = get_extractor(cfg(llm_extractor="nonexistent"))
    assert isinstance(result, MemoryExtractor)
