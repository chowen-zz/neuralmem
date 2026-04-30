import sys
from unittest.mock import MagicMock, patch
import pytest
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, NeuralMemError


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _mock_openai_sdk(response_text: str = '{"facts": [], "entities": []}'):
    mock = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    mock.OpenAI.return_value.chat.completions.create.return_value.choices = [choice]
    return mock


def test_openai_extractor_missing_api_key_raises():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.extraction.openai_extractor" in sys.modules:
            del sys.modules["neuralmem.extraction.openai_extractor"]
        from neuralmem.extraction.openai_extractor import OpenAIExtractor
        with pytest.raises(ConfigError, match="openai_api_key"):
            OpenAIExtractor(cfg())


def test_openai_extractor_extracts_entities():
    mock_sdk = _mock_openai_sdk(
        '{"facts": ["Alice works at OpenAI"], "entities": [{"name": "Alice", "type": "person"}, {"name": "OpenAI", "type": "project"}]}'
    )
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.extraction.openai_extractor" in sys.modules:
            del sys.modules["neuralmem.extraction.openai_extractor"]
        from neuralmem.extraction.openai_extractor import OpenAIExtractor
        extractor = OpenAIExtractor(cfg(openai_api_key="sk-test"))
        items = extractor.extract("Alice works at OpenAI")
        all_names = [e.name for e in items[0].entities] if items else []
        assert "Alice" in all_names or "OpenAI" in all_names


def test_openai_extractor_fallback_on_llm_error():
    mock_sdk = _mock_openai_sdk()
    mock_sdk.OpenAI.return_value.chat.completions.create.side_effect = Exception("rate limit")
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.extraction.openai_extractor" in sys.modules:
            del sys.modules["neuralmem.extraction.openai_extractor"]
        from neuralmem.extraction.openai_extractor import OpenAIExtractor
        extractor = OpenAIExtractor(cfg(openai_api_key="sk-test"))
        items = extractor.extract("hello world")
        assert isinstance(items, list)


def test_openai_extractor_uses_configured_model():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.extraction.openai_extractor" in sys.modules:
            del sys.modules["neuralmem.extraction.openai_extractor"]
        from neuralmem.extraction.openai_extractor import OpenAIExtractor
        extractor = OpenAIExtractor(cfg(openai_api_key="sk-test"))
        extractor.extract("test content")
        call_kwargs = mock_sdk.OpenAI.return_value.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"
