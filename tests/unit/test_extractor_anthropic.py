import sys
from unittest.mock import MagicMock, patch
import pytest
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, NeuralMemError


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _mock_anthropic_sdk(response_text: str = '{"facts": [], "entities": []}'):
    mock = MagicMock()
    content_block = MagicMock()
    content_block.text = response_text
    mock.Anthropic.return_value.messages.create.return_value.content = [content_block]
    return mock


def test_anthropic_extractor_missing_api_key_raises():
    mock_sdk = _mock_anthropic_sdk()
    with patch.dict(sys.modules, {"anthropic": mock_sdk}):
        if "neuralmem.extraction.anthropic_extractor" in sys.modules:
            del sys.modules["neuralmem.extraction.anthropic_extractor"]
        from neuralmem.extraction.anthropic_extractor import AnthropicExtractor
        with pytest.raises(ConfigError, match="anthropic_api_key"):
            AnthropicExtractor(cfg())


def test_anthropic_extractor_extracts_entities():
    mock_sdk = _mock_anthropic_sdk(
        '{"facts": [], "entities": [{"name": "Claude", "type": "technology"}]}'
    )
    with patch.dict(sys.modules, {"anthropic": mock_sdk}):
        if "neuralmem.extraction.anthropic_extractor" in sys.modules:
            del sys.modules["neuralmem.extraction.anthropic_extractor"]
        from neuralmem.extraction.anthropic_extractor import AnthropicExtractor
        extractor = AnthropicExtractor(cfg(anthropic_api_key="sk-ant-test"))
        items = extractor.extract("Claude is made by Anthropic")
        all_names = [e.name for e in items[0].entities] if items else []
        assert "Claude" in all_names


def test_anthropic_extractor_fallback_on_llm_error():
    mock_sdk = _mock_anthropic_sdk()
    mock_sdk.Anthropic.return_value.messages.create.side_effect = Exception("overloaded")
    with patch.dict(sys.modules, {"anthropic": mock_sdk}):
        if "neuralmem.extraction.anthropic_extractor" in sys.modules:
            del sys.modules["neuralmem.extraction.anthropic_extractor"]
        from neuralmem.extraction.anthropic_extractor import AnthropicExtractor
        extractor = AnthropicExtractor(cfg(anthropic_api_key="sk-ant-test"))
        items = extractor.extract("hello world")
        assert isinstance(items, list)


def test_anthropic_extractor_uses_configured_model():
    mock_sdk = _mock_anthropic_sdk()
    with patch.dict(sys.modules, {"anthropic": mock_sdk}):
        if "neuralmem.extraction.anthropic_extractor" in sys.modules:
            del sys.modules["neuralmem.extraction.anthropic_extractor"]
        from neuralmem.extraction.anthropic_extractor import AnthropicExtractor
        extractor = AnthropicExtractor(cfg(anthropic_api_key="sk-ant-test"))
        extractor.extract("test")
        call_kwargs = mock_sdk.Anthropic.return_value.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-haiku-4-5-20251001"
