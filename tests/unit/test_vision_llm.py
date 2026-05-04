"""VisionLLMExtractor unit tests — all mock-based, no real API calls."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from neuralmem.multimodal.vision_llm import (
    VisionLLMExtractor,
    VisionLLMResult,
)


# --------------------------------------------------------------------------- #
# Constructor / configuration
# --------------------------------------------------------------------------- #

def test_constructor_defaults_openai():
    ext = VisionLLMExtractor(provider="openai")
    assert ext.provider == "openai"
    assert ext.model == "gpt-4o"
    assert ext.max_tokens == 1024
    assert ext.temperature == 0.2


def test_constructor_defaults_anthropic():
    ext = VisionLLMExtractor(provider="anthropic")
    assert ext.provider == "anthropic"
    assert ext.model == "claude-3-opus-20240229"


def test_constructor_defaults_gemini():
    ext = VisionLLMExtractor(provider="gemini")
    assert ext.provider == "gemini"
    assert ext.model == "gemini-1.5-pro-vision"


def test_constructor_custom_model():
    ext = VisionLLMExtractor(provider="openai", model="gpt-4-turbo")
    assert ext.model == "gpt-4-turbo"


def test_constructor_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        VisionLLMExtractor(provider="unknown")


# --------------------------------------------------------------------------- #
# describe_image — OpenAI mock
# --------------------------------------------------------------------------- #

def test_describe_image_openai_mock():
    ext = VisionLLMExtractor(provider="openai")

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "A red apple on a wooden table."
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 120
    mock_resp.usage = mock_usage

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    ext._client = mock_client

    result = ext.describe_image(b"fake_image", mime_type="image/jpeg")

    assert isinstance(result, VisionLLMResult)
    assert result.text == "A red apple on a wooden table."
    assert result.provider == "openai"
    assert result.usage_tokens == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    mock_client.chat.completions.create.assert_called_once()


# --------------------------------------------------------------------------- #
# describe_image — Anthropic mock
# --------------------------------------------------------------------------- #

def test_describe_image_anthropic_mock():
    ext = VisionLLMExtractor(provider="anthropic")

    mock_content = MagicMock()
    mock_content.text = "A scenic mountain landscape."
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    mock_usage = MagicMock()
    mock_usage.input_tokens = 80
    mock_usage.output_tokens = 15
    mock_resp.usage = mock_usage

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_resp
    ext._client = mock_client

    result = ext.describe_image(b"fake_image")

    assert result.text == "A scenic mountain landscape."
    assert result.provider == "anthropic"
    assert result.usage_tokens == {"input_tokens": 80, "output_tokens": 15}
    mock_client.messages.create.assert_called_once()


# --------------------------------------------------------------------------- #
# describe_image — Gemini mock
# --------------------------------------------------------------------------- #

def test_describe_image_gemini_mock():
    ext = VisionLLMExtractor(provider="gemini")

    mock_resp = MagicMock()
    mock_resp.text = "A cat sleeping on a sofa."
    mock_um = MagicMock()
    mock_um.prompt_token_count = 50
    mock_um.candidates_token_count = 10
    mock_um.total_token_count = 60
    mock_resp.usage_metadata = mock_um

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_resp

    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    ext._client = mock_genai

    with patch("neuralmem.multimodal.vision_llm.genai", mock_genai):
        result = ext.describe_image(b"fake_image")

    assert result.text == "A cat sleeping on a sofa."
    assert result.provider == "gemini"
    assert result.usage_tokens == {
        "prompt_token_count": 50,
        "candidates_token_count": 10,
        "total_token_count": 60,
    }


# --------------------------------------------------------------------------- #
# extract_structured
# --------------------------------------------------------------------------- #

def test_extract_structured_openai_mock():
    ext = VisionLLMExtractor(provider="openai")

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = json.dumps(
        {"objects": ["car", "tree"], "text_in_image": "STOP", "scene": "street", "confidence": 0.95}
    )
    mock_resp.usage = None

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    ext._client = mock_client

    result = ext.extract_structured(b"fake_image")

    assert result.structured == {
        "objects": ["car", "tree"],
        "text_in_image": "STOP",
        "scene": "street",
        "confidence": 0.95,
    }


def test_extract_structured_with_markdown_fences():
    ext = VisionLLMExtractor(provider="openai")

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = (
        "```json\n"
        + json.dumps({"scene": "beach", "confidence": 0.88})
        + "\n```"
    )
    mock_resp.usage = None

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    ext._client = mock_client

    result = ext.extract_structured(b"fake_image")
    assert result.structured == {"scene": "beach", "confidence": 0.88}


def test_extract_structured_invalid_json():
    ext = VisionLLMExtractor(provider="openai")

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "not valid json {"
    mock_resp.usage = None

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    ext._client = mock_client

    result = ext.extract_structured(b"fake_image")
    assert result.structured == {}


# --------------------------------------------------------------------------- #
# visual_qa
# --------------------------------------------------------------------------- #

def test_visual_qa_openai_mock():
    ext = VisionLLMExtractor(provider="openai")

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "There are 3 people in the image."
    mock_resp.usage = None

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    ext._client = mock_client

    result = ext.visual_qa(b"fake_image", "How many people are in this image?")

    assert result.text == "There are 3 people in the image."
    assert "How many people" in mock_client.chat.completions.create.call_args[1]["messages"][0]["content"][0]["text"]


# --------------------------------------------------------------------------- #
# batch_describe
# --------------------------------------------------------------------------- #

def test_batch_describe_mock():
    ext = VisionLLMExtractor(provider="openai")

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "Description."
    mock_resp.usage = None

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    ext._client = mock_client

    images = [(b"img1", "image/png"), (b"img2", "image/jpeg")]
    results = ext.batch_describe(images)

    assert len(results) == 2
    assert all(r.text == "Description." for r in results)
    assert mock_client.chat.completions.create.call_count == 2


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def test_safe_parse_json_plain():
    parsed = VisionLLMExtractor._safe_parse_json('{"a": 1}')
    assert parsed == {"a": 1}


def test_safe_parse_json_markdown():
    parsed = VisionLLMExtractor._safe_parse_json("```json\n{\"b\": 2}\n```")
    assert parsed == {"b": 2}


def test_safe_parse_json_invalid():
    parsed = VisionLLMExtractor._safe_parse_json("not json")
    assert parsed == {}


def test_available_providers_mocked():
    with patch("neuralmem.multimodal.vision_llm._HAS_OPENAI", True):
        with patch("neuralmem.multimodal.vision_llm._HAS_ANTHROPIC", False):
            with patch("neuralmem.multimodal.vision_llm._HAS_GEMINI", True):
                assert VisionLLMExtractor.available_providers() == ["openai", "gemini"]


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #

def test_call_provider_unimplemented():
    ext = VisionLLMExtractor(provider="openai")
    ext.provider = "unknown"
    with pytest.raises(RuntimeError, match="not implemented"):
        ext._call_provider(b"img", "image/png", "prompt")


def test_openai_call_no_client():
    ext = VisionLLMExtractor(provider="openai")
    ext._client = None
    with pytest.raises(RuntimeError, match="OpenAI client is not initialised"):
        ext._call_openai(b"img", "image/png", "prompt")


def test_anthropic_call_no_client():
    ext = VisionLLMExtractor(provider="anthropic")
    ext._client = None
    with pytest.raises(RuntimeError, match="Anthropic client is not initialised"):
        ext._call_anthropic(b"img", "image/png", "prompt")


def test_gemini_call_no_client():
    ext = VisionLLMExtractor(provider="gemini")
    ext._client = None
    with pytest.raises(RuntimeError, match="Gemini client is not initialised"):
        ext._call_gemini(b"img", "image/png", "prompt")
