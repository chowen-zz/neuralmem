"""Vision LLM extractor — GPT-4V / Claude 3 / Gemini Pro Vision support.

Provides a unified interface for image understanding, structured extraction,
and visual QA across multiple cloud vision-LLM providers.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

_logger = logging.getLogger(__name__)

# Graceful dependency checks --------------------------------------------------
try:
    import openai  # type: ignore[import-untyped]

    _HAS_OPENAI = True
except ImportError:
    openai = None  # type: ignore[misc]
    _HAS_OPENAI = False

try:
    import anthropic  # type: ignore[import-untyped]

    _HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore[misc]
    _HAS_ANTHROPIC = False

try:
    import google.generativeai as genai  # type: ignore[import-untyped]

    _HAS_GEMINI = True
except ImportError:
    genai = None  # type: ignore[misc]
    _HAS_GEMINI = False


@dataclass
class VisionLLMResult:
    """Structured result from a vision LLM call.

    Attributes:
        text: The raw textual response from the model.
        structured: Parsed JSON dict if the response was valid JSON.
        provider: Which provider generated the result.
        model: Model name used.
        usage_tokens: Token usage metadata (if available).
    """

    text: str = ""
    structured: dict[str, Any] = field(default_factory=dict)
    provider: str = ""
    model: str = ""
    usage_tokens: dict[str, int] = field(default_factory=dict)


class VisionLLMExtractor:
    """Unified vision-LLM extractor supporting GPT-4V, Claude 3, and Gemini Pro Vision.

    Args:
        provider: One of ``openai``, ``anthropic``, ``gemini``.
        model: Model identifier. Defaults are provider-specific.
        api_key: API key for the provider. Falls back to env vars.
        max_tokens: Max completion tokens. Defaults to 1024.
        temperature: Sampling temperature. Defaults to 0.2.
        top_p: Nucleus sampling parameter. Defaults to 1.0.
    """

    _PROVIDER_MODELS: dict[str, str] = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-opus-20240229",
        "gemini": "gemini-1.5-pro-vision",
    }

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 1.0,
    ) -> None:
        self.provider = provider.lower().strip()
        if self.provider not in self._PROVIDER_MODELS:
            raise ValueError(
                f"Unsupported provider {provider!r}. "
                f"Choose from {list(self._PROVIDER_MODELS.keys())}."
            )
        self.model = model or self._PROVIDER_MODELS[self.provider]
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        self._client: Any | None = None
        self._init_client()

    # ------------------------------------------------------------------ #
    # Client initialisation
    # ------------------------------------------------------------------ #

    def _init_client(self) -> None:
        """Lazy-initialise the underlying SDK client."""
        key = self.api_key or self._env_key()
        if not key:
            _logger.debug("No API key available for %s; client left uninitialised.", self.provider)
            return
        if self.provider == "openai":
            if _HAS_OPENAI and openai is not None:
                self._client = openai.OpenAI(api_key=key)
        elif self.provider == "anthropic":
            if _HAS_ANTHROPIC and anthropic is not None:
                self._client = anthropic.Anthropic(api_key=key)
        elif self.provider == "gemini":
            if _HAS_GEMINI and genai is not None:
                genai.configure(api_key=key)
                self._client = genai

    def _env_key(self) -> str | None:
        """Return the API key from a well-known environment variable."""
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }
        return os.environ.get(env_map.get(self.provider, ""))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def describe_image(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
        prompt: str | None = None,
    ) -> VisionLLMResult:
        """Generate a natural-language description of an image.

        Args:
            image_bytes: Raw image bytes.
            mime_type: MIME type of the image.
            prompt: Optional custom prompt. Defaults to a generic description prompt.

        Returns:
            VisionLLMResult with the model's description.
        """
        system_prompt = prompt or (
            "Describe this image in detail. Include objects, people, text, "
            "setting, colors, and any notable visual elements."
        )
        return self._call_provider(image_bytes, mime_type, system_prompt)

    def extract_structured(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
        schema: dict[str, Any] | None = None,
    ) -> VisionLLMResult:
        """Extract structured data from an image according to a JSON schema.

        Args:
            image_bytes: Raw image bytes.
            mime_type: MIME type of the image.
            schema: JSON schema dict describing the desired output shape.

        Returns:
            VisionLLMResult with ``structured`` populated when valid JSON is returned.
        """
        schema = schema or {
            "type": "object",
            "properties": {
                "objects": {"type": "array", "items": {"type": "string"}},
                "text_in_image": {"type": "string"},
                "scene": {"type": "string"},
                "confidence": {"type": "number"},
            },
        }
        prompt = (
            "Analyze this image and return a JSON object matching the schema below. "
            "Respond with ONLY valid JSON, no markdown fences.\n\n"
            f"Schema: {json.dumps(schema, indent=2)}"
        )
        result = self._call_provider(image_bytes, mime_type, prompt)
        result.structured = self._safe_parse_json(result.text)
        return result

    def visual_qa(
        self,
        image_bytes: bytes,
        question: str,
        mime_type: str = "image/png",
    ) -> VisionLLMResult:
        """Answer a question about the contents of an image.

        Args:
            image_bytes: Raw image bytes.
            question: The question to ask about the image.
            mime_type: MIME type of the image.

        Returns:
            VisionLLMResult with the answer.
        """
        prompt = (
            f"Answer the following question about this image concisely and accurately.\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        return self._call_provider(image_bytes, mime_type, prompt)

    def batch_describe(
        self,
        images: Sequence[tuple[bytes, str]],
        prompt: str | None = None,
    ) -> list[VisionLLMResult]:
        """Describe multiple images in one batch.

        Args:
            images: Sequence of (image_bytes, mime_type) tuples.
            prompt: Optional shared prompt.

        Returns:
            List of VisionLLMResult, one per image.
        """
        return [self.describe_image(b, m, prompt) for b, m in images]

    # ------------------------------------------------------------------ #
    # Provider-specific calling
    # ------------------------------------------------------------------ #

    def _call_provider(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> VisionLLMResult:
        """Dispatch to the correct provider implementation."""
        if self.provider == "openai":
            return self._call_openai(image_bytes, mime_type, prompt)
        if self.provider == "anthropic":
            return self._call_anthropic(image_bytes, mime_type, prompt)
        if self.provider == "gemini":
            return self._call_gemini(image_bytes, mime_type, prompt)
        raise RuntimeError(f"Provider {self.provider!r} not implemented.")

    def _call_openai(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> VisionLLMResult:
        """Call OpenAI GPT-4V / GPT-4o vision."""
        if self._client is None:
            raise RuntimeError("OpenAI client is not initialised.")
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            text = resp.choices[0].message.content or ""
            usage: dict[str, int] = {}
            if resp.usage:
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                    "total_tokens": getattr(resp.usage, "total_tokens", 0),
                }
            return VisionLLMResult(
                text=text,
                provider="openai",
                model=self.model,
                usage_tokens=usage,
            )
        except Exception as exc:
            _logger.error("OpenAI vision call failed: %s", exc)
            raise

    def _call_anthropic(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> VisionLLMResult:
        """Call Anthropic Claude 3 vision."""
        if self._client is None:
            raise RuntimeError("Anthropic client is not initialised.")
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        media_type = mime_type

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.temperature,
                top_p=self.top_p,
            )
            text = ""
            if resp.content:
                text = resp.content[0].text if hasattr(resp.content[0], "text") else str(resp.content[0])
            usage: dict[str, int] = {}
            if resp.usage:
                usage = {
                    "input_tokens": getattr(resp.usage, "input_tokens", 0),
                    "output_tokens": getattr(resp.usage, "output_tokens", 0),
                }
            return VisionLLMResult(
                text=text,
                provider="anthropic",
                model=self.model,
                usage_tokens=usage,
            )
        except Exception as exc:
            _logger.error("Anthropic vision call failed: %s", exc)
            raise

    def _call_gemini(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> VisionLLMResult:
        """Call Google Gemini Pro Vision."""
        if self._client is None or genai is None:
            raise RuntimeError("Gemini client is not initialised.")
        try:
            model = genai.GenerativeModel(self.model)
            # Gemini accepts PIL Image or base64 data URI
            parts = [
                {"mime_type": mime_type, "data": image_bytes},
                prompt,
            ]
            resp = model.generate_content(parts)
            text = ""
            if resp.text:
                text = resp.text
            usage: dict[str, int] = {}
            if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                um = resp.usage_metadata
                usage = {
                    "prompt_token_count": getattr(um, "prompt_token_count", 0),
                    "candidates_token_count": getattr(um, "candidates_token_count", 0),
                    "total_token_count": getattr(um, "total_token_count", 0),
                }
            return VisionLLMResult(
                text=text,
                provider="gemini",
                model=self.model,
                usage_tokens=usage,
            )
        except Exception as exc:
            _logger.error("Gemini vision call failed: %s", exc)
            raise

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_parse_json(text: str) -> dict[str, Any]:
        """Attempt to parse JSON from a model response, stripping markdown fences."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            _logger.warning("Failed to parse JSON from model response.")
            return {}

    @classmethod
    def available_providers(cls) -> list[str]:
        """Return a list of providers whose SDKs are installed."""
        available: list[str] = []
        if _HAS_OPENAI:
            available.append("openai")
        if _HAS_ANTHROPIC:
            available.append("anthropic")
        if _HAS_GEMINI:
            available.append("gemini")
        return available
