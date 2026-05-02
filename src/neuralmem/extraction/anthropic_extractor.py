"""Anthropic Claude LLM Extractor — requires neuralmem[anthropic]."""
from __future__ import annotations

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, NeuralMemError
from neuralmem.extraction.base_llm_extractor import BaseLLMExtractor


class AnthropicExtractor(BaseLLMExtractor):
    def __init__(self, config: NeuralMemConfig) -> None:
        super().__init__(config)
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise NeuralMemError(
                "Install neuralmem[anthropic] to use AnthropicExtractor:"
                " pip install 'neuralmem[anthropic]'"
            ) from exc
        if not config.anthropic_api_key:
            raise ConfigError("anthropic_api_key is required for AnthropicExtractor.")
        self._client = _anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._model = config.anthropic_model

    def _call_llm(self, prompt: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text if message.content else "{}"
