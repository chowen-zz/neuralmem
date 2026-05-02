"""OpenAI LLM Extractor — requires neuralmem[openai]."""
from __future__ import annotations

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, NeuralMemError
from neuralmem.extraction.base_llm_extractor import BaseLLMExtractor


class OpenAIExtractor(BaseLLMExtractor):
    def __init__(self, config: NeuralMemConfig) -> None:
        super().__init__(config)
        try:
            import openai as _openai
        except ImportError as exc:
            raise NeuralMemError(
                "Install neuralmem[openai] to use OpenAIExtractor: pip install 'neuralmem[openai]'"
            ) from exc
        if not config.openai_api_key:
            raise ConfigError("openai_api_key is required for OpenAIExtractor.")
        self._client = _openai.OpenAI(api_key=config.openai_api_key)
        self._model = config.openai_extractor_model

    def _call_llm(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content or "{}"
