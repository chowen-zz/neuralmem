"""Ollama-backed LLM extractor — inherits BaseLLMExtractor."""
from __future__ import annotations

import logging

from neuralmem.core.config import NeuralMemConfig
from neuralmem.extraction.base_llm_extractor import BaseLLMExtractor

_logger = logging.getLogger(__name__)


class LLMExtractor(BaseLLMExtractor):
    """Ollama LLM 增强提取器。enable_llm_extraction=True 或 llm_extractor='ollama' 时使用。"""

    def __init__(self, config: NeuralMemConfig) -> None:
        super().__init__(config)
        self._available: bool | None = None

    def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import httpx
            resp = httpx.get(f"{self._config.ollama_url}/api/tags", timeout=2.0)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
            _logger.info(
                "Ollama not available at %s, using rule extractor.",
                self._config.ollama_url,
            )
        return self._available

    def _call_llm(self, prompt: str) -> str:
        if not self._check_available():
            raise RuntimeError("Ollama not available — check NEURALMEM_OLLAMA_URL")
        import httpx
        resp = httpx.post(
            f"{self._config.ollama_url}/api/generate",
            json={"model": self._config.ollama_model, "prompt": prompt, "stream": False},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json().get("response", "{}")
