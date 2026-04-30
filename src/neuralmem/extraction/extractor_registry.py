"""Extractor factory — maps config.llm_extractor to a concrete extractor instance."""
from __future__ import annotations

from neuralmem.core.config import NeuralMemConfig
from neuralmem.extraction.extractor import MemoryExtractor


def get_extractor(config: NeuralMemConfig) -> MemoryExtractor:
    """Return the configured LLM extractor, falling back to rule-based on unknown values."""
    match config.llm_extractor:
        case "openai":
            from neuralmem.extraction.openai_extractor import OpenAIExtractor
            return OpenAIExtractor(config)  # type: ignore[return-value]
        case "anthropic":
            from neuralmem.extraction.anthropic_extractor import AnthropicExtractor
            return AnthropicExtractor(config)  # type: ignore[return-value]
        case "ollama":
            from neuralmem.extraction.llm_extractor import LLMExtractor
            return LLMExtractor(config)  # type: ignore[return-value]
        case _:
            return MemoryExtractor(config)
