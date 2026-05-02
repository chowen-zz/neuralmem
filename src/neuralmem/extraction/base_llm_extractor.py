"""Shared base for all LLM-backed extractors — subclasses only implement _call_llm()."""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.types import Entity
from neuralmem.extraction.extractor import ExtractedItem, MemoryExtractor

_logger = logging.getLogger(__name__)

_EXTRACT_PROMPT_PREFIX = (
    "Extract key facts and entities from the following text.\n"
    'Return JSON with: {"facts": ["fact1", "fact2"], '
    '"entities": [{"name": "X", "type": "person|project|technology|concept"}]}\n'
    "Text: "
)


class BaseLLMExtractor(ABC):
    def __init__(self, config: NeuralMemConfig) -> None:
        self._config = config
        self._rule_extractor = MemoryExtractor(config)

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API and return raw response string."""
        ...

    def extract(self, content: str, **kwargs: object) -> list[ExtractedItem]:
        try:
            prompt = _EXTRACT_PROMPT_PREFIX + content[:1000] + "\nJSON:"
            raw = self._call_llm(prompt)
            raw = (
                raw.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            data = json.loads(raw)
            items = self._rule_extractor.extract(content, **kwargs)  # type: ignore[arg-type]
            extra_entities = [
                Entity(name=str(e.get("name", ""))[:200], entity_type=e.get("type", "concept"))
                for e in data.get("entities", [])
                if e.get("name")
            ]
            if items and extra_entities:
                first = items[0]
                merged = list({e.name: e for e in first.entities + extra_entities}.values())
                items[0] = ExtractedItem(
                    content=first.content,
                    memory_type=first.memory_type,
                    entities=merged,
                    relations=first.relations,
                    tags=first.tags,
                    importance=first.importance,
                )
            return items
        except Exception as exc:
            _logger.warning("LLM extraction failed (%s), falling back to rules.", exc)
            return self._rule_extractor.extract(content, **kwargs)  # type: ignore[arg-type]
